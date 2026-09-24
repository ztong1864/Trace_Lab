"""Chunked GP planner for large finite (categorical) lab design spaces.

Atlas builds the full Cartesian product of options as Python lists and filters it
one candidate at a time, which does not finish on spaces with millions of
combinations. This planner instead:

1. numbers every combination (mixed-radix index) and decodes one chunk of
   indices at a time with numpy, so the full product is never materialized;
2. encodes each option through its numeric descriptors (min-max scaled) or a
   one-hot block when the variable has none;
3. fits one GP per ask and scores every legal combination chunk by chunk with
   analytic EI or UCB from the exact latent posterior, keeping a running top-M
   plus the best candidate for every (variable, option) pair. Spaces larger than
   `max_scan_size` are scored on a seeded random subsample instead, and say so;
4. picks the batch from that pool with a kriging-believer update of the posterior
   covariance (fixed hyperparameters), requiring picks to differ from each other
   in at least `min_changed_variables` variables while the pool allows it.

`acquisition_function: ei_ucb` builds an EI batch and a UCB batch from the same
scan, merges and dedupes them, ranks the merged set by the GP's predicted mean
and keeps the top N; each pick records which acquisition(s) proposed it.

With fewer than two distinct completed results there is nothing to fit, so the
batch is a seeded random initial design (reported as a planner warning, not a
failure). In full-scan mode the first pick is the global acquisition maximum,
up to float32 rounding at the finalist cutoff.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Callable, Iterator

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from olympus.objects import ParameterVector

from chem_agent_bo.bo.base import BasePlanner, ExclusionConstraint, normalize_acquisition_type


# "ei_ucb": build an EI batch and a UCB batch, merge, dedupe, rank by predicted mean.
CHUNKED_GP_ACQUISITION_TYPES = ("ei", "ucb", "ei_ucb")


class ChunkedGPPlanner(BasePlanner):
    """Finite-space GP planner that scans combinations in bounded-memory chunks."""

    def __init__(
        self,
        *,
        seed: int = 7,
        goal: str = "maximize",
        known_constraints: list[Callable[[Any], bool]] | None = None,
        use_descriptors: bool = False,
        acquisition_type: str = "ei",
        chunk_size: int = 100_000,
        finalist_count: int = 2_000,
        max_scan_size: int = 50_000_000,
        min_changed_variables: int = 2,
        ucb_beta: float = 2.0,
        score_dtype: torch.dtype = torch.float32,
    ) -> None:
        self._seed = int(seed)
        self._goal = str(goal).strip().lower()
        self._known_constraints = list(known_constraints or [])
        self._use_descriptors = bool(use_descriptors)
        self._acquisition_type = normalize_acquisition_type(
            acquisition_type,
            supported=CHUNKED_GP_ACQUISITION_TYPES,
            planner_name="chunked_gp",
        )
        self._score_kinds = (
            ["ei", "ucb"] if self._acquisition_type == "ei_ucb" else [self._acquisition_type]
        )
        self._chunk_size = max(1, int(chunk_size))
        self._finalist_count = max(1, int(finalist_count))
        self._max_scan_size = max(1, int(max_scan_size))
        self._min_changed_variables = max(1, int(min_changed_variables))
        self._ucb_beta = float(ucb_beta)
        self._score_dtype = score_dtype
        self._last_stats: dict[str, Any] = {}

    def reset(self) -> None:
        self._last_stats = {}

    def set_known_constraints(
        self,
        known_constraints: list[Callable[[Any], bool]] | None,
        *,
        signature: str | None = None,  # noqa: ARG002
    ) -> None:
        self._known_constraints = list(known_constraints or [])

    def suggest(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
        known_constraints: list[Callable[[Any], bool]] | None = None,
        known_constraints_signature: str | None = None,
    ):  # noqa: ANN201
        return self.suggest_shortlist(
            observations=observations,
            subspace=subspace,
            shortlist_size=1,
            known_constraints=known_constraints,
            known_constraints_signature=known_constraints_signature,
        )

    def suggest_shortlist(
        self,
        observations,  # noqa: ANN001
        subspace,  # noqa: ANN001
        shortlist_size: int,
        known_constraints: list[Callable[[Any], bool]] | None = None,
        known_constraints_signature: str | None = None,  # noqa: ARG002
    ):  # noqa: ANN201
        started = time.perf_counter()
        if known_constraints is not None:
            self._known_constraints = list(known_constraints)
        size = max(1, int(shortlist_size))
        space = self._encode_space(subspace)
        exclusion_idx, post_filters = self._split_constraints(space)
        train_idx, train_y, dropped = self._training_data(observations, space)
        excluded = np.union1d(exclusion_idx, train_idx)
        warnings: list[str] = []
        if dropped:
            warnings.append(
                f"{dropped} completed observation(s) use option values outside the current "
                "design space and were left out of the GP fit."
            )
        stats: dict[str, Any] = {
            "space_size": int(space["size"]),
            "excluded_count": int(excluded.size),
            "train_count": int(train_idx.size),
            "dropped_observation_count": int(dropped),
            "feature_dim": int(space["feature_dim"]),
            "encodings": dict(zip(space["names"], space["encodings"], strict=True)),
            "post_filter_constraint_count": len(post_filters),
        }

        if train_idx.size < 2 or np.unique(train_y).size < 2:
            chosen = self._random_design(space, excluded, post_filters, size)
            warnings.append(
                "Fewer than two completed observations with different results, so the GP "
                "cannot be fitted; this batch is a seeded random initial design."
            )
            stats["selection_mode"] = "random_initial_design"
            stats["pick_acquisitions"] = ["random"] * len(chosen)
        else:
            fit_started = time.perf_counter()
            posterior = _LatentPosterior(self._fit(space, train_idx, train_y), self._score_dtype)
            stats["fit_seconds"] = round(time.perf_counter() - fit_started, 3)
            pools, scanned, scan_mode = self._scan(
                space, posterior, excluded, max(size, self._finalist_count)
            )
            if scan_mode == "subsample":
                warnings.append(
                    f"The design space has {space['size']:,} combinations, more than the "
                    f"{self._max_scan_size:,} scan limit; scored a seeded random subsample "
                    "instead, so the best combination may have been missed."
                )
            batches: dict[str, list[int]] = {}
            post_filtered = 0
            relaxed = False
            for kind in self._score_kinds:
                picks, filtered, kind_relaxed = self._select_batch(
                    space, posterior, pools[kind], size, post_filters, kind
                )
                batches[kind] = picks
                post_filtered += filtered
                relaxed = relaxed or kind_relaxed
            if len(self._score_kinds) == 1:
                chosen = batches[self._acquisition_type]
                pick_acquisitions = [self._acquisition_type] * len(chosen)
                selection_mode = "gp_" + self._acquisition_type
            else:
                chosen, pick_acquisitions = self._merge_rank_by_mean(space, posterior, batches, size)
                selection_mode = "gp_ei_ucb_merge_rank_by_mean"
                stats.update(
                    batch_sizes={kind: len(picks) for kind, picks in batches.items()},
                    overlap_count=len(set(batches["ei"]) & set(batches["ucb"])),
                )
            stats.update(
                selection_mode=selection_mode,
                pick_acquisitions=pick_acquisitions,
                scan_mode=scan_mode,
                scanned_count=int(scanned),
                chunk_size=self._chunk_size,
                pool_count=int(np.unique(np.concatenate(list(pools.values()))).size),
                post_filtered_count=int(post_filtered),
                min_changed_variables=self._min_changed_variables,
                diversity_relaxed=bool(relaxed),
            )
        stats["warnings"] = warnings
        stats["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        self._last_stats = stats
        return [self._to_parameter_vector(index, space, subspace) for index in chosen]

    def planner_diagnostics(self) -> dict[str, Any]:
        return {
            "planner_name": "chunked_gp",
            "planner_family": "bo",
            "supports_shortlist": True,
            "use_descriptors": self._use_descriptors,
            "search_space_type": "finite_categorical",
            "candidate_pool_mode": "chunked_scan",
            "batch_selection": "kriging_believer_min_changed_variables",
            "surrogate_name": "botorch_SingleTaskGP",
            "acquisition_name": self._acquisition_type,
            "encoding_name": "descriptor_or_one_hot",
            **self._last_stats,
        }

    # -- space encoding -------------------------------------------------------------

    def _encode_space(self, subspace) -> dict[str, Any]:  # noqa: ANN001
        names: list[str] = []
        options: list[list[str]] = []
        blocks: list[torch.Tensor] = []
        encodings: list[str] = []
        for param in subspace:
            param_options = [str(option) for option in list(getattr(param, "options", []) or [])]
            if not param_options:
                raise ValueError(
                    f"ChunkedGPPlanner supports finite categorical spaces only; "
                    f"`{param.name}` has no options."
                )
            block = self._descriptor_block(param, len(param_options)) if self._use_descriptors else None
            encodings.append("one_hot" if block is None else "descriptor")
            if block is None:
                block = torch.eye(len(param_options), dtype=torch.double)
            names.append(param.name)
            options.append(param_options)
            blocks.append(block)
        radices = np.array([len(item) for item in options], dtype=np.int64)
        # Last variable varies fastest: stride[v] = product of radices after v.
        strides = np.ones_like(radices)
        for v in range(len(radices) - 2, -1, -1):
            strides[v] = strides[v + 1] * radices[v + 1]
        return {
            "names": names,
            "options": options,
            "lookup": [{option: i for i, option in enumerate(opts)} for opts in options],
            "blocks": blocks,
            "encodings": encodings,
            "radices": radices,
            "strides": strides,
            "size": int(np.prod(radices, dtype=np.int64)),
            "feature_dim": int(sum(block.shape[1] for block in blocks)),
        }

    @staticmethod
    def _descriptor_block(param, option_count: int) -> torch.Tensor | None:  # noqa: ANN001
        rows = list(getattr(param, "descriptors", None) or [])
        if len(rows) != option_count or any(row is None for row in rows):
            return None
        try:
            matrix = np.asarray(rows, dtype=float)
        except (TypeError, ValueError):
            return None
        if matrix.ndim != 2 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
            return None
        low, high = matrix.min(axis=0), matrix.max(axis=0)
        varying = high > low
        if not varying.any():
            return None
        scaled = (matrix[:, varying] - low[varying]) / (high[varying] - low[varying])
        return torch.tensor(scaled, dtype=torch.double)

    @staticmethod
    def _decode(indices: np.ndarray, space: dict[str, Any]) -> np.ndarray:
        return (indices[:, None] // space["strides"][None, :]) % space["radices"][None, :]

    @staticmethod
    def _features(digits: np.ndarray, blocks: list[torch.Tensor]) -> torch.Tensor:
        index = torch.from_numpy(digits)
        return torch.cat([block[index[:, v]] for v, block in enumerate(blocks)], dim=1)

    @staticmethod
    def _index_of(option_indices: list[int], space: dict[str, Any]) -> int:
        return int(np.dot(np.asarray(option_indices, dtype=np.int64), space["strides"]))

    def _candidate(self, index: int, space: dict[str, Any]) -> dict[str, str]:
        digits = self._decode(np.asarray([index], dtype=np.int64), space)[0]
        return {name: space["options"][v][int(digits[v])] for v, name in enumerate(space["names"])}

    # -- data and constraints -------------------------------------------------------

    def _training_data(
        self, observations, space: dict[str, Any]  # noqa: ANN001
    ) -> tuple[np.ndarray, np.ndarray, int]:
        params = observations.get_params(as_array=True)
        values = np.asarray(observations.get_values(as_array=True), dtype=float).reshape(-1)
        indices: list[int] = []
        targets: list[float] = []
        dropped = 0
        for row, value in zip(params, values, strict=True):
            try:
                digits = [space["lookup"][v][str(row[v])] for v in range(len(space["names"]))]
            except KeyError:
                dropped += 1
                continue
            indices.append(self._index_of(digits, space))
            targets.append(float(value))
        return np.asarray(indices, dtype=np.int64), np.asarray(targets, dtype=float), dropped

    def _split_constraints(
        self, space: dict[str, Any]
    ) -> tuple[np.ndarray, list[Callable[[Any], bool]]]:
        """Vectorize exclusion constraints; keep any other constraint as a post-filter."""
        excluded: list[int] = []
        post_filters: list[Callable[[Any], bool]] = []
        for constraint in self._known_constraints:
            if isinstance(constraint, ExclusionConstraint):
                excluded.extend(self._exclusion_indices(constraint, space))
            else:
                post_filters.append(constraint)
        return np.unique(np.asarray(excluded, dtype=np.int64)), post_filters

    @staticmethod
    def _exclusion_indices(constraint: ExclusionConstraint, space: dict[str, Any]) -> list[int]:
        names = space["names"]
        if sorted(constraint.variable_names) != sorted(names):
            raise ValueError(
                f"Exclusion constraint covers variables {constraint.variable_names}, but the "
                f"search space has {names}; its keys cannot be matched by variable name."
            )
        positions = [constraint.variable_names.index(name) for name in names]
        normalized_lookup = [
            {constraint.normalize_value(option): i for option, i in lookup.items()}
            for lookup in space["lookup"]
        ]
        indices: list[int] = []
        for key in constraint.excluded_keys:
            try:
                digits = [normalized_lookup[v][key[p]] for v, p in enumerate(positions)]
            except (KeyError, IndexError):
                continue  # uses a value that is not an option here, so it can't be proposed
            indices.append(int(np.dot(np.asarray(digits, dtype=np.int64), space["strides"])))
        return indices

    def _passes(self, post_filters: list[Callable[[Any], bool]], index: int, space: dict[str, Any]) -> bool:
        candidate = self._candidate(index, space)
        return all(constraint(candidate) for constraint in post_filters)

    def _random_design(
        self,
        space: dict[str, Any],
        excluded: np.ndarray,
        post_filters: list[Callable[[Any], bool]],
        count: int,
    ) -> list[int]:
        rng = np.random.default_rng(self._seed)
        total = space["size"]
        legal = total - int(excluded.size)
        count = min(count, max(legal, 0))
        # Bound the search so rejecting post-filters can't loop forever on a huge space.
        budget = min(legal, 200_000 + 100 * count)
        chosen: list[int] = []
        seen: set[int] = set()
        while len(chosen) < count and len(seen) < budget:
            for index in rng.integers(0, total, size=max(64, 4 * count)).tolist():
                if index in seen or _is_member(excluded, index):
                    continue
                seen.add(index)
                if post_filters and not self._passes(post_filters, index, space):
                    continue
                chosen.append(index)
                if len(chosen) == count or len(seen) >= budget:
                    break
        return chosen

    # -- model and scoring ----------------------------------------------------------

    def _fit(self, space: dict[str, Any], train_idx: np.ndarray, train_y: np.ndarray) -> SingleTaskGP:
        y = -train_y if self._goal == "minimize" else train_y
        train_x = self._features(self._decode(train_idx, space), space["blocks"])
        torch.manual_seed(self._seed)
        model = SingleTaskGP(
            train_x,
            torch.tensor(y, dtype=torch.double).view(-1, 1),
            outcome_transform=Standardize(m=1),
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
        model.eval()
        return model

    def _acquisition(
        self, mean: torch.Tensor, var: torch.Tensor, best_f: torch.Tensor, kind: str
    ) -> torch.Tensor:
        sigma = var.clamp_min(1e-12).sqrt()
        if kind == "ucb":
            return mean + (self._ucb_beta ** 0.5) * sigma
        u = (mean - best_f) / sigma
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        return sigma * (torch.exp(normal.log_prob(u)) + u * normal.cdf(u))

    def _index_chunks(self, space: dict[str, Any], excluded: np.ndarray) -> tuple[Iterator[np.ndarray], str]:
        total = space["size"]
        if total - int(excluded.size) <= self._max_scan_size:

            def full() -> Iterator[np.ndarray]:
                for start in range(0, total, self._chunk_size):
                    stop = min(start + self._chunk_size, total)
                    indices = np.arange(start, stop, dtype=np.int64)
                    lo, hi = np.searchsorted(excluded, [start, stop])
                    if hi > lo:
                        keep = np.ones(indices.size, dtype=bool)
                        keep[excluded[lo:hi] - start] = False
                        indices = indices[keep]
                    yield indices

            return full(), "full"

        rng = np.random.default_rng(self._seed)

        def subsample() -> Iterator[np.ndarray]:
            remaining = self._max_scan_size
            while remaining > 0:
                draw = min(self._chunk_size, remaining)
                remaining -= draw
                indices = np.unique(rng.integers(0, total, size=draw, dtype=np.int64))
                if excluded.size:
                    indices = indices[~np.isin(indices, excluded, assume_unique=True)]
                yield indices

        return subsample(), "subsample"

    def _scan(
        self,
        space: dict[str, Any],
        posterior: "_LatentPosterior",
        excluded: np.ndarray,
        keep: int,
    ) -> tuple[dict[str, np.ndarray], int, str]:
        """Return one candidate pool per acquisition kind, from a single pass.

        Each pool is that acquisition's global top-`keep` plus its best candidate per
        (variable, option). EI and UCB share the posterior mean and variance, so
        scoring both costs one scan.
        """
        blocks = [block.to(self._score_dtype) for block in space["blocks"]]
        kinds = self._score_kinds
        top_scores = {kind: np.empty(0, dtype=np.float64) for kind in kinds}
        top_idx = {kind: np.empty(0, dtype=np.int64) for kind in kinds}
        option_best = {kind: [np.full(radix, -np.inf) for radix in space["radices"]] for kind in kinds}
        option_idx = {
            kind: [np.full(radix, -1, dtype=np.int64) for radix in space["radices"]] for kind in kinds
        }
        chunks, mode = self._index_chunks(space, excluded)
        scanned = 0
        for indices in chunks:
            if indices.size == 0:
                continue
            digits = self._decode(indices, space)
            with torch.no_grad():
                mean, var = posterior.mean_var(self._features(digits, blocks))
            scanned += indices.size
            for kind in kinds:
                with torch.no_grad():
                    scores = self._acquisition(mean, var, posterior.best_f_score, kind).double().numpy()
                top_scores[kind], top_idx[kind] = _merge_top_k(
                    top_scores[kind],
                    top_idx[kind],
                    scores,
                    indices,
                    keep=keep,
                    dedupe=mode == "subsample",  # the same index can be drawn in several chunks
                )

                # Best candidate per (variable, option), so regions away from the global
                # optimum stay available to the diversity-aware batch selection.
                order = np.argsort(-scores, kind="stable")
                for v in range(len(space["names"])):
                    opts, first = np.unique(digits[order, v], return_index=True)
                    positions = order[first]
                    better = scores[positions] > option_best[kind][v][opts]
                    option_best[kind][v][opts[better]] = scores[positions][better]
                    option_idx[kind][v][opts[better]] = indices[positions][better]

        pools = {
            kind: np.unique(
                np.concatenate([top_idx[kind], *[idx[idx >= 0] for idx in option_idx[kind]]])
            )
            for kind in kinds
        }
        return pools, scanned, mode

    def _select_batch(
        self,
        space: dict[str, Any],
        posterior: "_LatentPosterior",
        pool: np.ndarray,
        size: int,
        post_filters: list[Callable[[Any], bool]],
        kind: str,
    ) -> tuple[list[int], int, bool]:
        if pool.size == 0:
            return [], 0, False
        digits = self._decode(pool, space)
        with torch.no_grad():
            mean, cov = posterior.mean_cov(self._features(digits, space["blocks"]))
        best = posterior.best_f.clone()
        blocked = torch.zeros(pool.size, dtype=torch.bool)
        chosen: list[int] = []
        post_filtered = 0
        relaxed = False
        while len(chosen) < size:
            scores = self._acquisition(mean, torch.diagonal(cov), best, kind)
            scores[blocked] = -torch.inf
            if chosen and self._min_changed_variables > 1:
                changed = (digits[:, None, :] != digits[chosen][None, :, :]).sum(axis=-1).min(axis=1)
                far = torch.from_numpy(changed >= self._min_changed_variables)
                if bool((far & ~blocked).any()):
                    scores = torch.where(far, scores, torch.full_like(scores, -torch.inf))
                else:
                    relaxed = True
            j = int(torch.argmax(scores))
            if not torch.isfinite(scores[j]):
                break
            blocked[j] = True
            if post_filters and not self._passes(post_filters, int(pool[j]), space):
                post_filtered += 1
                continue
            chosen.append(j)
            # Kriging believer: fantasize y_j = mean_j (mean unchanged) and apply the
            # rank-1 posterior covariance update, which shrinks variance near pick j.
            column = cov[:, j].clone()
            cov -= torch.outer(column, column) / (column[j] + posterior.noise)
            best = torch.maximum(best, mean[j])
        return [int(pool[j]) for j in chosen], post_filtered, relaxed

    def _merge_rank_by_mean(
        self,
        space: dict[str, Any],
        posterior: "_LatentPosterior",
        batches: dict[str, list[int]],
        size: int,
    ) -> tuple[list[int], list[str]]:
        """Merge per-acquisition batches, dedupe, rank by predicted mean, keep the top `size`.

        Returns the chosen indices and, for each, the acquisition(s) that proposed it
        (e.g. "ei", "ucb" or "ei+ucb"). Ties keep EI picks first.
        """
        proposers: dict[int, list[str]] = {}
        for kind, picks in batches.items():
            for index in picks:
                proposers.setdefault(index, []).append(kind)
        if not proposers:
            return [], []
        merged = np.asarray(list(proposers), dtype=np.int64)
        with torch.no_grad():
            mean, _ = posterior.mean_cov(self._features(self._decode(merged, space), space["blocks"]))
        order = sorted(range(merged.size), key=lambda i: -float(mean[i]))
        chosen = [int(merged[i]) for i in order[:size]]
        return chosen, ["+".join(proposers[index]) for index in chosen]

    def _to_parameter_vector(self, index: int, space: dict[str, Any], subspace):  # noqa: ANN001, ANN202
        return ParameterVector().from_dict(self._candidate(index, space), param_space=subspace)


class _LatentPosterior:
    """Exact GP posterior of the latent function in standardized outcome units.

    Computed directly from the fitted kernel and a Cholesky factor of the training
    covariance, so large chunks can be scored in `score_dtype` without gpytorch's
    approximate fast-predictive-variance path. EI/UCB rankings are invariant to the
    positive affine map back to raw outcome units.
    """

    def __init__(self, model: SingleTaskGP, score_dtype: torch.dtype) -> None:
        with torch.no_grad():
            train_x = model.train_inputs[0]
            n = train_x.shape[0]
            self.noise = float(model.likelihood.noise.detach().reshape(-1)[0])
            train_cov = model.covar_module(train_x).to_dense() + self.noise * torch.eye(
                n, dtype=train_x.dtype
            )
            chol = torch.linalg.cholesky(train_cov)
            prior_mean = model.mean_module(train_x[:1]).reshape(-1)[0]
            resid = (model.train_targets - prior_mean).unsqueeze(-1)
            alpha = torch.cholesky_solve(resid, chol).squeeze(-1)
            self.best_f = model.train_targets.max()
        self._kernel = model.covar_module
        self._train_x, self._chol, self._alpha, self._prior_mean = train_x, chol, alpha, prior_mean
        self._score_kernel = copy.deepcopy(model.covar_module).to(score_dtype)
        self._score_train_x = train_x.to(score_dtype)
        self._score_chol = chol.to(score_dtype)
        self._score_alpha = alpha.to(score_dtype)
        self._score_prior_mean = prior_mean.to(score_dtype)
        self.best_f_score = self.best_f.to(score_dtype)

    def mean_var(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Marginal posterior mean and variance for each row of `x` (score dtype)."""
        cross = self._score_kernel(x, self._score_train_x).to_dense()
        mean = self._score_prior_mean + cross @ self._score_alpha
        solved = torch.linalg.solve_triangular(self._score_chol, cross.T, upper=False)
        var = self._score_kernel(x, diag=True) - (solved * solved).sum(dim=0)
        return mean, var

    def mean_cov(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint posterior mean and covariance for the rows of `x` (float64)."""
        cross = self._kernel(x, self._train_x).to_dense()
        mean = self._prior_mean + cross @ self._alpha
        solved = torch.linalg.solve_triangular(self._chol, cross.T, upper=False)
        cov = self._kernel(x).to_dense() - solved.T @ solved
        return mean, cov


def _merge_top_k(
    top_scores: np.ndarray,
    top_idx: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
    *,
    keep: int,
    dedupe: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge one chunk into the running top-`keep` (scores, indices), unordered."""
    merged_scores = np.concatenate([top_scores, scores])
    merged_idx = np.concatenate([top_idx, indices])
    if dedupe:
        merged_idx, first = np.unique(merged_idx, return_index=True)
        merged_scores = merged_scores[first]
    if merged_scores.size > keep:
        best = np.argpartition(-merged_scores, keep - 1)[:keep]
        merged_scores, merged_idx = merged_scores[best], merged_idx[best]
    return merged_scores, merged_idx


def _is_member(sorted_values: np.ndarray, value: int) -> bool:
    position = int(np.searchsorted(sorted_values, value))
    return position < sorted_values.size and int(sorted_values[position]) == value
