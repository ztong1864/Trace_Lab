"""Finite-pool tabular dataset registry and helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from olympus.campaigns import ParameterSpace
from olympus.objects import ParameterContinuous, ParameterVector

try:
    from olympus.objects import ParameterCategorical
except ImportError:  # pragma: no cover
    from olympus.objects import Parameter as _OlympusParameter

    def ParameterCategorical(*, name: str, options: list[str]):  # type: ignore[misc]
        return _OlympusParameter(kind="categorical", name=name, options=options)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FinitePoolDatasetSpec:
    name: str
    csv_path: str
    feature_columns: tuple[str, ...]
    target_column: str
    goal: str = "maximize"
    fixed_filters: tuple[tuple[str, str], ...] = ()
    description: str = ""
    scaffold_dims: tuple[str, ...] = ()
    key_dimensions: tuple[str, ...] = ()
    reaction_type: str = "generic"


BUCHWALD_PRODUCT_SUBTASKS: tuple[tuple[str, str], ...] = (
    ("buchwald_task_1", "Cc1ccc(Nc2ccc(C(F)(F)F)cc2)cc1"),
    ("buchwald_task_2", "Cc1ccc(Nc2ccccn2)cc1"),
    ("buchwald_task_3", "Cc1ccc(Nc2cccnc2)cc1"),
    ("buchwald_task_4", "CCc1ccc(Nc2ccc(C)cc2)cc1"),
    ("buchwald_task_5", "COc1ccc(Nc2ccc(C)cc2)cc1"),
)

BUCHWALD_REPRESENTATIVE_SUBTASKS: tuple[tuple[str, str], ...] = (
    ("buchwald_sub1", "Cc1ccc(Nc2ccccn2)cc1"),
    ("buchwald_sub2", "COc1ccc(Nc2ccc(C)cc2)cc1"),
)


def _build_specs() -> dict[str, FinitePoolDatasetSpec]:
    specs: dict[str, FinitePoolDatasetSpec] = {
        "suzuki_hte_full": FinitePoolDatasetSpec(
            name="suzuki_hte_full",
            csv_path="dataset/suzuki/experiment_index.csv",
            feature_columns=(
                "electrophile",
                "nucleophile",
                "catalyst",
                "ligand",
                "base",
                "solvent",
            ),
            target_column="yield",
            goal="maximize",
            description="Suzuki HTE finite-pool benchmark (full CSV).",
            scaffold_dims=("electrophile", "nucleophile"),
            key_dimensions=("electrophile", "nucleophile"),
            reaction_type="suzuki_miyaura",
        ),
        "suzuki_chembomas_handled": FinitePoolDatasetSpec(
            name="suzuki_chembomas_handled",
            csv_path="dataset/suzuki/chembomas_handled.csv",
            feature_columns=(
                "electrophile",
                "nucleophile",
                "catalyst",
                "ligand",
                "base",
                "solvent",
            ),
            target_column="yield",
            goal="maximize",
            description=(
                "Suzuki ChemBOMAS handled subset for paper-protocol comparisons."
            ),
            scaffold_dims=("electrophile", "nucleophile"),
            key_dimensions=("electrophile", "nucleophile"),
            reaction_type="suzuki_miyaura",
        ),
        "buchwald_full": FinitePoolDatasetSpec(
            name="buchwald_full",
            csv_path="dataset/buchwald/processed.csv",
            feature_columns=("Reactant2", "Ligand", "Base", "Additive"),
            target_column="Yield",
            goal="maximize",
            description="Buchwald finite-pool benchmark (full processed.csv).",
            scaffold_dims=("Reactant2", "Ligand"),
            key_dimensions=("Reactant2", "Ligand"),
            reaction_type="buchwald_hartwig_amination",
        ),
        "arylation": FinitePoolDatasetSpec(
            name="arylation",
            csv_path="dataset/arylation/processed.csv",
            feature_columns=(
                "Aryl_halide_SMILES",
                "Additive_SMILES",
                "Base_SMILES",
                "Ligand_SMILES",
            ),
            target_column="yield",
            goal="maximize",
            description="C-H arylation finite-pool benchmark (full ChemBOMAS raw searchspace).",
            scaffold_dims=("Aryl_halide_SMILES", "Additive_SMILES"),
            key_dimensions=("Aryl_halide_SMILES", "Additive_SMILES"),
            reaction_type="c_h_arylation",
        ),
        "arylation_chembomas_handled": FinitePoolDatasetSpec(
            name="arylation_chembomas_handled",
            csv_path="dataset/arylation/chembomas_handled.csv",
            feature_columns=(
                "Aryl_halide_SMILES",
                "Additive_SMILES",
                "Base_SMILES",
                "Ligand_SMILES",
            ),
            target_column="yield",
            goal="maximize",
            description=(
                "C-H arylation ChemBOMAS handled subset for paper-protocol comparisons."
            ),
            scaffold_dims=("Aryl_halide_SMILES", "Additive_SMILES"),
            key_dimensions=("Aryl_halide_SMILES", "Additive_SMILES"),
            reaction_type="c_h_arylation",
        ),
        "buchwald_sub1_chembomas_handled": FinitePoolDatasetSpec(
            name="buchwald_sub1_chembomas_handled",
            csv_path="dataset/buchwald/sub1_chembomas_handled.csv",
            feature_columns=("Reactant2", "Ligand", "Base", "Additive"),
            target_column="Yield",
            goal="maximize",
            description=(
                "Buchwald ChemBOMAS handled product subset 1 "
                "(Cc1ccc(Nc2ccccn2)cc1) for paper-protocol comparisons."
            ),
            scaffold_dims=("Reactant2", "Ligand"),
            key_dimensions=("Reactant2", "Ligand"),
            reaction_type="buchwald_hartwig_amination",
        ),
        "buchwald_sub2_chembomas_handled": FinitePoolDatasetSpec(
            name="buchwald_sub2_chembomas_handled",
            csv_path="dataset/buchwald/sub2_chembomas_handled.csv",
            feature_columns=("Reactant2", "Ligand", "Base", "Additive"),
            target_column="Yield",
            goal="maximize",
            description=(
                "Buchwald ChemBOMAS handled product subset 2 "
                "(COc1ccc(Nc2ccc(C)cc2)cc1) for paper-protocol comparisons."
            ),
            scaffold_dims=("Reactant2", "Ligand"),
            key_dimensions=("Reactant2", "Ligand"),
            reaction_type="buchwald_hartwig_amination",
        ),
    }

    for task_name, product_smiles in BUCHWALD_PRODUCT_SUBTASKS:
        specs[task_name] = FinitePoolDatasetSpec(
            name=task_name,
            csv_path="dataset/buchwald/processed.csv",
            feature_columns=("Reactant2", "Ligand", "Base", "Additive"),
            target_column="Yield",
            goal="maximize",
            fixed_filters=(("Product", product_smiles),),
            description=f"Buchwald product-specific finite-pool benchmark ({product_smiles}).",
            scaffold_dims=("Reactant2", "Ligand"),
            key_dimensions=("Reactant2", "Ligand"),
            reaction_type="buchwald_hartwig_amination",
        )
    for task_name, product_smiles in BUCHWALD_REPRESENTATIVE_SUBTASKS:
        specs[task_name] = FinitePoolDatasetSpec(
            name=task_name,
            csv_path="dataset/buchwald/processed.csv",
            feature_columns=("Reactant2", "Ligand", "Base", "Additive"),
            target_column="Yield",
            goal="maximize",
            fixed_filters=(("Product", product_smiles),),
            description=(
                "Buchwald representative full product-level subset "
                f"({product_smiles}); not ChemBOMAS handled/test split."
            ),
            scaffold_dims=("Reactant2", "Ligand"),
            key_dimensions=("Reactant2", "Ligand"),
            reaction_type="buchwald_hartwig_amination",
        )
    return specs


FINITE_POOL_DATASET_SPECS = _build_specs()


def get_finite_pool_dataset_names() -> tuple[str, ...]:
    return tuple(sorted(FINITE_POOL_DATASET_SPECS))


def is_finite_pool_dataset(dataset_name: str) -> bool:
    return dataset_name in FINITE_POOL_DATASET_SPECS


class FinitePoolTable:
    """Tabular finite-pool task with exact lookup semantics."""

    def __init__(self, spec: FinitePoolDatasetSpec) -> None:
        self.spec = spec
        self.dataset_name = spec.name
        self.goal = spec.goal
        self.source_path = str((PROJECT_ROOT / spec.csv_path).resolve())
        self._df = self._load_dataframe()
        self.feature_columns = list(spec.feature_columns)
        self.target_column = spec.target_column
        self._record_map = self._build_record_map()
        self._key_set = set(self._record_map)
        self._param_space = self._build_param_space()
        self._value_space = self._build_value_space()

    def _load_dataframe(self) -> pd.DataFrame:
        csv_path = PROJECT_ROOT / self.spec.csv_path
        if not csv_path.exists():
            raise FileNotFoundError(f"Finite-pool dataset CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        for col, val in self.spec.fixed_filters:
            if col not in df.columns:
                raise ValueError(f"Filter column `{col}` missing in {csv_path}")
            df = df.loc[df[col].astype(str) == str(val)]
        required = [*self.spec.feature_columns, self.spec.target_column]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in `{csv_path}`: {missing}")
        task_df = df.loc[:, required].copy()
        if task_df.empty:
            raise ValueError(f"Finite-pool dataset `{self.spec.name}` has no rows after filtering.")
        task_df = task_df.dropna(subset=required)
        task_df = task_df.drop_duplicates(subset=list(self.spec.feature_columns), keep="first")
        task_df = task_df.reset_index(drop=True)
        return task_df

    def _build_record_map(self) -> dict[tuple[str, ...], float]:
        records: dict[tuple[str, ...], float] = {}
        for _, row in self._df.iterrows():
            key = self._row_to_key(row)
            records[key] = float(row[self.target_column])
        return records

    def _build_param_space(self) -> ParameterSpace:
        space = ParameterSpace()
        for col in self.feature_columns:
            options = list(dict.fromkeys(self._df[col].astype(str).tolist()))
            space.add(ParameterCategorical(name=col, options=options))
        return space

    def _build_value_space(self) -> ParameterSpace:
        target_values = self._df[self.target_column].astype(float)
        low = float(target_values.min())
        high = float(target_values.max())
        value_space = ParameterSpace()
        value_space.add(ParameterContinuous(name=self.target_column, low=low, high=high))
        return value_space

    def _row_to_key(self, row: pd.Series) -> tuple[str, ...]:
        return tuple(str(row[col]) for col in self.feature_columns)

    def _dict_to_key(self, sample: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(sample[col]) for col in self.feature_columns)

    def _vector_to_key(self, sample: ParameterVector) -> tuple[str, ...]:
        return tuple(str(getattr(sample, col)) for col in self.feature_columns)

    def sample_to_key(self, sample: Any) -> tuple[str, ...]:
        if isinstance(sample, dict):
            return self._dict_to_key(sample)
        if isinstance(sample, ParameterVector):
            return self._vector_to_key(sample)
        if isinstance(sample, (list, tuple)):
            if len(sample) != len(self.feature_columns):
                raise ValueError("Sample length does not match finite-pool feature dimension.")
            return tuple(str(v) for v in sample)
        raise TypeError(f"Unsupported finite-pool sample type: {type(sample)}")

    @property
    def param_space(self) -> ParameterSpace:
        return self._param_space

    @property
    def value_space(self) -> ParameterSpace:
        return self._value_space

    @property
    def candidate_count(self) -> int:
        return len(self._key_set)

    def record_keys(self) -> tuple[tuple[str, ...], ...]:
        return tuple(sorted(self._key_set))

    def key_to_candidate(self, key: tuple[str, ...]) -> dict[str, str]:
        if len(key) != len(self.feature_columns):
            raise ValueError("Finite-pool key length does not match feature dimension.")
        return {name: str(value) for name, value in zip(self.feature_columns, key, strict=False)}

    def candidate_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for idx, row in self._df.iterrows():
            key = self._row_to_key(row)
            records.append(
                {
                    "candidate_id": f"cand_{int(idx):06d}",
                    "key": key,
                    "candidate": self.key_to_candidate(key),
                    "value": float(row[self.target_column]),
                }
            )
        return records

    def is_valid_candidate(self, sample: Any) -> bool:
        try:
            key = self.sample_to_key(sample)
        except Exception:  # noqa: BLE001
            return False
        return key in self._key_set

    def evaluate(self, sample: Any) -> float:
        key = self.sample_to_key(sample)
        if key not in self._record_map:
            raise ValueError(f"Finite-pool candidate not found in dataset `{self.dataset_name}`: {key}")
        return float(self._record_map[key])

    def membership_constraint(
        self,
        allowed_keys: set[tuple[str, ...]] | None = None,
    ) -> Callable[[Any], bool]:
        legal_keys = allowed_keys if allowed_keys is not None else self._key_set

        def _constraint(values: Any) -> bool:
            if isinstance(values, ParameterVector):
                key = self._vector_to_key(values)
            elif isinstance(values, dict):
                key = self._dict_to_key(values)
            elif hasattr(values, "tolist"):
                arr = values.tolist()
                if isinstance(arr, list):
                    key = tuple(str(v) for v in arr)
                else:
                    key = (str(arr),)
            elif isinstance(values, (list, tuple)):
                key = tuple(str(v) for v in values)
            else:
                return False
            return key in legal_keys

        return _constraint

    def filter_keys_by_focus(
        self,
        *,
        focus_variables: list[str] | None,
        best_observation: dict[str, Any] | None,
        filter_mode: str = "exact_match",
    ) -> tuple[set[tuple[str, ...]], bool]:
        """Return legal key set and whether we fell back from an empty sub-pool to the full pool.

        ``subpool_empty_fallback`` is True only when focus variables and best observation
        define a strict sub-filter but no rows match; we then use the full candidate set.
        """
        if not focus_variables or not best_observation:
            return set(self._key_set), False
        if str(filter_mode) != "exact_match":
            return set(self._key_set), False
        valid_focus = [name for name in focus_variables if name in self.feature_columns]
        if not valid_focus:
            return set(self._key_set), False
        filtered: set[tuple[str, ...]] = set()
        idx_map = {name: i for i, name in enumerate(self.feature_columns)}
        for key in self._key_set:
            keep = True
            for name in valid_focus:
                target_val = best_observation.get(name)
                if target_val is None:
                    keep = False
                    break
                if key[idx_map[name]] != str(target_val):
                    keep = False
                    break
            if keep:
                filtered.add(key)
        if not filtered:
            return set(self._key_set), True
        return filtered, False

    def filter_keys_by_llm_constraint(
        self,
        include_map: dict[str, list[str]] | None = None,
        exclude_map: dict[str, list[str]] | None = None,
        min_pool_fraction: float = 0.05,
    ) -> tuple[set[tuple[str, ...]], bool]:
        """Filter candidate keys by LLM-generated include/exclude constraints.

        Returns (filtered_keys, fallback_triggered).
        fallback_triggered=True when the filtered pool falls below min_pool_fraction
        of the full candidate count, in which case the full candidate set is returned.
        """
        if not include_map and not exclude_map:
            return set(self._key_set), False

        idx_map = {name: i for i, name in enumerate(self.feature_columns)}
        filtered: set[tuple[str, ...]] = set()
        for key in self._key_set:
            keep = True
            if include_map:
                for col, allowed in include_map.items():
                    idx = idx_map.get(col)
                    if idx is None:
                        continue
                    if key[idx] not in allowed:
                        keep = False
                        break
            if keep and exclude_map:
                for col, blocked in exclude_map.items():
                    idx = idx_map.get(col)
                    if idx is None:
                        continue
                    if key[idx] in blocked:
                        keep = False
                        break
            if keep:
                filtered.add(key)

        min_size = max(1, int(len(self._key_set) * min_pool_fraction))
        if len(filtered) < min_size:
            return set(self._key_set), True
        return filtered, False

    @staticmethod
    def allowed_keys_signature(allowed_keys: set[tuple[str, ...]]) -> str:
        digest = hashlib.sha1()
        for key in sorted(allowed_keys):
            digest.update("\x1f".join(key).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def dataset_meta(self) -> dict[str, Any]:
        return {
            "backend": "finite_pool",
            "dataset_name": self.dataset_name,
            "source_path": self.source_path,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "goal": self.goal,
            "candidate_count": self.candidate_count,
            "fixed_filters": {k: v for k, v in self.spec.fixed_filters},
            "description": self.spec.description,
            "scaffold_dims": list(self.spec.scaffold_dims),
            "key_dimensions": list(self.spec.key_dimensions or self.spec.scaffold_dims),
            "reaction_type": self.spec.reaction_type,
        }


def load_finite_pool_table(dataset_name: str) -> FinitePoolTable:
    if dataset_name not in FINITE_POOL_DATASET_SPECS:
        known = ", ".join(sorted(FINITE_POOL_DATASET_SPECS))
        raise ValueError(f"Unknown finite-pool dataset `{dataset_name}`. Available: {known}")
    return FinitePoolTable(FINITE_POOL_DATASET_SPECS[dataset_name])
