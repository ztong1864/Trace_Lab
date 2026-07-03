"""LLM-based decision engine using LangChain create_agent."""
from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from chem_agent_bo.agent.llm_usage import (
    estimate_cost_usd,
    extract_usage_from_result,
    resolve_price,
)
from chem_agent_bo.agent.action_package_policy import (
    append_reasoning_note,
    build_v06_action_admissibility,
    controller_policy_signals,
    normalize_v06_action_package,
    realign_action_package_fields,
)
from chem_agent_bo.config.schema import PromptConfig
from chem_agent_bo.knowledge import (
    load_translation_cache,
    merge_translation_entry,
    render_annotation_context,
    save_translation_cache,
)
from chem_agent_bo.procedural_skills import ProceduralSkillRegistry
from chem_agent_bo.prompts.templates import (
    SYSTEM_PROMPT,
    build_controller_plan_prompt,
    build_completion_action_prompt,
    build_decision_action_prompt,
    build_coverage_insight_prompt,
    build_feasibility_action_prompt,
    build_hypothesis_action_prompt,
    build_init_design_prompt,
    build_intervention_plan_prompt,
    build_lab_batch_composition_prompt,
    build_candidate_verification_prompt,
    build_reflection_action_prompt,
    build_search_constraint_prompt,
    build_shortlist_rerank_prompt,
    build_semantic_assessment_prompt,
    build_stagnation_diagnosis_prompt,
)

LOGGER = logging.getLogger(__name__)


class DecisionAction(BaseModel):
    stage: str = Field(default="exploration")
    active_variables: list[str] = Field(default_factory=list)
    fixed_variables_strategy: Literal[
        "llm_proposed",
        "best_observed",
        "fallback_default",
        "mixed",
    ] = (
        "llm_proposed"
    )
    fixed_variables: dict[str, float | str] = Field(default_factory=dict)
    decision_mode: Literal["explore", "exploit", "mixed"] = "mixed"
    reasoning: str = Field(default="Fallback decision action.")


class CompletionAction(BaseModel):
    full_candidate: dict[str, float | str] = Field(default_factory=dict)
    reasoning: str = Field(default="Fallback completion action.")


class FeasibilityAction(BaseModel):
    action: Literal["accept", "revise", "reject"] = "accept"
    reasoning: str = Field(default="Fallback feasibility action.")
    revised_candidate: dict[str, float | str] | None = None


class ReflectionAction(BaseModel):
    insight: str = Field(default="Fallback reflection memo.")
    confidence: Literal["low", "medium", "high"] = "low"
    next_step_hypothesis: str = Field(default="Continue with broader exploration.")
    suggested_focus: list[str] = Field(default_factory=list)
    avoid_pattern: list[str] = Field(default_factory=list)


class HypothesisAction(BaseModel):
    hypotheses: list[str] = Field(default_factory=list)
    suggested_focus_variables: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="No specific hypothesis.")


class StagnationDiagnosis(BaseModel):
    is_stagnating: bool = False
    stagnation_type: str = Field(default="none")
    suspected_causes: list[str] = Field(default_factory=list)
    recommended_intervention: str = Field(default="keep_full_space")
    reasoning: str = Field(default="No stagnation detected.")


class SemanticAssessment(BaseModel):
    risk_level: Literal["low", "medium", "high"] = "low"
    plausibility_score: float = 0.7
    novelty_score: float = 0.5
    soft_comment: str = Field(default="Candidate is acceptable for execution.")
    suggested_bias: list[str] = Field(default_factory=list)


class VerificationPass(BaseModel):
    status: Literal["pass", "caution", "fail_soft"] = "pass"
    confidence: Literal["low", "medium", "high"] = "medium"
    risk_flags: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    suggested_action: Literal["keep", "add_caution"] = "keep"
    reasoning: str = Field(default="No additional verification concerns.")


class CoverageInsight(BaseModel):
    coverage_status: str = Field(default="unknown")
    underexplored_dimensions: list[str] = Field(default_factory=list)
    repetition_risk: str = Field(default="low")
    suggested_probe_regions: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="Coverage signal unavailable.")


class InterventionPlan(BaseModel):
    intervention_type: Literal[
        "none",
        "soft_guidance",
        "subspace_window",
        "bo_direct",
        "bo_rerank_topk",
        "bo_focus_then_rerank",
    ] = "bo_direct"
    action_schema_version: Literal["compat_v1", "v2", "v0.6"] = "compat_v1"
    requested_execution_action: Literal[
        "direct_bo_pick",
        "shape_only_bo_pick",
        "shape_then_probe_topk",
        "shortlist_alt_pick",
        "focused_shortlist_alt_pick",
        "finite_pool_candidate_probe",
        "mask_scaffold_corridor_resuggest",
        "mask_dominant_resuggest",
        "mask_low_repeat_resuggest",
    ] = "direct_bo_pick"
    intent: Literal["exploit", "probe", "balance"] = "balance"
    shortlist_policy: Literal[
        "plain",
        "diversity_shape",
        "coverage_shape",
        "contrast_shape",
    ] = "plain"
    repeat_policy: Literal[
        "allow",
        "avoid_near_duplicate",
        "avoid_anchor_repeat",
    ] = "allow"
    selection_policy: Literal[
        "bo_top1",
        "bo_top1_from_shaped_shortlist",
        "select_from_shaped_shortlist",
    ] = "bo_top1"
    verification_policy: Literal["normal", "strict"] = "normal"
    focus_policy: Literal["full_space", "temporary_focus"] = "full_space"
    use_subspace: bool = False
    focus_variables: list[str] = Field(default_factory=list)
    window_rounds: int = 0
    candidate_probe_include: list[str] = Field(
        default_factory=list,
        description=(
            "Optional finite-pool probe direction as 'column=value' entries. "
            "Use only for finite_pool_candidate_probe; do not name a final candidate."
        ),
    )
    candidate_probe_reasoning: str = Field(
        default="",
        description="Brief evidence for the candidate_probe_include direction.",
    )
    reasoning: str = Field(default="No intervention needed.")


class ShortlistCandidateScore(BaseModel):
    candidate_index: int = 0
    overall_score: float = 0.0
    plausibility_score: float = 0.5
    novelty_score: float = 0.5
    transfer_value_score: float = 0.5
    hypothesis_value_score: float = 0.5
    structural_shift_type: Literal[
        "none",
        "local_refinement",
        "cross_scaffold_transfer",
        "mechanistic_contrast",
    ] = "none"
    hypothesis_summary: str = Field(
        default="No explicit hypothesis beyond the BO shortlist ranking."
    )
    local_overfit_risk: Literal["low", "medium", "high"] = "low"
    reasoning: str = Field(default="Fallback shortlist score.")


class ShortlistRerankAction(BaseModel):
    selected_index: int = 0
    candidate_scores: list[ShortlistCandidateScore] = Field(default_factory=list)
    reasoning: str = Field(default="Fallback to BO top-1.")


class LabBatchSlotAction(BaseModel):
    slot_id: int = 0
    role: str = Field(default="batch_member")
    candidate_index: int = 0
    purpose: str = Field(default="Use one legal planner candidate in the lab batch.")
    varied_variables: list[str] = Field(default_factory=list)
    controlled_variables: list[str] = Field(default_factory=list)
    rationale: str = Field(default="Fallback slot rationale.")
    risk_note: str = Field(default="")
    evidence_refs: list[str] = Field(default_factory=list)


class LabBatchCompositionAction(BaseModel):
    batch_strategy: str = Field(default="planner_anchor_diversity")
    batch_rationale: str = Field(
        default="Fallback: keep the planner anchor and add diverse legal candidates."
    )
    global_constraints: list[str] = Field(default_factory=list)
    slots: list[LabBatchSlotAction] = Field(default_factory=list)


class InitDesignCandidate(BaseModel):
    # Use list[str] with "col=value" format to avoid OpenAI strict-mode additionalProperties issues.
    # Example: ["electrophile=1d, 6-I-Q", "nucleophile=2a, Boronic Acid", "catalyst=Pd(OAc)2", ...]
    assignments: list[str] = Field(
        default_factory=list,
        description="One entry per feature column, each formatted as 'column_name=value'",
    )
    rationale: str = Field(default="No rationale provided.")


class InitDesignAction(BaseModel):
    designed_experiments: list[InitDesignCandidate] = Field(default_factory=list)
    design_strategy: str = Field(default="scaffold_diversity_first")
    reasoning: str = Field(default="Fallback: random init.")


class ValueTranslationEntry(BaseModel):
    original_value: str = Field(default="")
    translated_description: str = Field(default="")
    brief_properties: list[str] = Field(default_factory=list)
    likely_role: str = Field(default="")
    confidence: Literal["low", "medium", "high"] = "medium"


class LLMConstraintSpec(BaseModel):
    variable: str = Field(default="")
    constraint_type: Literal["include_values", "exclude_values"] = "include_values"
    values: list[str] = Field(default_factory=list)
    rationale: str = Field(default="No rationale.")
    confidence: Literal["low", "medium", "high"] = "medium"
    duration_rounds: int = 5


class LLMSearchConstraintAction(BaseModel):
    constraints: list[LLMConstraintSpec] = Field(default_factory=list)
    constraint_summary: str = Field(default="No constraints generated.")
    reasoning: str = Field(default="Fallback: full space search.")
    retain_full_space_fallback: bool = True


class DecisionEngine:
    """Decision policy powered by LangChain 1.x create_agent."""

    def __init__(
        self,
        model_name: str = "gpt-5.4-mini",
        temperature: float = 0.0,
        api_base: str | None = None,
        timeout_sec: float = 45.0,
        request_max_retries: int = 2,
        structured_retry_attempts: int = 3,
        retry_backoff_sec: float = 1.5,
        retry_max_backoff_sec: float = 12.0,
        retry_jitter_sec: float = 0.5,
        fallback_model_name: str | None = None,
        fallback_attempts: int = 2,
        fail_on_nonretryable_error: bool = True,
        pricing_profile: str = "auto",
        input_cost_per_1m: float | None = None,
        output_cost_per_1m: float | None = None,
        cached_input_cost_per_1m: float | None = None,
        prompt_config: PromptConfig | None = None,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        resolved_base = (
            api_base
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        if resolved_base is not None:
            resolved_base = str(resolved_base).strip() or None

        model_kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "timeout": timeout_sec,
            "max_retries": request_max_retries,
        }
        self._structured_retry_attempts = max(1, int(structured_retry_attempts))
        self._retry_backoff_sec = max(0.0, float(retry_backoff_sec))
        self._retry_max_backoff_sec = max(self._retry_backoff_sec, float(retry_max_backoff_sec))
        self._retry_jitter_sec = max(0.0, float(retry_jitter_sec))
        self._fallback_attempts = max(0, int(fallback_attempts))
        self._fail_on_nonretryable_error = bool(fail_on_nonretryable_error)
        self.model_name = str(model_name)
        self.pricing_profile, self._llm_price = resolve_price(
            model_name=self.model_name,
            pricing_profile=pricing_profile,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
            cached_input_cost_per_1m=cached_input_cost_per_1m,
        )
        self._fallback_model_name = str(fallback_model_name or "").strip() or None
        self._fallback_pricing_profile: str | None = None
        self._fallback_llm_price = None
        self._usage_records: list[dict[str, Any]] = []
        self._skill_trace_by_call: dict[str, dict[str, Any]] = {}
        self.prompt_config = prompt_config or PromptConfig()
        self.procedural_skill_registry = ProceduralSkillRegistry(
            enabled=bool(getattr(self.prompt_config.skills, "enabled", True)),
            cards_dir=getattr(self.prompt_config.skills, "cards_dir", None),
        )
        self._translation_cache_path = Path(
            os.getenv(
                "CHEM_AGENT_VALUE_TRANSLATION_CACHE",
                str(Path(__file__).resolve().parents[1] / "knowledge" / "value_translation_cache.yaml"),
            )
        )
        self._translation_cache = load_translation_cache(self._translation_cache_path)
        if api_key:
            model_kwargs["api_key"] = api_key
        if resolved_base:
            model_kwargs["base_url"] = resolved_base

        if api_key:
            primary_agents = self._build_agent_bundle(model_kwargs)
            self.decision_agent = primary_agents["decision_action"]
            self.completion_agent = primary_agents["completion_action"]
            self.feasibility_agent = primary_agents["feasibility_action"]
            self.reflection_agent = primary_agents["reflection_action"]
            self.hypothesis_agent = primary_agents["hypothesis_action"]
            self.diagnosis_agent = primary_agents["stagnation_diagnosis"]
            self.semantic_assessment_agent = primary_agents["semantic_assessment"]
            self.verification_agent = primary_agents["candidate_verification"]
            self.coverage_agent = primary_agents["coverage_insight"]
            self.intervention_agent = primary_agents["intervention_plan"]
            self.shortlist_rerank_agent = primary_agents["shortlist_rerank"]
            self.lab_batch_composition_agent = primary_agents["lab_batch_composition"]
            self.init_design_agent = primary_agents["design_init_experiments"]
            self.search_constraint_agent = primary_agents["generate_search_constraints"]
            self.value_translation_agent = primary_agents["value_translation"]

            self._fallback_agents: dict[str, Any] = {}
            if self._fallback_model_name and self._fallback_model_name != self.model_name:
                fallback_kwargs = dict(model_kwargs)
                fallback_kwargs["model"] = self._fallback_model_name
                self._fallback_agents = self._build_agent_bundle(fallback_kwargs)
                self._fallback_pricing_profile, self._fallback_llm_price = resolve_price(
                    model_name=self._fallback_model_name,
                    pricing_profile=pricing_profile,
                    input_cost_per_1m=input_cost_per_1m,
                    output_cost_per_1m=output_cost_per_1m,
                    cached_input_cost_per_1m=cached_input_cost_per_1m,
                )
        else:
            self.decision_agent = None
            self.completion_agent = None
            self.feasibility_agent = None
            self.reflection_agent = None
            self.hypothesis_agent = None
            self.diagnosis_agent = None
            self.semantic_assessment_agent = None
            self.verification_agent = None
            self.coverage_agent = None
            self.intervention_agent = None
            self.shortlist_rerank_agent = None
            self.lab_batch_composition_agent = None
            self.init_design_agent = None
            self.search_constraint_agent = None
            self.value_translation_agent = None
            self._fallback_agents = {}
            LOGGER.warning(
                "No API key found in OPENAI_API_KEY. "
                "DecisionEngine will use fallback logic only."
            )

    @staticmethod
    def _build_agent_bundle(model_kwargs: dict[str, Any]) -> dict[str, Any]:
        model = ChatOpenAI(**model_kwargs)
        _rf = ToolStrategy
        decision_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(DecisionAction),
        )
        completion_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(CompletionAction),
        )
        feasibility_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(FeasibilityAction),
        )
        reflection_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(ReflectionAction),
        )
        hypothesis_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(HypothesisAction),
        )
        diagnosis_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(StagnationDiagnosis),
        )
        semantic_assessment_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(SemanticAssessment),
        )
        verification_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(VerificationPass),
        )
        coverage_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(CoverageInsight),
        )
        intervention_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(InterventionPlan),
        )
        shortlist_rerank_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(ShortlistRerankAction),
        )
        lab_batch_composition_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(LabBatchCompositionAction),
        )
        init_design_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(InitDesignAction),
        )
        search_constraint_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(LLMSearchConstraintAction),
        )
        value_translation_agent = create_agent(
            model=model,
            tools=[],
            system_prompt=SYSTEM_PROMPT,
            response_format=_rf(ValueTranslationEntry),
        )
        return {
            "decision_action": decision_agent,
            "completion_action": completion_agent,
            "feasibility_action": feasibility_agent,
            "reflection_action": reflection_agent,
            "hypothesis_action": hypothesis_agent,
            "stagnation_diagnosis": diagnosis_agent,
            "semantic_assessment": semantic_assessment_agent,
            "candidate_verification": verification_agent,
            "coverage_insight": coverage_agent,
            "intervention_plan": intervention_agent,
            "controller_plan": intervention_agent,
            "shortlist_rerank": shortlist_rerank_agent,
            "lab_batch_composition": lab_batch_composition_agent,
            "design_init_experiments": init_design_agent,
            "generate_search_constraints": search_constraint_agent,
            "value_translation": value_translation_agent,
        }

    @staticmethod
    def _is_retryable_connection_error(exc: Exception) -> bool:
        message = str(exc).lower()
        retryable_tokens = (
            "connection error",
            "connection reset",
            "connection aborted",
            "api connection",
            "service unavailable",
            "temporarily unavailable",
            "bad gateway",
            "gateway",
            "timeout",
            "timed out",
            "502",
            "503",
            "504",
            "rate limit",
            "overloaded",
            "server disconnected",
            "remoteprotocolerror",
        )
        return any(token in message for token in retryable_tokens)

    def _sleep_before_retry(self, attempt: int) -> None:
        if self._retry_backoff_sec <= 0:
            return
        delay = min(
            self._retry_backoff_sec * (2 ** max(0, attempt - 1)),
            self._retry_max_backoff_sec,
        )
        if self._retry_jitter_sec > 0:
            delay += random.uniform(0.0, self._retry_jitter_sec)
        time.sleep(delay)

    def _invoke_with_retries(
        self,
        agent,  # noqa: ANN001
        message_text: str,
        *,
        call_name: str,
        model_name: str,
        pricing_profile: str,
        price,
        max_attempts: int,
    ) -> tuple[BaseModel | None, Exception | None]:
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            started = time.time()
            try:
                result = agent.invoke({"messages": [{"role": "user", "content": message_text}]})
                usage, usage_source = extract_usage_from_result(result)
                latency_sec = time.time() - started
                structured = result.get("structured_response")
                if structured is not None:
                    self._record_usage(
                        call_name=call_name,
                        attempt=attempt,
                        success=True,
                        latency_sec=latency_sec,
                        usage=usage,
                        usage_source=usage_source,
                        model_name=model_name,
                        pricing_profile=pricing_profile,
                        price=price,
                    )
                    return structured, None
                self._record_usage(
                    call_name=call_name,
                    attempt=attempt,
                    success=False,
                    latency_sec=latency_sec,
                    usage=usage,
                    usage_source=usage_source,
                    error="missing_structured_response",
                    model_name=model_name,
                    pricing_profile=pricing_profile,
                    price=price,
                )
                LOGGER.warning(
                    "Agent response missing structured payload for %s on %s (attempt %s/%s).",
                    call_name,
                    model_name,
                    attempt,
                    max_attempts,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._record_usage(
                    call_name=call_name,
                    attempt=attempt,
                    success=False,
                    latency_sec=time.time() - started,
                    usage={
                        "input_tokens": None,
                        "output_tokens": None,
                        "cached_input_tokens": None,
                        "total_tokens": None,
                    },
                    usage_source="unavailable",
                    error=str(exc),
                    model_name=model_name,
                    pricing_profile=pricing_profile,
                    price=price,
                )
                LOGGER.warning(
                    "Agent structured output failed for %s on %s (attempt %s/%s): %s",
                    call_name,
                    model_name,
                    attempt,
                    max_attempts,
                    exc,
                )
            if attempt < max_attempts:
                self._sleep_before_retry(attempt)
        return None, last_error

    def _invoke_agent_structured(
        self,
        agent,  # noqa: ANN001
        message_text: str,
        fallback: BaseModel,
        *,
        call_name: str = "unknown",
    ) -> BaseModel:
        if agent is None:
            if self._fail_on_nonretryable_error:
                raise RuntimeError(
                    f"LLM agent unavailable for {call_name}; abort instead of using fallback."
                )
            self._record_usage(
                call_name=call_name,
                attempt=0,
                success=False,
                latency_sec=0.0,
                usage={
                    "input_tokens": None,
                    "output_tokens": None,
                    "cached_input_tokens": None,
                    "total_tokens": None,
                },
                usage_source="fallback_no_agent",
                error="agent_unavailable",
            )
            return fallback
        structured, last_error = self._invoke_with_retries(
            agent,
            message_text,
            call_name=call_name,
            model_name=self.model_name,
            pricing_profile=self.pricing_profile,
            price=self._llm_price,
            max_attempts=self._structured_retry_attempts,
        )
        if structured is not None:
            return structured
        fallback_agent = self._fallback_agents.get(call_name)
        if (
            fallback_agent is not None
            and self._fallback_model_name
            and self._fallback_attempts > 0
            and last_error is not None
            and self._is_retryable_connection_error(last_error)
        ):
            LOGGER.warning(
                "Primary model %s failed for %s with retryable connection error; retrying on fallback model %s.",
                self.model_name,
                call_name,
                self._fallback_model_name,
            )
            structured, fallback_error = self._invoke_with_retries(
                fallback_agent,
                message_text,
                call_name=call_name,
                model_name=self._fallback_model_name,
                pricing_profile=self._fallback_pricing_profile or self.pricing_profile,
                price=self._fallback_llm_price or self._llm_price,
                max_attempts=self._fallback_attempts,
            )
            if structured is not None:
                return structured
            if fallback_error is not None:
                last_error = fallback_error
        if (
            last_error is not None
            and self._fail_on_nonretryable_error
            and not self._is_retryable_connection_error(last_error)
        ):
            raise RuntimeError(
                f"Non-retryable LLM failure for {call_name} on {self.model_name}: {last_error}"
            ) from last_error
        if last_error is not None:
            LOGGER.warning("Using fallback due to repeated structured-call failure for %s.", call_name)
        return fallback

    def _record_usage(
        self,
        *,
        call_name: str,
        attempt: int,
        success: bool,
        latency_sec: float,
        usage: dict[str, int | None],
        usage_source: str,
        error: str | None = None,
        model_name: str | None = None,
        pricing_profile: str | None = None,
        price = None,
    ) -> None:
        cost = estimate_cost_usd(
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            cached_input_tokens=usage.get("cached_input_tokens"),
            price=price or self._llm_price,
        )
        self._usage_records.append(
            {
                "call_name": call_name,
                "model_name": model_name or self.model_name,
                "pricing_profile": pricing_profile or self.pricing_profile,
                "attempt": int(attempt),
                "success": bool(success),
                "latency_sec": round(float(latency_sec), 6),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cached_input_tokens": usage.get("cached_input_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "usage_source": usage_source,
                "estimated_cost_usd": None if cost is None else round(float(cost), 10),
                "error": error,
            }
        )

    def usage_records(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._usage_records]

    def reset_skill_trace(self) -> None:
        self._skill_trace_by_call = {}

    def skill_trace_snapshot(self) -> dict[str, Any]:
        return {
            key: {
                sub_key: value
                for sub_key, value in item.items()
            }
            for key, item in self._skill_trace_by_call.items()
        }

    def _render_procedural_skill_block(self, node_name: str, *, call_name: str) -> str:
        block, meta = self.procedural_skill_registry.render_for_node(node_name)
        meta = {
            **meta,
            "call_name": call_name,
        }
        if bool(getattr(self.prompt_config.skills, "trace_loaded_skills", True)):
            self._skill_trace_by_call[call_name] = meta
        return block

    def usage_summary(self) -> dict[str, Any]:
        records = self.usage_records()

        def _sum_int(field: str) -> int | None:
            values = [item.get(field) for item in records if item.get(field) is not None]
            if not values:
                return None
            return int(sum(int(value) for value in values))

        costs = [item.get("estimated_cost_usd") for item in records if item.get("estimated_cost_usd") is not None]
        return {
            "llm_model_name": self.model_name,
            "llm_pricing_profile": self.pricing_profile,
            "llm_total_calls": len(records),
            "llm_successful_calls": sum(1 for item in records if item.get("success")),
            "llm_total_input_tokens": _sum_int("input_tokens"),
            "llm_total_output_tokens": _sum_int("output_tokens"),
            "llm_total_cached_input_tokens": _sum_int("cached_input_tokens"),
            "llm_total_tokens": _sum_int("total_tokens"),
            "llm_estimated_cost_usd": round(float(sum(costs)), 10) if costs else None,
            "llm_usage_available": any(item.get("total_tokens") is not None for item in records),
        }

    @staticmethod
    def _search_space_info(param_space) -> list[dict[str, Any]]:  # noqa: ANN001
        info: list[dict[str, Any]] = []
        for param in param_space:
            item: dict[str, Any] = {"name": param.name, "type": param.type}
            if hasattr(param, "options"):
                item["options"] = list(getattr(param, "options"))
            if hasattr(param, "low"):
                item["low"] = getattr(param, "low")
            if hasattr(param, "high"):
                item["high"] = getattr(param, "high")
            info.append(item)
        return info

    @staticmethod
    def _search_space_meta(decision_context: dict[str, Any]) -> dict[str, Any]:
        meta = decision_context.get("search_space_meta", {})
        if isinstance(meta, dict):
            return meta
        return {}

    @staticmethod
    def _node_state_view(
        decision_context: dict[str, Any],
        node_name: str,
    ) -> dict[str, Any]:
        views = decision_context.get("node_state_views", {})
        if isinstance(views, dict):
            node_view = views.get(node_name)
            if isinstance(node_view, dict):
                return node_view
        return decision_context

    def _knowledge_for_prompt(
        self,
        decision_context: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        snippets = decision_context.get("knowledge_units", [])
        if not isinstance(snippets, list):
            snippets = []
        meta = decision_context.get("knowledge_meta", {})
        if not isinstance(meta, dict):
            meta = {}
        compact: list[dict[str, Any]] = []
        max_items = self.prompt_config.decision_engine_knowledge_max_items
        max_chars = self.prompt_config.decision_engine_knowledge_max_chars
        for item in snippets[:max_items]:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content", ""))
            compact.append(
                {
                    "id": item.get("id"),
                    "content": (
                        content[:max_chars] + "..."
                        if len(content) > max_chars
                        else content
                    ),
                    "source_type": item.get("source_type", "unknown"),
                    "confidence": item.get("confidence", 0.0),
                    "score": item.get("score"),
                }
            )
        return compact, meta

    @staticmethod
    def _coerce_stage(
        output: DecisionAction,
        decision_context: dict[str, Any],
    ) -> None:
        # Lightweight anti-stagnation nudge for v0.2.1: keep LLM-driven flow,
        # but avoid being stuck in one stage with no progress.
        stage_streak = int(decision_context.get("current_stage_streak", 0) or 0)
        best_improvement_last_3 = decision_context.get("best_improvement_last_3")
        if best_improvement_last_3 is None:
            return
        try:
            stalled = float(best_improvement_last_3) < 0.5
        except (TypeError, ValueError):
            stalled = False
        if stalled and stage_streak >= 4 and output.stage == "exploration":
            output.stage = "exploitation"
            output.decision_mode = "exploit"
            output.reasoning = (
                f"{output.reasoning} [v0.2.1 auto-nudge: stagnation detected, "
                "switch to exploitation.]"
            )


    @staticmethod
    def _collect_requested_values(payloads: list[dict[str, Any]] | None) -> dict[str, list[str]]:
        ignored_keys = {
            "backend",
            "candidate_count",
            "dataset_name",
            "description",
            "feature_columns",
            "goal",
            "reaction_type",
            "scaffold_dims",
            "source_path",
            "target_column",
            "constraint_signature",
            "constraint_summary",
            "reasoning",
            "valid_values_per_col",
        }
        collected: dict[str, list[str]] = {}
        seen: dict[str, set[str]] = {}
        allowed_columns: set[str] = set()

        for payload in payloads or []:
            if not isinstance(payload, dict):
                continue
            feature_columns = payload.get("feature_columns")
            if isinstance(feature_columns, list):
                allowed_columns.update(str(item) for item in feature_columns if str(item).strip())
            valid_values = payload.get("valid_values_per_col")
            if isinstance(valid_values, dict):
                allowed_columns.update(str(key) for key in valid_values if str(key).strip())

        def _push(column: str, raw_value: Any) -> None:
            if column in ignored_keys or column not in allowed_columns:
                return
            if not isinstance(raw_value, (str, int, float)):
                return
            text = str(raw_value).strip()
            if not text or text.isdigit():
                return
            collected.setdefault(column, [])
            seen.setdefault(column, set())
            if text not in seen[column]:
                collected[column].append(text)
                seen[column].add(text)

        def _walk(value: Any) -> None:
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    child_column = str(child_key)
                    if child_column in allowed_columns:
                        if isinstance(child_value, list):
                            for item in child_value:
                                _push(child_column, item)
                        else:
                            _push(child_column, child_value)
                    else:
                        _walk(child_value)
                return
            if isinstance(value, list):
                for item in value:
                    _walk(item)

        for payload in payloads or []:
            if isinstance(payload, dict):
                _walk(payload)
        return collected



    def _translate_value(self, value: str, column: str = "") -> dict[str, Any]:
        cached = self._translation_cache.get(str(value))
        if isinstance(cached, dict):
            return cached
        prompt_text = f"""
Task: translate one chemical value string into a short reusable annotation card.

Column:
{column or 'unknown'}

Original value:
{value}

Return a structured translation with:
- translated_description: short human-readable interpretation
- brief_properties: 1-4 short phrases about recognizable motifs / likely behavior / simple chemical cues
- likely_role: what role this value most likely plays in reaction design
- confidence: low / medium / high

Rules:
- Be conservative. If uncertain, describe only what is recognizable.
- Do not invent an exact identity unless strongly supported by the string.
- This translation will be reused as external prompt context, not executed directly.
""".strip()
        fallback = ValueTranslationEntry(
            original_value=str(value),
            translated_description=str(value),
            brief_properties=[],
            likely_role=str(column or "unknown"),
            confidence="low",
        )
        output = self._invoke_agent_structured(
            self.value_translation_agent,
            prompt_text,
            fallback,
            call_name="value_translation",
        )
        entry = merge_translation_entry(
            cached,
            original_value=str(value),
            column=str(column),
            translated_description=str(output.translated_description or value),
            brief_properties=list(output.brief_properties or []),
            likely_role=str(output.likely_role or column or "unknown"),
            confidence=str(output.confidence or "medium"),
        )
        self._translation_cache[str(value)] = entry
        save_translation_cache(self._translation_cache, self._translation_cache_path)
        return entry

    def _value_annotations_for_payloads(
        self,
        payloads: list[dict[str, Any]] | None,
        *,
        force_translate_missing: bool = False,
        max_entries: int = 40,
    ) -> list[dict[str, Any]]:
        requested = self._collect_requested_values(payloads)
        if force_translate_missing:
            for column, values in requested.items():
                for value in values:
                    if str(value) not in self._translation_cache:
                        self._translate_value(str(value), str(column))
        return render_annotation_context(
            self._translation_cache,
            requested,
            max_entries=max_entries,
        )

    def _trim_init_value_annotations(
        self,
        value_annotations: list[dict[str, Any]],
        dataset_meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
        reaction_type = str(dataset_meta.get("reaction_type", "") or "").lower()
        override_key = "buchwald" if reaction_type.startswith("buchwald") else reaction_type
        override = self.prompt_config.reaction_overrides.get(override_key)
        if override is None:
            return value_annotations

        priority_columns = override.priority_columns or []
        priority_order = {name: idx for idx, name in enumerate(priority_columns)}
        per_column_caps = override.per_column_caps or {}
        kept: list[dict[str, Any]] = []
        kept_per_column: dict[str, int] = {}

        sorted_annotations = sorted(
            value_annotations,
            key=lambda item: (
                priority_order.get(str(item.get("column", "")), 99),
                str(item.get("original_value", "")),
            ),
        )
        for item in sorted_annotations:
            column = str(item.get("column", ""))
            limit = per_column_caps.get(column)
            current = kept_per_column.get(column, 0)
            if limit is not None and current >= limit:
                continue
            kept.append(item)
            kept_per_column[column] = current + 1
        return kept

    def decide_action(
        self,
        decision_context: dict[str, Any],
        search_space,
    ) -> dict[str, Any]:  # noqa: ANN001
        search_info = self._search_space_info(search_space)
        default_vars = [item["name"] for item in search_info[:2]]
        value_annotations = self._value_annotations_for_payloads(
            [decision_context, self._search_space_meta(decision_context)],
            force_translate_missing=False,
        )
        skill_block = self._render_procedural_skill_block(
            "decision_action",
            call_name="decision_action",
        )
        prompt_text = build_decision_action_prompt(
            decision_context=decision_context,
            search_space=search_info,
            search_space_meta=self._search_space_meta(decision_context),
            value_annotations=value_annotations,
            skill_block=skill_block,
        )
        output = self._invoke_agent_structured(
            self.decision_agent,
            prompt_text,
            DecisionAction(active_variables=default_vars),
            call_name="decision_action",
        )
        valid_names = {item["name"] for item in search_info}
        output.active_variables = [
            name for name in output.active_variables if name in valid_names
        ]
        if not output.active_variables:
            output.active_variables = default_vars
            output.reasoning = "Fallback: empty/invalid selection from model output."
        if len(output.active_variables) < 3:
            for name in (item["name"] for item in search_info):
                if name not in output.active_variables:
                    output.active_variables.append(name)
                if len(output.active_variables) >= 3:
                    break
        self._coerce_stage(output, decision_context)
        return output.model_dump()

    def complete_candidate_action(
        self,
        partial_candidate: dict[str, Any],
        decision_action: dict[str, Any],
        decision_context: dict[str, Any],
        search_space,
    ) -> dict[str, Any]:  # noqa: ANN001
        prompt_text = build_completion_action_prompt(
            partial_candidate=partial_candidate,
            decision_action=decision_action,
            decision_context=decision_context,
            search_space=self._search_space_info(search_space),
            search_space_meta=self._search_space_meta(decision_context),
            value_annotations=self._value_annotations_for_payloads(
                [partial_candidate, decision_action, decision_context, self._search_space_meta(decision_context)],
                force_translate_missing=False,
            ),
        )
        fallback = CompletionAction(full_candidate=dict(partial_candidate))
        output = self._invoke_agent_structured(
            self.completion_agent,
            prompt_text,
            fallback,
            call_name="completion_action",
        )
        return output.model_dump()

    def feasibility_action(
        self,
        candidate: dict[str, Any],
        decision_context: dict[str, Any],
        reaction_context: dict[str, Any],
    ) -> dict[str, Any]:
        prompt_text = build_feasibility_action_prompt(
            candidate_condition=candidate,
            reaction_context=reaction_context,
            decision_context=decision_context,
            value_annotations=self._value_annotations_for_payloads(
                [candidate, reaction_context, decision_context],
                force_translate_missing=False,
            ),
        )
        output = self._invoke_agent_structured(
            self.feasibility_agent,
            prompt_text,
            FeasibilityAction(),
            call_name="feasibility_action",
        )
        if output.action == "revise":
            revised_candidate = output.revised_candidate or {}
            cleaned = {k: v for k, v in revised_candidate.items() if k in candidate}
            has_effective_change = any(
                candidate.get(key) != value for key, value in cleaned.items()
            )
            if not has_effective_change:
                output.action = "accept"
                output.revised_candidate = None
                output.reasoning = (
                    f"{output.reasoning} [v0.2.1 auto-accept: revise has no effective change.]"
                )
            else:
                output.revised_candidate = cleaned
        return output.model_dump()

    def reflection_action(
        self,
        decision_action: dict[str, Any],
        candidate: dict[str, Any],
        feasibility_action: dict[str, Any],
        result: float,
        decision_context: dict[str, Any],
    ) -> dict[str, Any]:
        node_view = self._node_state_view(decision_context, "reflection_action")
        knowledge_context, knowledge_meta = self._knowledge_for_prompt(node_view)
        prompt_text = build_reflection_action_prompt(
            decision_action=decision_action,
            candidate=candidate,
            feasibility_action=feasibility_action,
            result=result,
            decision_context=node_view,
            knowledge_context=knowledge_context,
            knowledge_meta=knowledge_meta,
            value_annotations=self._value_annotations_for_payloads(
                [decision_action, candidate, feasibility_action, node_view],
                force_translate_missing=False,
            ),
        )
        output = self._invoke_agent_structured(
            self.reflection_agent,
            prompt_text,
            ReflectionAction(),
            call_name="reflection_action",
        )
        return output.model_dump()

    def generate_hypothesis(
        self,
        decision_context: dict[str, Any],
        reaction_context: dict[str, Any],
        search_space,
    ) -> dict[str, Any]:  # noqa: ANN001
        node_view = self._node_state_view(decision_context, "hypothesis_action")
        knowledge_context, knowledge_meta = self._knowledge_for_prompt(node_view)
        skill_block = self._render_procedural_skill_block(
            "hypothesis_action",
            call_name="hypothesis_action",
        )
        prompt_text = build_hypothesis_action_prompt(
            decision_context=node_view,
            reaction_context=reaction_context,
            search_space=self._search_space_info(search_space),
            search_space_meta=self._search_space_meta(node_view),
            knowledge_context=knowledge_context,
            knowledge_meta=knowledge_meta,
            value_annotations=self._value_annotations_for_payloads(
                [node_view, reaction_context, self._search_space_meta(node_view)],
                force_translate_missing=False,
            ),
            skill_block=skill_block,
        )
        fallback = HypothesisAction(
            hypotheses=["Continue balanced exploration with BO main loop."],
            suggested_focus_variables=[],
        )
        output = self._invoke_agent_structured(
            self.hypothesis_agent,
            prompt_text,
            fallback,
            call_name="hypothesis_action",
        )
        valid_names = {p.name for p in search_space}
        output.suggested_focus_variables = [
            name for name in output.suggested_focus_variables if name in valid_names
        ]
        return output.model_dump()

    def diagnose_stagnation(
        self,
        decision_context: dict[str, Any],
        history_tail: list[dict[str, Any]],
    ) -> dict[str, Any]:
        node_view = self._node_state_view(decision_context, "stagnation_diagnosis")
        knowledge_context, knowledge_meta = self._knowledge_for_prompt(node_view)
        skill_block = self._render_procedural_skill_block(
            "stagnation_diagnosis",
            call_name="stagnation_diagnosis",
        )
        prompt_text = build_stagnation_diagnosis_prompt(
            decision_context=node_view,
            history_tail=history_tail,
            knowledge_context=knowledge_context,
            knowledge_meta=knowledge_meta,
            value_annotations=self._value_annotations_for_payloads(
                [node_view, *history_tail],
                force_translate_missing=False,
            ),
            skill_block=skill_block,
        )
        fallback = StagnationDiagnosis()
        output = self._invoke_agent_structured(
            self.diagnosis_agent,
            prompt_text,
            fallback,
            call_name="stagnation_diagnosis",
        )
        return output.model_dump()

    def analyze_coverage(
        self,
        decision_context: dict[str, Any],
        search_space,
    ) -> dict[str, Any]:  # noqa: ANN001
        node_view = self._node_state_view(decision_context, "coverage_insight")
        prompt_text = build_coverage_insight_prompt(
            decision_context=node_view,
            search_space=self._search_space_info(search_space),
            search_space_meta=self._search_space_meta(node_view),
            value_annotations=self._value_annotations_for_payloads(
                [node_view, self._search_space_meta(node_view)],
                force_translate_missing=False,
            ),
        )
        fallback = CoverageInsight(
            coverage_status="partial",
            repetition_risk="medium",
        )
        output = self._invoke_agent_structured(
            self.coverage_agent,
            prompt_text,
            fallback,
            call_name="coverage_insight",
        )
        valid_names = {p.name for p in search_space}
        output.underexplored_dimensions = [
            name for name in output.underexplored_dimensions if name in valid_names
        ]
        return output.model_dump()

    def plan_intervention(
        self,
        decision_context: dict[str, Any],
        diagnosis: dict[str, Any],
        hypothesis_action: dict[str, Any],
        coverage_insight: dict[str, Any],
        search_space,
    ) -> dict[str, Any]:  # noqa: ANN001
        knowledge_context, knowledge_meta = self._knowledge_for_prompt(decision_context)
        prompt_text = build_intervention_plan_prompt(
            decision_context=decision_context,
            diagnosis=diagnosis,
            hypothesis_action=hypothesis_action,
            coverage_insight=coverage_insight,
            search_space=self._search_space_info(search_space),
            search_space_meta=self._search_space_meta(decision_context),
            knowledge_context=knowledge_context,
            knowledge_meta=knowledge_meta,
            value_annotations=self._value_annotations_for_payloads(
                [decision_context, diagnosis, hypothesis_action, coverage_insight, self._search_space_meta(decision_context)],
                force_translate_missing=False,
            ),
        )
        fallback = InterventionPlan(
            intervention_type="none",
            use_subspace=False,
            focus_variables=[],
            window_rounds=0,
        )
        output = self._invoke_agent_structured(
            self.intervention_agent,
            prompt_text,
            fallback,
            call_name="intervention_plan",
        )
        valid_names = {p.name for p in search_space}
        output.focus_variables = [name for name in output.focus_variables if name in valid_names]
        if output.use_subspace and len(output.focus_variables) == 0:
            output.use_subspace = False
            output.intervention_type = "soft_guidance"
        if output.window_rounds < 0:
            output.window_rounds = 0
        return output.model_dump()

    @staticmethod
    def _legacy_action_package_from_mode(
        *,
        mode: str,
        focus_variables: list[str],
        window_rounds: int,
        reasoning: str,
        schema_version: str,
    ) -> dict[str, Any]:
        if mode == "bo_focus_then_rerank":
            return {
                "schema_version": schema_version,
                "intent": "probe",
                "shortlist_policy": "coverage_shape",
                "repeat_policy": "avoid_anchor_repeat",
                "selection_policy": "select_from_shaped_shortlist",
                "verification_policy": "normal",
                "focus_policy": "temporary_focus",
                "focus_variables": list(focus_variables),
                "window_rounds": int(window_rounds),
                "reasoning": reasoning,
            }
        if mode == "bo_rerank_topk":
            return {
                "schema_version": schema_version,
                "intent": "balance",
                "shortlist_policy": "plain",
                "repeat_policy": "allow",
                "selection_policy": "select_from_shaped_shortlist",
                "verification_policy": "normal",
                "focus_policy": "full_space",
                "focus_variables": [],
                "window_rounds": 0,
                "reasoning": reasoning,
            }
        return {
            "schema_version": schema_version,
            "intent": "exploit",
            "shortlist_policy": "plain",
            "repeat_policy": "allow",
            "selection_policy": "bo_top1",
            "verification_policy": "normal",
            "focus_policy": "full_space",
            "focus_variables": [],
            "window_rounds": 0,
            "reasoning": reasoning,
        }

    @staticmethod
    def _append_reasoning_note(reasoning: str, note: str) -> str:
        return append_reasoning_note(reasoning, note)

    @staticmethod
    def _controller_policy_signals(
        *,
        decision_context: dict[str, Any],
        controller_trigger_reasons: list[str],
    ) -> dict[str, Any]:
        return controller_policy_signals(
            decision_context=decision_context,
            controller_trigger_reasons=controller_trigger_reasons,
        )

    @staticmethod
    def _controller_v06_action_policy(
        *,
        decision_context: dict[str, Any],
        controller_trigger_reasons: list[str],
    ) -> dict[str, Any]:
        return build_v06_action_admissibility(
            decision_context=decision_context,
            controller_trigger_reasons=controller_trigger_reasons,
        )

    def _normalize_legacy_controller_plan(
        self,
        *,
        output: InterventionPlan,
        controller_trigger_reasons: list[str],
        decision_context: dict[str, Any],
    ) -> dict[str, Any]:
        mode = output.intervention_type
        if mode not in {"bo_direct", "bo_rerank_topk", "bo_focus_then_rerank"}:
            mode = "bo_direct"
        output.intervention_type = mode
        trigger_set = set(controller_trigger_reasons)
        no_improvement_rounds = int(decision_context.get("no_improvement_rounds", 0) or 0)
        if mode == "bo_focus_then_rerank":
            output.use_subspace = len(output.focus_variables) > 0
            if not output.use_subspace:
                output.intervention_type = "bo_rerank_topk"
                output.reasoning = (
                    f"{output.reasoning} [fallback: invalid focus_variables, downgrade to shortlist rerank.]"
                )
            elif not (
                "coverage_low" in trigger_set
                and "scaffold_concentration_high" in trigger_set
                and no_improvement_rounds >= 4
            ):
                output.intervention_type = "bo_rerank_topk"
                output.use_subspace = False
                output.focus_variables = []
                output.window_rounds = 0
                output.reasoning = (
                    f"{output.reasoning} [fallback: focus_then_rerank requires coverage_low + scaffold_concentration_high + sustained no-improvement.]"
                )
        else:
            output.use_subspace = False
            output.focus_variables = []
            output.window_rounds = 0
        payload = output.model_dump()
        payload["action_package"] = self._legacy_action_package_from_mode(
            mode=str(output.intervention_type),
            focus_variables=list(output.focus_variables),
            window_rounds=int(output.window_rounds),
            reasoning=str(output.reasoning),
            schema_version="compat_v1",
        )
        return payload

    def _normalize_action_package_controller_plan(
        self,
        *,
        output: InterventionPlan,
        controller_trigger_reasons: list[str],
        decision_context: dict[str, Any],
    ) -> dict[str, Any]:
        signals = self._controller_policy_signals(
            decision_context=decision_context,
            controller_trigger_reasons=controller_trigger_reasons,
        )
        trigger_set = signals["trigger_set"]
        no_improvement_rounds = int(signals["no_improvement_rounds"])

        intent = str(output.intent or "balance").strip().lower()
        if intent not in {"exploit", "probe", "balance"}:
            intent = "balance"
        shortlist_policy = str(output.shortlist_policy or "plain").strip().lower()
        if shortlist_policy not in {"plain", "diversity_shape", "coverage_shape", "contrast_shape"}:
            shortlist_policy = "plain"
        repeat_policy = str(output.repeat_policy or "allow").strip().lower()
        if repeat_policy not in {"allow", "avoid_near_duplicate", "avoid_anchor_repeat"}:
            repeat_policy = "allow"
        selection_policy = str(output.selection_policy or "bo_top1").strip().lower()
        if selection_policy not in {
            "bo_top1",
            "bo_top1_from_shaped_shortlist",
            "select_from_shaped_shortlist",
        }:
            selection_policy = "bo_top1"
        verification_policy = str(output.verification_policy or "normal").strip().lower()
        if verification_policy not in {"normal", "strict"}:
            verification_policy = "normal"
        focus_policy = str(output.focus_policy or "full_space").strip().lower()
        if focus_policy not in {"full_space", "temporary_focus"}:
            focus_policy = "full_space"

        stabilized = realign_action_package_fields(
            intent=intent,
            shortlist_policy=shortlist_policy,
            repeat_policy=repeat_policy,
            selection_policy=selection_policy,
            verification_policy=verification_policy,
            focus_policy=focus_policy,
            reasoning=str(output.reasoning),
            decision_context=decision_context,
            controller_trigger_reasons=controller_trigger_reasons,
        )
        intent = str(stabilized["intent"])
        shortlist_policy = str(stabilized["shortlist_policy"])
        repeat_policy = str(stabilized["repeat_policy"])
        selection_policy = str(stabilized["selection_policy"])
        verification_policy = str(stabilized["verification_policy"])
        focus_policy = str(stabilized["focus_policy"])
        output.reasoning = str(stabilized["reasoning"])

        if selection_policy == "bo_top1":
            focus_policy = "full_space"
            output.focus_variables = []
            output.window_rounds = 0

        if focus_policy == "temporary_focus":
            output.use_subspace = len(output.focus_variables) > 0
            if not output.use_subspace:
                focus_policy = "full_space"
                output.reasoning = (
                    f"{output.reasoning} [fallback: temporary_focus requires valid focus_variables.]"
                )
            elif not (
                "coverage_low" in trigger_set
                and "scaffold_concentration_high" in trigger_set
                and no_improvement_rounds >= 4
            ):
                focus_policy = "full_space"
                output.use_subspace = False
                output.focus_variables = []
                output.window_rounds = 0
                output.reasoning = (
                    f"{output.reasoning} [fallback: temporary_focus requires coverage_low + scaffold_concentration_high + sustained no-improvement.]"
                )
        else:
            output.use_subspace = False
            output.focus_variables = []
            output.window_rounds = 0

        if selection_policy in {
            "bo_top1_from_shaped_shortlist",
            "select_from_shaped_shortlist",
        }:
            mode = "bo_focus_then_rerank" if focus_policy == "temporary_focus" else "bo_rerank_topk"
        else:
            mode = "bo_direct"

        action_package = {
            "schema_version": "v2",
            "intent": intent,
            "shortlist_policy": shortlist_policy,
            "repeat_policy": repeat_policy,
            "selection_policy": selection_policy,
            "verification_policy": verification_policy,
            "focus_policy": focus_policy,
            "focus_variables": list(output.focus_variables),
            "window_rounds": int(output.window_rounds),
            "reasoning": str(output.reasoning),
        }

        output.action_schema_version = "v2"
        output.intent = intent
        output.shortlist_policy = shortlist_policy
        output.repeat_policy = repeat_policy
        output.selection_policy = selection_policy
        output.verification_policy = verification_policy
        output.focus_policy = focus_policy
        output.intervention_type = mode
        output.use_subspace = focus_policy == "temporary_focus"
        payload = output.model_dump()
        payload["action_package"] = action_package
        return payload

    def _normalize_v06_controller_plan(
        self,
        *,
        output: InterventionPlan,
        controller_trigger_reasons: list[str],
        decision_context: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = normalize_v06_action_package(
            requested_execution_action=str(output.requested_execution_action or "direct_bo_pick"),
            intent=str(output.intent or "balance"),
            shortlist_policy=str(output.shortlist_policy or "plain"),
            repeat_policy=str(output.repeat_policy or "allow"),
            verification_policy=str(output.verification_policy or "normal"),
            focus_variables=list(output.focus_variables or []),
            window_rounds=int(output.window_rounds or 0),
            reasoning=str(output.reasoning or ""),
            decision_context=decision_context,
            controller_trigger_reasons=controller_trigger_reasons,
        )
        output.action_schema_version = "v0.6"
        output.requested_execution_action = str(normalized["requested_execution_action"])
        output.intent = str(normalized["intent"])
        output.shortlist_policy = str(normalized["shortlist_policy"])
        output.repeat_policy = str(normalized["repeat_policy"])
        output.selection_policy = str(normalized["selection_policy"])
        output.verification_policy = str(normalized["verification_policy"])
        output.focus_policy = str(normalized["focus_policy"])
        output.focus_variables = list(normalized["focus_variables"])
        output.window_rounds = int(normalized["window_rounds"])
        output.reasoning = str(normalized["reasoning"])
        output.use_subspace = output.focus_policy == "temporary_focus"
        output.intervention_type = "bo_direct"
        action_package = {
            "schema_version": "v0.6",
            "intent": str(normalized["intent"]),
            "shortlist_policy": str(normalized["shortlist_policy"]),
            "repeat_policy": str(normalized["repeat_policy"]),
            "selection_policy": str(normalized["selection_policy"]),
            "verification_policy": str(normalized["verification_policy"]),
            "focus_policy": str(normalized["focus_policy"]),
            "focus_variables": list(normalized["focus_variables"]),
            "window_rounds": int(normalized["window_rounds"]),
            "requested_execution_action": str(normalized["requested_execution_action"]),
            "admissible_execution_actions": list(normalized["admissible_execution_actions"]),
            "preferred_execution_action": str(normalized["preferred_execution_action"]),
            "planner_action_policy_name": normalized.get("planner_action_policy_name"),
            "selection_authority_level": normalized.get("selection_authority_level"),
            "allowed_mainline_actions": list(normalized.get("allowed_mainline_actions", [])),
            "allowed_rare_actions": list(normalized.get("allowed_rare_actions", [])),
            "fallback_reason": normalized.get("fallback_reason"),
            "reasoning": str(normalized["reasoning"]),
        }
        if str(normalized["requested_execution_action"]) == "finite_pool_candidate_probe":
            action_package["candidate_probe_include"] = list(output.candidate_probe_include or [])
            action_package["candidate_probe_reasoning"] = str(output.candidate_probe_reasoning or "")
        payload = output.model_dump()
        payload["admissible_execution_actions"] = list(normalized["admissible_execution_actions"])
        payload["preferred_execution_action"] = str(normalized["preferred_execution_action"])
        payload["planner_action_policy_name"] = normalized.get("planner_action_policy_name")
        payload["selection_authority_level"] = normalized.get("selection_authority_level")
        payload["fallback_reason"] = normalized.get("fallback_reason")
        payload["action_package"] = action_package
        return payload

    def choose_controller_plan(
        self,
        decision_context: dict[str, Any],
        diagnosis: dict[str, Any],
        hypothesis_action: dict[str, Any],
        coverage_insight: dict[str, Any],
        controller_trigger_reasons: list[str],
        search_space,
        *,
        enable_action_package_v2: bool = False,
        enable_action_package_v06: bool = False,
    ) -> dict[str, Any]:  # noqa: ANN001
        node_view = self._node_state_view(decision_context, "controller_plan")
        execution_policy = None
        if enable_action_package_v06:
            execution_policy = self._controller_v06_action_policy(
                decision_context=decision_context,
                controller_trigger_reasons=controller_trigger_reasons,
            )
        skill_block = self._render_procedural_skill_block(
            "controller_plan",
            call_name="controller_plan",
        )
        prompt_text = build_controller_plan_prompt(
            decision_context=node_view,
            diagnosis=diagnosis,
            hypothesis_action=hypothesis_action,
            coverage_insight=coverage_insight,
            search_space=self._search_space_info(search_space),
            search_space_meta=self._search_space_meta(node_view),
            controller_trigger_reasons=controller_trigger_reasons,
            value_annotations=self._value_annotations_for_payloads(
                [node_view, diagnosis, hypothesis_action, coverage_insight, self._search_space_meta(node_view)],
                force_translate_missing=False,
            ),
            skill_block=skill_block,
            enable_action_package_v2=enable_action_package_v2,
            enable_action_package_v06=enable_action_package_v06,
            admissible_execution_actions=(
                list(execution_policy.get("admissible_execution_actions", []))
                if isinstance(execution_policy, dict)
                else None
            ),
            preferred_execution_action=(
                str(execution_policy.get("preferred_execution_action"))
                if isinstance(execution_policy, dict)
                else None
            ),
        )
        fallback = InterventionPlan(
            intervention_type="bo_direct",
            action_schema_version=(
                "v0.6"
                if enable_action_package_v06
                else ("v2" if enable_action_package_v2 else "compat_v1")
            ),
            requested_execution_action=(
                str(execution_policy.get("preferred_execution_action", "direct_bo_pick"))
                if isinstance(execution_policy, dict)
                else "direct_bo_pick"
            ),
            intent="balance",
            shortlist_policy="plain",
            repeat_policy="allow",
            selection_policy="bo_top1",
            verification_policy="normal",
            focus_policy="full_space",
            use_subspace=False,
            focus_variables=[],
            window_rounds=0,
            reasoning="Fallback: keep BO direct.",
        )
        output = self._invoke_agent_structured(
            self.intervention_agent,
            prompt_text,
            fallback,
            call_name="controller_plan",
        )
        valid_names = {p.name for p in search_space}
        output.focus_variables = [name for name in output.focus_variables if name in valid_names]
        if output.window_rounds < 0:
            output.window_rounds = 0

        if enable_action_package_v06:
            return self._normalize_v06_controller_plan(
                output=output,
                controller_trigger_reasons=controller_trigger_reasons,
                decision_context=decision_context,
            )
        if enable_action_package_v2:
            return self._normalize_action_package_controller_plan(
                output=output,
                controller_trigger_reasons=controller_trigger_reasons,
                decision_context=decision_context,
            )
        return self._normalize_legacy_controller_plan(
            output=output,
            controller_trigger_reasons=controller_trigger_reasons,
            decision_context=decision_context,
        )

    def compose_lab_batch(
        self,
        *,
        decision_context: dict[str, Any],
        candidate_pool: list[dict[str, Any]],
        controller_plan: dict[str, Any],
        diagnosis: dict[str, Any],
        hypothesis_action: dict[str, Any],
        coverage_insight: dict[str, Any],
        batch_size: int,
        search_space,  # noqa: ANN001
        reaction_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Design a whole real-lab batch as one bounded portfolio decision."""

        node_view = self._node_state_view(decision_context, "lab_batch_composition")
        knowledge_context, knowledge_meta = self._knowledge_for_prompt(node_view)
        skill_block = self._render_procedural_skill_block(
            "lab_batch_composition",
            call_name="lab_batch_composition",
        )
        prompt_text = build_lab_batch_composition_prompt(
            decision_context=node_view,
            candidate_pool=candidate_pool,
            controller_plan=controller_plan,
            diagnosis=diagnosis,
            hypothesis_action=hypothesis_action,
            coverage_insight=coverage_insight,
            batch_size=batch_size,
            search_space=self._search_space_info(search_space),
            search_space_meta=self._search_space_meta(node_view),
            reaction_context=reaction_context,
            knowledge_context=knowledge_context,
            knowledge_meta=knowledge_meta,
            value_annotations=self._value_annotations_for_payloads(
                [
                    node_view,
                    controller_plan,
                    diagnosis,
                    hypothesis_action,
                    coverage_insight,
                    reaction_context,
                    *candidate_pool,
                    self._search_space_meta(node_view),
                ],
                force_translate_missing=False,
            ),
            skill_block=skill_block,
        )
        fallback_slots = [
            LabBatchSlotAction(
                slot_id=idx + 1,
                role="planner_anchor" if idx == 0 else "batch_diversity_probe",
                candidate_index=int(item.get("candidate_index", idx) or idx),
                purpose=(
                    "Keep the planner's strongest candidate visible."
                    if idx == 0
                    else "Add a legal candidate that can broaden the batch portfolio."
                ),
                rationale=(
                    "Fallback slot: planner anchor."
                    if idx == 0
                    else "Fallback slot: selected from the bounded planner pool."
                ),
            )
            for idx, item in enumerate(candidate_pool[: max(1, int(batch_size))])
        ]
        fallback = LabBatchCompositionAction(
            batch_strategy="fallback_planner_anchor_diversity",
            batch_rationale=(
                "Fallback: keep the planner anchor and fill the remaining slots "
                "from the admissible planner pool."
            ),
            global_constraints=[
                "select only candidate_index values from the provided pool",
                "avoid duplicate candidates",
            ],
            slots=fallback_slots,
        )
        output = self._invoke_agent_structured(
            self.lab_batch_composition_agent,
            prompt_text,
            fallback,
            call_name="lab_batch_composition",
        )
        valid_indices = {
            int(item.get("candidate_index", idx) or idx)
            for idx, item in enumerate(candidate_pool)
        }
        cleaned_slots: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        for idx, slot in enumerate(output.slots, start=1):
            candidate_index = int(slot.candidate_index)
            if candidate_index not in valid_indices or candidate_index in seen_indices:
                continue
            seen_indices.add(candidate_index)
            cleaned_slots.append(
                {
                    "slot_id": int(slot.slot_id or idx),
                    "role": str(slot.role or "batch_member"),
                    "candidate_index": candidate_index,
                    "purpose": str(slot.purpose or ""),
                    "varied_variables": list(slot.varied_variables or []),
                    "controlled_variables": list(slot.controlled_variables or []),
                    "rationale": str(slot.rationale or ""),
                    "risk_note": str(slot.risk_note or ""),
                    "evidence_refs": list(slot.evidence_refs or []),
                }
            )
            if len(cleaned_slots) >= max(1, int(batch_size)):
                break
        return {
            "batch_strategy": str(output.batch_strategy or "planner_anchor_diversity"),
            "batch_rationale": str(output.batch_rationale or ""),
            "global_constraints": list(output.global_constraints or []),
            "slots": cleaned_slots,
        }

    def rerank_shortlist(
        self,
        decision_context: dict[str, Any],
        shortlist_candidates: list[dict[str, Any]],
        controller_plan: dict[str, Any],
        rerank_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        rerank_policy = dict(rerank_policy or {})
        state_router_guidance = dict(rerank_policy.get("state_router_guidance") or {})

        def _normalize_indices(values: Any) -> list[int]:
            ordered: list[int] = []
            seen: set[int] = set()
            for value in values or []:
                try:
                    idx = int(value)
                except (TypeError, ValueError):
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                ordered.append(idx)
            return ordered

        prompt_style = str(rerank_policy.get("prompt_style", "default"))
        hybrid_policy = dict(rerank_policy.get("resuggest_hybrid_policy") or {})
        requested_execution_action = str(
            (controller_plan.get("action_package") or {}).get("requested_execution_action")
            or controller_plan.get("requested_execution_action")
            or ""
        ).strip()
        corridor_probe_mode = (
            prompt_style == "resuggest_probe_topk"
            and requested_execution_action == "mask_scaffold_corridor_resuggest"
        )
        if prompt_style == "resuggest_probe_topk" and shortlist_candidates:
            hybrid_planner_rank_topk = max(
                2,
                int(hybrid_policy.get("planner_rank_topk", 3) or 3),
            )
            if corridor_probe_mode:
                hybrid_planner_rank_topk = max(hybrid_planner_rank_topk, 4)
            fallback_candidate_index = int(
                state_router_guidance.get(
                    "fallback_candidate_index",
                    shortlist_candidates[0].get("candidate_index", 0),
                )
                or shortlist_candidates[0].get("candidate_index", 0)
            )
            controlled_non_top_indices: list[int] = []
            for idx, item in enumerate(shortlist_candidates):
                candidate_index = int(item.get("candidate_index", idx) or idx)
                if candidate_index == fallback_candidate_index:
                    continue
                bo_rank = int(item.get("bo_rank", idx + 1) or (idx + 1))
                main_pool_rank = int(item.get("main_pool_rank", idx + 1) or (idx + 1))
                pool_source = str(item.get("pool_source", "main_pool"))
                shortlist_source = str(item.get("shortlist_source", "bo_top_ranked"))
                if (
                    bo_rank <= hybrid_planner_rank_topk
                    or main_pool_rank <= hybrid_planner_rank_topk
                    or pool_source == "diversity_pool"
                    or shortlist_source == "diversity_injected"
                ):
                    controlled_non_top_indices.append(candidate_index)
            existing_preferred = _normalize_indices(
                state_router_guidance.get("preferred_candidate_indices")
            )
            controlled_preferred = [
                idx for idx in existing_preferred if idx in set(controlled_non_top_indices)
            ]
            if controlled_non_top_indices:
                state_router_guidance = {
                    **state_router_guidance,
                    "admissible_candidate_indices": [
                        int(fallback_candidate_index),
                        *controlled_non_top_indices,
                    ],
                    "preferred_candidate_indices": (
                        controlled_preferred or controlled_non_top_indices
                    ),
                    "rationale": (
                        f"{state_router_guidance.get('rationale', '')} "
                        "Post-resuggest hybrid probe is constrained to the planner near-top slice."
                    ).strip(),
                }
                rerank_policy = {
                    **rerank_policy,
                    "state_router_guidance": state_router_guidance,
                }
        elif prompt_style == "shape_probe_topk" and shortlist_candidates:
            shape_policy = dict(rerank_policy.get("shape_hybrid_policy") or {})
            shape_planner_rank_topk = max(2, int(shape_policy.get("planner_rank_topk", 3) or 3))
            fallback_candidate_index = int(
                state_router_guidance.get(
                    "fallback_candidate_index",
                    shortlist_candidates[0].get("candidate_index", 0),
                )
                or shortlist_candidates[0].get("candidate_index", 0)
            )
            controlled_non_top_indices: list[int] = []
            for idx, item in enumerate(shortlist_candidates):
                candidate_index = int(item.get("candidate_index", idx) or idx)
                if candidate_index == fallback_candidate_index:
                    continue
                bo_rank = int(item.get("bo_rank", idx + 1) or (idx + 1))
                main_pool_rank = int(item.get("main_pool_rank", idx + 1) or (idx + 1))
                pool_source = str(item.get("pool_source", "main_pool"))
                shortlist_source = str(item.get("shortlist_source", "bo_top_ranked"))
                if (
                    bo_rank <= shape_planner_rank_topk
                    or main_pool_rank <= shape_planner_rank_topk
                    or pool_source == "diversity_pool"
                    or shortlist_source == "diversity_injected"
                ):
                    controlled_non_top_indices.append(candidate_index)
            existing_preferred = _normalize_indices(
                state_router_guidance.get("preferred_candidate_indices")
            )
            controlled_preferred = [
                idx for idx in existing_preferred if idx in set(controlled_non_top_indices)
            ]
            if controlled_non_top_indices:
                state_router_guidance = {
                    **state_router_guidance,
                    "preferred_candidate_indices": (
                        controlled_preferred or controlled_non_top_indices
                    ),
                    "rationale": (
                        f"{state_router_guidance.get('rationale', '')} "
                        "Shape-only probe is constrained to a controlled planner near-top slice."
                    ).strip(),
                }
                rerank_policy = {
                    **rerank_policy,
                    "state_router_guidance": state_router_guidance,
                }

        skill_block = self._render_procedural_skill_block(
            "shortlist_rerank",
            call_name="shortlist_rerank",
        )
        prompt_text = build_shortlist_rerank_prompt(
            decision_context=decision_context,
            shortlist_candidates=shortlist_candidates,
            controller_plan=controller_plan,
            search_space_meta=self._search_space_meta(decision_context),
            value_annotations=self._value_annotations_for_payloads(
                [decision_context, controller_plan, *shortlist_candidates, self._search_space_meta(decision_context)],
                force_translate_missing=False,
            ),
            rerank_policy=rerank_policy,
            skill_block=skill_block,
        )
        fallback = ShortlistRerankAction(
            selected_index=0,
            candidate_scores=[
                ShortlistCandidateScore(
                    candidate_index=item.get("candidate_index", idx),
                    overall_score=max(0.0, 1.0 - 0.05 * idx),
                    plausibility_score=0.6,
                    novelty_score=max(0.0, 0.6 - 0.03 * idx),
                    transfer_value_score=0.5,
                    hypothesis_value_score=0.5,
                    structural_shift_type="none",
                    hypothesis_summary="Fallback: no explicit hypothesis.",
                    local_overfit_risk="medium",
                    reasoning="Fallback: preserve BO shortlist order.",
                )
                for idx, item in enumerate(shortlist_candidates)
            ],
            reasoning="Fallback: keep BO top-1 shortlist candidate.",
        )
        output = self._invoke_agent_structured(
            self.shortlist_rerank_agent,
            prompt_text,
            fallback,
            call_name="shortlist_rerank",
        )
        valid_indices = {item.get("candidate_index", idx) for idx, item in enumerate(shortlist_candidates)}
        shortlist_fallback_index = int(shortlist_candidates[0].get("candidate_index", 0)) if shortlist_candidates else 0
        fallback_candidate_index = int(
            state_router_guidance.get("fallback_candidate_index", shortlist_fallback_index) or shortlist_fallback_index
        )
        if fallback_candidate_index not in valid_indices:
            fallback_candidate_index = shortlist_fallback_index
        admissible_candidate_indices = _normalize_indices(
            state_router_guidance.get("admissible_candidate_indices")
        )
        admissible_candidate_indices = [
            idx for idx in admissible_candidate_indices if idx in valid_indices
        ] or [fallback_candidate_index]
        admissible_candidate_set = set(admissible_candidate_indices)
        preferred_candidate_indices = _normalize_indices(
            state_router_guidance.get("preferred_candidate_indices")
        )
        preferred_candidate_indices = [
            idx for idx in preferred_candidate_indices if idx in admissible_candidate_set
        ] or [fallback_candidate_index]
        effective_preferred_candidate_indices = list(preferred_candidate_indices)
        preferred_candidate_set = set(preferred_candidate_indices)
        preferred_bonus = {
            idx: len(preferred_candidate_indices) - pos
            for pos, idx in enumerate(preferred_candidate_indices)
        }
        visible_evidence_state = str(state_router_guidance.get("visible_evidence_state", "") or "")
        preferred_selection_policy = str(
            state_router_guidance.get("preferred_selection_policy", "") or ""
        )
        requested_selected_index = int(output.selected_index)
        selection_status = "selected_requested_candidate"
        if requested_selected_index not in valid_indices:
            output.selected_index = fallback_candidate_index
            selection_status = "invalid_requested_index_fallback"
        elif requested_selected_index not in admissible_candidate_set:
            output.selected_index = fallback_candidate_index
            selection_status = "inadmissible_requested_index_fallback"
        cleaned_scores: list[dict[str, Any]] = []
        seen_indices: set[int] = set()
        for item in output.candidate_scores:
            if item.candidate_index not in valid_indices or item.candidate_index in seen_indices:
                continue
            seen_indices.add(item.candidate_index)
            cleaned_scores.append(
                {
                    "candidate_index": int(item.candidate_index),
                    "overall_score": max(0.0, min(1.0, float(item.overall_score))),
                    "plausibility_score": max(0.0, min(1.0, float(item.plausibility_score))),
                    "novelty_score": max(0.0, min(1.0, float(item.novelty_score))),
                    "transfer_value_score": max(0.0, min(1.0, float(item.transfer_value_score))),
                    "hypothesis_value_score": max(
                        0.0,
                        min(1.0, float(item.hypothesis_value_score)),
                    ),
                    "structural_shift_type": item.structural_shift_type,
                    "hypothesis_summary": item.hypothesis_summary,
                    "local_overfit_risk": item.local_overfit_risk,
                    "reasoning": item.reasoning,
                }
            )
        for idx, item in enumerate(shortlist_candidates):
            candidate_index = item.get("candidate_index", idx)
            if candidate_index in seen_indices:
                continue
            cleaned_scores.append(
                {
                    "candidate_index": int(candidate_index),
                    "overall_score": max(0.0, 1.0 - 0.05 * idx),
                    "plausibility_score": 0.6,
                    "novelty_score": max(0.0, 0.6 - 0.03 * idx),
                    "transfer_value_score": 0.5,
                    "hypothesis_value_score": 0.5,
                    "structural_shift_type": "none",
                    "hypothesis_summary": "Filled missing hypothesis score with fallback.",
                    "local_overfit_risk": "medium",
                    "reasoning": "Filled missing shortlist score with fallback.",
                }
            )
        cleaned_scores = sorted(cleaned_scores, key=lambda item: item["candidate_index"])
        near_best_margin = float(rerank_policy.get("near_best_margin", 0.03) or 0.03)
        risk_rank = {"low": 2, "medium": 1, "high": 0}
        contrastive_evidence_by_index = dict(
            ((rerank_policy.get("candidate_contrastive_evidence") or {}).get("by_candidate_index") or {})
        )

        def _history_condition_variant_delta(candidate_index: int) -> float | None:
            evidence = contrastive_evidence_by_index.get(str(int(candidate_index))) or {}
            changed_scaffold_dims = list(evidence.get("changed_scaffold_dims") or [])
            changed_condition_dims = list(evidence.get("changed_condition_dims") or [])
            if changed_scaffold_dims or not changed_condition_dims:
                return None
            deltas: list[float] = []
            for item in evidence.get("changed_dimensions") or []:
                delta = item.get("candidate_minus_bo_best")
                try:
                    value = float(delta)
                except (TypeError, ValueError):
                    continue
                deltas.append(value)
            if not deltas:
                return None
            return max(deltas)

        def _support_gap(candidate_index: int) -> float:
            evidence = contrastive_evidence_by_index.get(str(int(candidate_index))) or {}
            deltas: list[float] = []
            for block_name in ("same_anchor_support", "same_scaffold_support", "analogue_support"):
                value = (evidence.get(block_name) or {}).get("candidate_minus_bo_best")
                try:
                    deltas.append(float(value))
                except (TypeError, ValueError):
                    continue
            for item in evidence.get("changed_dimensions") or []:
                value = item.get("candidate_minus_bo_best")
                try:
                    deltas.append(float(value))
                except (TypeError, ValueError):
                    continue
            return max(deltas) if deltas else 0.0

        def _anchor_match_count(candidate_index: int) -> int:
            evidence = contrastive_evidence_by_index.get(str(int(candidate_index))) or {}
            return int(
                ((evidence.get("same_anchor_support") or {}).get("candidate") or {}).get("match_count", 0) or 0
            )

        def _same_scaffold_match_count(candidate_index: int) -> int:
            evidence = contrastive_evidence_by_index.get(str(int(candidate_index))) or {}
            return int(
                ((evidence.get("same_scaffold_support") or {}).get("candidate") or {}).get("match_count", 0) or 0
            )

        def _changed_condition_count(candidate_index: int) -> int:
            evidence = contrastive_evidence_by_index.get(str(int(candidate_index))) or {}
            return len(list(evidence.get("changed_condition_dims") or []))

        def _changed_scaffold_count(candidate_index: int) -> int:
            evidence = contrastive_evidence_by_index.get(str(int(candidate_index))) or {}
            return len(list(evidence.get("changed_scaffold_dims") or []))

        def _corridor_probe_has_real_support(candidate_index: int) -> bool:
            return (
                _anchor_match_count(candidate_index) > 0
                or _same_scaffold_match_count(candidate_index) > 0
                or _support_gap(candidate_index) > 0.0
            )

        shortlist_meta_by_index = {
            int(item.get("candidate_index", idx)): {
                "recent_scaffold_hits": int(item.get("recent_scaffold_hits", 0) or 0),
                "recent_primary_dim_hits": int(item.get("recent_primary_dim_hits", 0) or 0),
                "recent_secondary_dim_hits": int(item.get("recent_secondary_dim_hits", 0) or 0),
                "bo_rank": int(item.get("bo_rank", idx + 1) or (idx + 1)),
                "main_pool_rank": int(item.get("main_pool_rank", idx + 1) or (idx + 1)),
                "shortlist_source": str(item.get("shortlist_source", "bo_top_ranked")),
                "pool_source": str(item.get("pool_source", "main_pool")),
            }
            for idx, item in enumerate(shortlist_candidates)
        }
        best_overall = max((item["overall_score"] for item in cleaned_scores), default=0.0)
        admissible_scores = [
            item for item in cleaned_scores if int(item["candidate_index"]) in admissible_candidate_set
        ]
        requested_score_item = next(
            (
                item
                for item in cleaned_scores
                if int(item["candidate_index"]) == int(output.selected_index)
            ),
            None,
        )
        requested_score_gap = (
            None
            if requested_score_item is None
            else max(0.0, best_overall - float(requested_score_item.get("overall_score", 0.0) or 0.0))
        )
        near_best = [
            item for item in cleaned_scores if abs(item["overall_score"] - best_overall) <= near_best_margin
        ]
        admissible_near_best = [
            item for item in near_best if int(item["candidate_index"]) in admissible_candidate_set
        ]
        candidates_for_choice = admissible_near_best or admissible_scores or cleaned_scores
        requested_commit_window_applied = False
        hybrid_probe_mode = prompt_style in {"resuggest_probe_topk", "shape_probe_topk"}
        hybrid_planner_rank_topk = max(2, int(hybrid_policy.get("planner_rank_topk", 3) or 3))
        shape_hybrid_policy = dict(rerank_policy.get("shape_hybrid_policy") or {})
        shape_planner_rank_topk = max(2, int(shape_hybrid_policy.get("planner_rank_topk", 3) or 3))
        if hybrid_probe_mode:
            hybrid_candidate_source = admissible_scores or cleaned_scores
            controlled_probe_candidates = [
                item
                for item in hybrid_candidate_source
                if (
                    int(item["candidate_index"]) == int(fallback_candidate_index)
                    or shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get("bo_rank", 9999)
                    <= (
                        hybrid_planner_rank_topk
                        if prompt_style == "resuggest_probe_topk"
                        else shape_planner_rank_topk
                    )
                    or shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get("main_pool_rank", 9999)
                    <= (
                        hybrid_planner_rank_topk
                        if prompt_style == "resuggest_probe_topk"
                        else shape_planner_rank_topk
                    )
                    or shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                        "pool_source",
                        "main_pool",
                    )
                    == "diversity_pool"
                    or shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                        "shortlist_source",
                        "bo_top_ranked",
                    )
                    == "diversity_injected"
                )
            ]
            candidates_for_choice = controlled_probe_candidates or candidates_for_choice
        enforce_preferred_probe = bool(
            prompt_style == "shape_probe_topk"
            and preferred_selection_policy in {"shape_probe_topk", "deeper_diversity_probe"}
            and effective_preferred_candidate_indices
            and int(fallback_candidate_index) not in preferred_candidate_set
        )
        if enforce_preferred_probe:
            preferred_probe_candidates = [
                item
                for item in candidates_for_choice
                if int(item["candidate_index"]) in preferred_candidate_set
                and str(item.get("local_overfit_risk", "medium")) != "high"
            ]
            if preferred_probe_candidates:
                candidates_for_choice = preferred_probe_candidates
        if (
            requested_score_item is not None
            and int(output.selected_index) in admissible_candidate_set
            and (
                int(output.selected_index) in preferred_candidate_set
                or not hybrid_probe_mode
            )
            and (
                not hybrid_probe_mode
                or int(output.selected_index)
                in {int(item["candidate_index"]) for item in candidates_for_choice}
            )
            and int(output.selected_index)
            not in {int(item["candidate_index"]) for item in candidates_for_choice}
            and visible_evidence_state in {"local_lock", "rank_uncertain"}
            and preferred_selection_policy in {"low_repeat_probe", "challenger_pick"}
            and requested_score_gap is not None
            and requested_score_gap
            <= (
                max(near_best_margin, 0.08)
                if hybrid_probe_mode
                else max(near_best_margin, 0.05)
            )
            and float(requested_score_item.get("transfer_value_score", 0.0) or 0.0)
            >= (0.6 if hybrid_probe_mode else 0.75)
            and float(requested_score_item.get("hypothesis_value_score", 0.0) or 0.0)
            >= (0.6 if hybrid_probe_mode else 0.75)
            and str(requested_score_item.get("local_overfit_risk", "medium")) != "high"
        ):
            candidates_for_choice = [requested_score_item, *candidates_for_choice]
            requested_commit_window_applied = True
        if prompt_style in {
            "challenger_with_incumbent",
            "resuggest_probe_topk",
            "shape_probe_topk",
            "candidate_direction_review",
        }:
            if corridor_probe_mode:
                best_candidate = max(
                    candidates_for_choice,
                    key=lambda item: (
                        int(int(item["candidate_index"]) in preferred_candidate_set),
                        risk_rank.get(str(item["local_overfit_risk"]), 0),
                        -max(0, _changed_condition_count(int(item["candidate_index"])) - 1),
                        _anchor_match_count(int(item["candidate_index"])),
                        _same_scaffold_match_count(int(item["candidate_index"])),
                        _support_gap(int(item["candidate_index"])),
                        float(item.get("transfer_value_score", 0.0) or 0.0),
                        -shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                            "bo_rank",
                            9999,
                        ),
                        -shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                            "main_pool_rank",
                            9999,
                        ),
                        float(item.get("overall_score", 0.0) or 0.0),
                        -float(item.get("novelty_score", 0.0) or 0.0),
                        -_changed_scaffold_count(int(item["candidate_index"])),
                        -item["candidate_index"],
                    ),
                )
            else:
                best_candidate = max(
                    candidates_for_choice,
                    key=lambda item: (
                        item["overall_score"],
                        item["hypothesis_value_score"],
                        item["transfer_value_score"],
                        risk_rank.get(str(item["local_overfit_risk"]), 0),
                        preferred_bonus.get(int(item["candidate_index"]), 0),
                        item["candidate_index"] == int(output.selected_index),
                        -shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                            "bo_rank",
                            9999,
                        ),
                        -shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                            "recent_primary_dim_hits",
                            0,
                        ),
                        -shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                            "recent_secondary_dim_hits",
                            0,
                        ),
                        -shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                            "recent_scaffold_hits",
                            0,
                        ),
                        shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                            "pool_source",
                            "main_pool",
                        )
                        == "diversity_pool",
                        shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                            "shortlist_source",
                            "bo_top_ranked",
                        )
                        == "diversity_injected",
                        -item["candidate_index"],
                    ),
                )
            if hybrid_probe_mode:
                fallback_score_item = next(
                    (
                        item
                        for item in cleaned_scores
                        if int(item["candidate_index"]) == int(fallback_candidate_index)
                    ),
                    None,
                )
                if fallback_score_item is not None:
                    condition_variant_candidates = [
                        item
                        for item in candidates_for_choice
                        if (
                            int(item["candidate_index"]) != int(fallback_candidate_index)
                            and _history_condition_variant_delta(int(item["candidate_index"])) is not None
                            and item["overall_score"]
                            >= float(fallback_score_item.get("overall_score", 0.0) or 0.0) - 0.05
                            and float(item.get("transfer_value_score", 0.0) or 0.0) >= 0.65
                            and float(item.get("hypothesis_value_score", 0.0) or 0.0) >= 0.65
                            and str(item.get("local_overfit_risk", "medium")) != "high"
                        )
                    ]
                    if condition_variant_candidates and int(best_candidate["candidate_index"]) == int(
                        fallback_candidate_index
                    ):
                        best_candidate = max(
                            condition_variant_candidates,
                            key=lambda item: (
                                _history_condition_variant_delta(int(item["candidate_index"])) or float("-inf"),
                                -shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                                    "bo_rank",
                                    9999,
                                ),
                                item["overall_score"],
                                item["hypothesis_value_score"],
                                item["transfer_value_score"],
                            ),
                        )
                if corridor_probe_mode and int(best_candidate["candidate_index"]) != int(
                    fallback_candidate_index
                ):
                    if not _corridor_probe_has_real_support(int(best_candidate["candidate_index"])):
                        best_candidate = fallback_score_item or best_candidate
        else:
            best_candidate = max(
                candidates_for_choice,
                key=lambda item: (
                    item["candidate_index"] == int(output.selected_index),
                    preferred_bonus.get(int(item["candidate_index"]), 0),
                    item["overall_score"],
                    item["hypothesis_value_score"],
                    item["transfer_value_score"],
                    risk_rank.get(str(item["local_overfit_risk"]), 0),
                    -shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                        "recent_primary_dim_hits",
                        0,
                    ),
                    -shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                        "recent_secondary_dim_hits",
                        0,
                    ),
                    -shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                        "recent_scaffold_hits",
                        0,
                    ),
                    shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                        "pool_source",
                        "main_pool",
                    )
                    == "diversity_pool",
                    shortlist_meta_by_index.get(int(item["candidate_index"]), {}).get(
                        "shortlist_source",
                        "bo_top_ranked",
                    )
                    == "diversity_injected",
                    -item["candidate_index"],
                ),
            )
        if int(best_candidate["candidate_index"]) != int(output.selected_index):
            if int(output.selected_index) in admissible_candidate_set:
                if (
                    hybrid_probe_mode
                    and int(best_candidate["candidate_index"]) != int(fallback_candidate_index)
                ):
                    selection_status = (
                        "preferred_probe_candidate_selected_after_scoring"
                        if enforce_preferred_probe
                        else "controlled_candidate_selected_after_scoring"
                    )
                else:
                    selection_status = "requested_candidate_not_committed_after_scoring"
                    fallback_score_item = next(
                        (
                            item
                            for item in cleaned_scores
                            if int(item["candidate_index"]) == int(fallback_candidate_index)
                        ),
                        None,
                    )
                    if fallback_score_item is not None:
                        best_candidate = fallback_score_item
            else:
                selection_status = "fallback_candidate_not_committed_after_scoring"
        return {
            "selected_index": int(best_candidate["candidate_index"]),
            "requested_selected_index": int(requested_selected_index),
            "candidate_scores": cleaned_scores,
            "reasoning": output.reasoning,
            "prompt_style": prompt_style,
            "selection_status": selection_status,
            "admissible_candidate_indices": admissible_candidate_indices,
            "preferred_candidate_indices": preferred_candidate_indices,
            "effective_preferred_candidate_indices": effective_preferred_candidate_indices,
            "fallback_candidate_index": int(fallback_candidate_index),
            "state_router_guidance": state_router_guidance or None,
            "requested_score_gap": requested_score_gap,
            "requested_commit_window_applied": requested_commit_window_applied,
        }

    def semantic_assessment(
        self,
        candidate: dict[str, Any],
        decision_context: dict[str, Any],
        reaction_context: dict[str, Any],
    ) -> dict[str, Any]:
        node_view = self._node_state_view(decision_context, "semantic_assessment")
        knowledge_context, knowledge_meta = self._knowledge_for_prompt(node_view)
        skill_block = self._render_procedural_skill_block(
            "semantic_assessment",
            call_name="semantic_assessment",
        )
        prompt_text = build_semantic_assessment_prompt(
            candidate_condition=candidate,
            reaction_context=reaction_context,
            decision_context=node_view,
            knowledge_context=knowledge_context,
            knowledge_meta=knowledge_meta,
            value_annotations=self._value_annotations_for_payloads(
                [candidate, reaction_context, node_view],
                force_translate_missing=False,
            ),
            skill_block=skill_block,
        )
        fallback = SemanticAssessment()
        output = self._invoke_agent_structured(
            self.semantic_assessment_agent,
            prompt_text,
            fallback,
            call_name="semantic_assessment",
        )
        output.plausibility_score = max(0.0, min(1.0, float(output.plausibility_score)))
        output.novelty_score = max(0.0, min(1.0, float(output.novelty_score)))
        return output.model_dump()

    def candidate_verification(
        self,
        *,
        candidate: dict[str, Any],
        decision_context: dict[str, Any],
        reaction_context: dict[str, Any],
        semantic_assessment: dict[str, Any],
        controller_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        node_view = self._node_state_view(decision_context, "verification_pass")
        knowledge_context, knowledge_meta = self._knowledge_for_prompt(node_view)
        prompt_text = build_candidate_verification_prompt(
            candidate=candidate,
            reaction_context=reaction_context,
            decision_context=node_view,
            semantic_assessment=semantic_assessment,
            controller_plan=controller_plan or {},
            knowledge_context=knowledge_context,
            knowledge_meta=knowledge_meta,
            value_annotations=self._value_annotations_for_payloads(
                [candidate, reaction_context, node_view, semantic_assessment, controller_plan or {}],
                force_translate_missing=False,
            ),
        )
        output = self._invoke_agent_structured(
            self.verification_agent,
            prompt_text,
            VerificationPass(),
            call_name="candidate_verification",
        )
        return output.model_dump()

    def design_init_experiments(
        self,
        search_space_meta: dict[str, Any],
        dataset_meta: dict[str, Any],
        init_budget: int,
        sample_pool: list[dict[str, Any]],
        knowledge_context: list[dict[str, Any]] | None = None,
        knowledge_meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Use LLM chemical prior knowledge to design initial experiments.

        Returns InitDesignAction.model_dump() with `designed_experiments` list of
        length init_budget (or fewer if LLM returns fewer; caller should pad with
        random candidates as needed).
        """
        value_annotations = self._trim_init_value_annotations(
            self._value_annotations_for_payloads(
                [search_space_meta, dataset_meta, *sample_pool[: self.prompt_config.init.sample_pool_size]],
                force_translate_missing=True,
                max_entries=self.prompt_config.init.annotation_max_entries,
            ),
            dataset_meta,
        )
        skill_block = self._render_procedural_skill_block(
            "design_init_experiments",
            call_name="design_init_experiments",
        )
        prompt_text = build_init_design_prompt(
            search_space_meta=search_space_meta,
            dataset_meta=dataset_meta,
            init_budget=init_budget,
            sample_pool=sample_pool,
            knowledge_context=knowledge_context,
            knowledge_meta=knowledge_meta,
            value_annotations=value_annotations,
            skill_block=skill_block,
        )
        fallback = InitDesignAction()
        output = self._invoke_agent_structured(
            self.init_design_agent,
            prompt_text,
            fallback,
            call_name="design_init_experiments",
        )
        return output.model_dump()

    def generate_search_constraints(
        self,
        decision_context: dict[str, Any],
        search_space_meta: dict[str, Any],
        history_tail: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate include/exclude search constraints for the BO candidate pool.

        Returns LLMSearchConstraintAction.model_dump().  constraints=[] means
        LLM chose not to restrict the search space (full pool remains active).
        """
        node_view = self._node_state_view(decision_context, "search_constraints")
        value_annotations = self._value_annotations_for_payloads(
            [node_view, search_space_meta, *history_tail],
            force_translate_missing=True,
            max_entries=self.prompt_config.search_constraint.annotation_max_entries,
        )
        skill_block = self._render_procedural_skill_block(
            "generate_search_constraints",
            call_name="generate_search_constraints",
        )
        prompt_text = build_search_constraint_prompt(
            decision_context=node_view,
            search_space_meta=search_space_meta,
            history_tail=history_tail,
            value_annotations=value_annotations,
            skill_block=skill_block,
        )
        fallback = LLMSearchConstraintAction()
        output = self._invoke_agent_structured(
            self.search_constraint_agent,
            prompt_text,
            fallback,
            call_name="generate_search_constraints",
        )
        return output.model_dump()
