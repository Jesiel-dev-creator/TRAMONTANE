"""Tramontane Agent — the core unit of work.

Every agent wraps a Mistral model call with identity (role/goal/backstory),
budget control, GDPR awareness, and automatic model routing.

The ``run()`` method is the single execution entry-point:
resolve model → check budget → call Mistral → track cost → return AgentResult.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import inspect
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from tramontane.core.exceptions import AgentTimeoutError, BudgetExceededError
from tramontane.core.profiles import FleetProfile, apply_profile
from tramontane.router.models import MISTRAL_MODELS, MistralModel, get_model

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RunContext:
    """Shared context for a multi-agent pipeline run.

    Tracks cumulative cost and enforces a shared budget.
    Pass to agent.run(run_context=ctx) — spent_eur is updated
    automatically after each call.
    """

    budget_eur: float | None = None
    spent_eur: float = 0.0
    run_id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent_costs: dict[str, float] = dataclasses.field(default_factory=dict)
    reallocation: Literal["fixed", "adaptive"] = "fixed"
    """'fixed': each agent uses its own budget_eur.
    'adaptive': unspent budget from earlier agents flows to later agents."""
    _agent_budgets: dict[str, float] = dataclasses.field(default_factory=dict)
    """Original budgets per agent role for tracking savings."""

    @property
    def remaining_eur(self) -> float | None:
        """Remaining budget, or None if unlimited."""
        if self.budget_eur is None:
            return None
        return max(0.0, self.budget_eur - self.spent_eur)

    def record(self, agent_role: str, cost_eur: float) -> None:
        """Record cost from an agent execution."""
        self.spent_eur += cost_eur
        self.agent_costs[agent_role] = self.agent_costs.get(agent_role, 0.0) + cost_eur

    def get_effective_budget(self, agent_role: str, agent_budget: float | None) -> float | None:
        """Get effective budget for an agent, including reallocated savings."""
        if self.reallocation == "fixed" or agent_budget is None:
            return agent_budget

        self._agent_budgets[agent_role] = agent_budget
        total_savings = 0.0
        for role, original_budget in self._agent_budgets.items():
            if role == agent_role:
                continue
            actual_cost = self.agent_costs.get(role, 0.0)
            if actual_cost < original_budget:
                total_savings += (original_budget - actual_cost)

        effective = agent_budget + total_savings
        if total_savings > 0:
            logger.info(
                "Adaptive reallocation: %s gets €%.4f (€%.4f base + €%.4f savings)",
                agent_role,
                effective,
                agent_budget,
                total_savings,
            )
        return effective


class AgentResult(BaseModel):
    """Result returned by Agent.run()."""

    output: str
    model_used: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_eur: float = 0.0
    duration_seconds: float = 0.0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_used: bool = False
    parsed_output: Any | None = None
    """Validated Pydantic model if Agent.output_schema was set."""


class StreamEvent(BaseModel):
    """Event emitted during streaming agent execution.

    Yielded by Agent.run_stream() — one per token chunk, then
    a final 'complete' event carrying the full AgentResult.
    """

    type: Literal[
        "start", "token", "complete", "error",
        "pattern_match", "validation_retry",
        "reasoning_escalation", "cascade_escalation",
        "tool_call",
    ]
    token: str = ""
    model_used: str = ""
    result: AgentResult | None = None
    error: str = ""
    pattern_id: str = ""
    tool_name: str = ""
    tool_args: str = ""


# ---------------------------------------------------------------------------
# Tool calling helpers
# ---------------------------------------------------------------------------

_PY_TO_JSON_TYPE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
}


def _function_to_tool(func: Callable[..., Any]) -> dict[str, Any]:
    """Convert a Python function with type hints to Mistral tool spec."""
    sig = inspect.signature(func)
    properties: dict[str, dict[str, str]] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        annotation = param.annotation
        type_name = getattr(annotation, "__name__", str(annotation))
        json_type = _PY_TO_JSON_TYPE.get(type_name, "string")
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


async def _execute_tool(
    tool_call: Any,
    tools: list[Any],
) -> str:
    """Execute a tool call by matching function name and calling it."""
    func_name = tool_call.function.name
    try:
        raw_args = tool_call.function.arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except (json.JSONDecodeError, TypeError):
        args = {}

    for tool in tools:
        if callable(tool) and tool.__name__ == func_name:
            if asyncio.iscoroutinefunction(tool):
                result = await tool(**args)
            else:
                result = tool(**args)
            return str(result)

    return f"Error: tool '{func_name}' not found"


class Agent(BaseModel):
    """A single Tramontane agent backed by a Mistral model.

    Combines CrewAI-style identity (role/goal/backstory) with
    Tramontane-unique budget control, GDPR levels, and automatic
    model routing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── IDENTITY (required — CrewAI UX pattern) ──────────────────────
    role: str
    goal: str
    backstory: str

    # ── MODEL ROUTING (Tramontane-unique) ────────────────────────────
    model: str = "auto"
    function_calling_model: str = "auto"
    reasoning_model: str | None = None
    locale: str = "en"
    routing_hint: str | None = None
    """Hint for the router classifier (e.g. 'text-only JSON output').
    Passed as context when model='auto'."""

    # ── TOOLS ────────────────────────────────────────────────────────
    tools: list[Any] = Field(default_factory=list)
    tool_choice: str | None = None
    """Tool choice strategy: "auto" (default when tools set), "none",
    "any" (force at least one tool call), or "required"."""
    parallel_tool_calls: bool = True
    """Allow model to call multiple tools in one turn."""
    allow_code_execution: bool = False
    code_execution_mode: Literal["safe", "unsafe"] = "safe"

    # ── EXECUTION GUARDS (from CrewAI) ───────────────────────────────
    max_iter: int = 20
    max_rpm: int | None = None
    max_execution_time: int | None = None  # seconds
    max_tokens: int | None = None  # max output tokens, None = model default
    max_retry_limit: int = 3
    respect_context_window: bool = True

    # ── INTELLIGENCE FLAGS ───────────────────────────────────────────
    reasoning: bool = False
    max_reasoning_attempts: int | None = 3
    streaming: bool = True
    inject_date: bool = False
    allow_delegation: bool = False
    temperature: float | None = None
    """Sampling temperature (0.0-1.5). None = model default."""
    reasoning_effort: Literal["none", "medium", "high"] | None = None
    """Reasoning effort for models that support it (e.g. mistral-small-4).
    None = model default. Only passed if the resolved model supports it."""
    reasoning_strategy: Literal["fixed", "progressive"] = "fixed"
    """'fixed' uses the set reasoning_effort. 'progressive' starts at
    'none' and escalates through 'medium' then 'high' if validate_output
    fails. Only works with models that support reasoning_effort."""

    # ── MEMORY (from Agno) ───────────────────────────────────────────
    memory: bool = True
    add_history_to_context: bool = True
    learning: bool = False

    # ── KNOWLEDGE (RAG) ─────────────────────────────────────────────
    knowledge: Any | None = None
    """Optional KnowledgeBase for RAG. Retrieves relevant chunks and
    injects them into the system prompt before generation."""
    knowledge_top_k: int = 5
    """Number of top chunks to retrieve from knowledge base."""

    # ── COST CONTROL (Tramontane-unique) ─────────────────────────────
    budget_eur: float | None = None

    # ── GDPR (Tramontane-unique) ─────────────────────────────────────
    gdpr_level: Literal["none", "standard", "strict"] = "none"
    store_on_cloud: bool = True

    # ── OUTPUT VALIDATION + CASCADE ─────────────────────────────────
    validate_output: Callable[[AgentResult], bool] | None = None
    """Optional output validator. Returns True if acceptable.
    If False, agent retries up to max_validation_retries times."""
    max_validation_retries: int = 2
    """Max retries when validate_output returns False."""
    cascade: list[str | dict[str, Any]] | None = None
    """Model cascade chain. If validate_output fails on the primary model,
    try each subsequent model. Entries are alias strings or dicts with
    model + overrides (e.g. {"model": "devstral-2", "max_tokens": 32000}).
    Cascade combines with progressive reasoning — effort levels are tried
    per model before cascading to the next."""
    fleet_profile: FleetProfile | None = None
    """Optional fleet profile preset. Applies when model='auto'."""
    output_schema: type[BaseModel] | None = None
    """Pydantic model to validate agent output. When set, activates
    JSON mode and validates against the schema."""

    # ── OBSERVABILITY ────────────────────────────────────────────────
    audit_actions: bool = True
    verbose: bool = False
    step_callback: Callable[..., Any] | None = None

    # ── PRIVATE STATE ────────────────────────────────────────────────
    # (v0.1.2: removed _cost_tracker, _mistral_agent_id, _conversation_id,
    #  _run_count — all unused. Agent is stateless by design.)

    # ── COST METHODS ─────────────────────────────────────────────────
    # Agent is STATELESS on cost. It reports per-call cost in AgentResult.
    # Pipeline is the single source of truth for accumulated cost.

    @staticmethod
    def estimate_cost(
        input_tokens: int,
        output_tokens: int,
        model_alias: str,
    ) -> float:
        """Calculate EUR cost from real token counts and model pricing."""
        model_info = get_model(model_alias)
        return (
            (input_tokens / 1_000_000) * model_info.cost_per_1m_input_eur
            + (output_tokens / 1_000_000) * model_info.cost_per_1m_output_eur
        )

    def check_budget(
        self,
        estimated_cost: float,
        spent_eur: float = 0.0,
    ) -> None:
        """Raise BudgetExceededError if estimated cost far exceeds budget.

        Pre-call estimation is rough — only reject if the estimate is
        more than 2x over the remaining budget.  Actual cost is always
        checked post-call by Pipeline (the hard guard).

        Args:
            estimated_cost: The projected cost of the next call.
            spent_eur: Total already spent (passed by Pipeline).
        """
        if self.budget_eur is not None:
            budget_remaining = self.budget_eur - spent_eur
            # Only reject if estimate is more than 2x over remaining budget
            # (estimation is rough — don't block affordable calls)
            if estimated_cost > budget_remaining * 2.0:
                raise BudgetExceededError(
                    budget_eur=self.budget_eur,
                    spent_eur=spent_eur,
                    pipeline_name=self.role,
                )

    def _check_budget_with_override(
        self,
        estimated_cost: float,
        spent_eur: float,
        budget_override: float | None,
    ) -> None:
        """Check budget using an explicit override when provided."""
        if budget_override is None:
            self.check_budget(estimated_cost, spent_eur=spent_eur)
            return
        budget_remaining = budget_override - spent_eur
        if estimated_cost > budget_remaining * 2.0:
            raise BudgetExceededError(
                budget_eur=budget_override,
                spent_eur=spent_eur,
                pipeline_name=self.role,
            )

    # ── PROMPT BUILDING ──────────────────────────────────────────────

    def system_prompt(self) -> str:
        """Build the system prompt from identity fields.

        Prepends UTC datetime if inject_date is True.
        Appends chain-of-thought instruction if reasoning is True.
        """
        parts: list[str] = []

        if self.inject_date:
            now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            parts.append(f"Current date and time: {now}")

        parts.append(f"Role: {self.role}")
        parts.append(f"Goal: {self.goal}")
        parts.append(f"Backstory: {self.backstory}")

        if self.reasoning:
            parts.append(
                "Think step by step. Show your reasoning before giving a final answer."
            )

        return "\n".join(parts)

    # ── MISTRAL API MAPPING ──────────────────────────────────────────

    def to_mistral_params(self) -> dict[str, Any]:
        """Return a dict ready for mistralai client agent creation.

        Maps Tramontane agent fields to the Mistral Agents API parameters.
        """
        params: dict[str, Any] = {
            "model": self.model if self.model != "auto" else None,
            "name": self.role,
            "instructions": self.system_prompt(),
        }

        if self.tools:
            params["tools"] = self.tools

        if self.max_execution_time is not None:
            params["timeout"] = self.max_execution_time

        return params

    # ── EXECUTION ────────────────────────────────────────────────────

    async def run(
        self,
        input_text: str,
        *,
        router: Any | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        run_id: str | None = None,
        spent_eur: float = 0.0,
        run_context: RunContext | None = None,
        context: str | None = None,
    ) -> AgentResult:
        """Execute this agent: resolve model, call Mistral, return result.

        Agent is stateless on cost. It calculates per-call cost from real
        API token counts and returns it in AgentResult. Pipeline is the
        single source of truth for accumulated cost.

        Args:
            input_text: The user/handoff message to process.
            router: Optional MistralRouter for model="auto" resolution.
            conversation_history: Prior messages to include as context.
            run_id: Trace identifier (generated if not provided).
            spent_eur: Total already spent by Pipeline (for budget check).
            run_context: Shared RunContext for multi-agent cost tracking.
            context: Dynamic per-call context appended to system prompt.

        Returns:
            AgentResult with output, model, tokens, cost, duration.

        Raises:
            BudgetExceededError: If estimated cost exceeds remaining budget.
            AgentTimeoutError: If max_execution_time is exceeded.
        """
        _spent = run_context.spent_eur if run_context else spent_eur
        profile_task_type = self.routing_hint if self.routing_hint else None
        profile_model_override: str | None = None
        profile_effort_override: str | None = None
        if self.fleet_profile and self.model == "auto":
            profile_model, profile_effort = apply_profile(
                self.fleet_profile,
                self.model,
                profile_task_type,
            )
            if profile_model != "auto":
                profile_model_override = profile_model
            if profile_effort is not None and self.reasoning_effort is None:
                profile_effort_override = profile_effort

        effective_budget = (
            run_context.get_effective_budget(self.role, self.budget_eur)
            if run_context and run_context.reallocation == "adaptive"
            else self.budget_eur
        )

        async def _try_model(
            model_alias: str | None = None,
            max_tok: int | None = None,
        ) -> tuple[AgentResult | None, bool]:
            """Try a single model with progressive/fixed strategy.

            Returns (result, accepted).
            """
            effective_model = model_alias or self.model
            supports_effort = False
            if self.reasoning_strategy == "progressive" and effective_model != "auto":
                info = MISTRAL_MODELS.get(effective_model)
                supports_effort = bool(info and info.supports_reasoning_effort)

            res: AgentResult | None = None
            if supports_effort and self.reasoning_strategy == "progressive":
                for effort in ("none", "medium", "high"):
                    res = await self._run_once(
                        input_text,
                        router=router,
                        conversation_history=conversation_history,
                        run_id=run_id,
                        spent_eur=_spent,
                        context=context,
                        effort_override=effort,
                        model_override=model_alias,
                        max_tokens_override=max_tok,
                        budget_override=effective_budget,
                        profile_effort_override=profile_effort_override,
                    )
                    if self.validate_output is None or self.validate_output(res):
                        return res, True
                    logger.warning(
                        "Progressive effort='%s' failed for %s",
                        effort, self.role,
                    )
                return res, False
            # Fixed strategy
            max_attempts = (
                (self.max_validation_retries + 1) if self.validate_output else 1
            )
            for attempt in range(max_attempts):
                res = await self._run_once(
                    input_text,
                    router=router,
                    conversation_history=conversation_history,
                    run_id=run_id,
                    spent_eur=_spent,
                    context=context,
                    model_override=model_alias,
                    max_tokens_override=max_tok,
                    budget_override=effective_budget,
                    profile_effort_override=profile_effort_override,
                )
                if self.validate_output is None or self.validate_output(res):
                    return res, True
                if attempt < self.max_validation_retries:
                    logger.warning(
                        "Validation failed for %s (attempt %d/%d)",
                        self.role, attempt + 1, self.max_validation_retries,
                    )
            return res, False

        # Try primary model
        result, accepted = await _try_model(profile_model_override)

        # Cascade: try subsequent models if validation failed
        if not accepted and self.cascade and self.validate_output:
            for i, entry in enumerate(self.cascade):
                if isinstance(entry, str):
                    c_model, c_max_tok = entry, None
                else:
                    c_model = str(entry["model"])
                    c_max_tok = entry.get("max_tokens")
                logger.info("Cascade level %d: trying %s", i, c_model)
                result, accepted = await _try_model(c_model, c_max_tok)
                if accepted:
                    logger.info("Cascade succeeded at level %d: %s", i, c_model)
                    break

        assert result is not None  # noqa: S101
        if run_context:
            run_context.record(self.role, result.cost_eur)
        return result

    async def _run_once(
        self,
        input_text: str,
        *,
        router: Any | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        run_id: str | None = None,
        spent_eur: float = 0.0,
        context: str | None = None,
        effort_override: str | None = None,
        model_override: str | None = None,
        max_tokens_override: int | None = None,
        budget_override: float | None = None,
        profile_effort_override: str | None = None,
    ) -> AgentResult:
        """Execute a single Mistral API call (no validation retry loop)."""
        import anyio
        from mistralai.client import Mistral

        # -- Input validation --
        if not input_text or not input_text.strip():
            msg = f"Agent '{self.role}': input_text must be a non-empty string"
            raise ValueError(msg)
        budget_for_check = budget_override if budget_override is not None else self.budget_eur
        if budget_for_check is not None and budget_for_check < 0:
            msg = f"Agent '{self.role}': budget_eur must be >= 0, got {budget_for_check}"
            raise ValueError(msg)

        rid = run_id or uuid.uuid4().hex[:12]

        # 1. Resolve model (model_override from cascade takes priority)
        model_alias = model_override or self.model
        routing_decision = None
        if model_alias == "auto" and router is not None:
            routing_decision = await router.route(
                prompt=input_text,
                agent_budget_eur=budget_for_check,
                locale=self.locale,
                context=self.routing_hint,
            )
            model_alias = routing_decision.primary_model

        model_info = MISTRAL_MODELS.get(model_alias)
        api_model = model_info.api_id if model_info else model_alias

        # 2. Build messages (with optional RAG context)
        sys_prompt = self.system_prompt()
        if self.knowledge is not None:
            retrieval = await self.knowledge.retrieve(
                input_text, top_k=self.knowledge_top_k,
            )
            kb_context = self.knowledge.format_context(retrieval)
            if kb_context:
                sys_prompt += f"\n\n{kb_context}"
                logger.info(
                    "Retrieved %d chunks from knowledge base",
                    len(retrieval.chunks),
                )
        if context:
            sys_prompt += f"\n\n## Additional Context\n{context}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": sys_prompt},
        ]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": input_text})

        # 3. Pre-call budget check with improved estimation
        if model_info:
            est_cost = self._estimate_call_cost(messages, model_info)
            self._check_budget_with_override(
                est_cost,
                spent_eur=spent_eur,
                budget_override=budget_for_check,
            )

        # 4. Call Mistral with retry + exponential backoff
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            msg = "MISTRAL_API_KEY environment variable is not set"
            raise RuntimeError(msg)

        client = Mistral(api_key=api_key)
        start_time = time.monotonic()
        max_retries = self.max_retry_limit
        response: Any = None

        # Resolve effective max_tokens: override > explicit > model default
        effective_max_tokens = max_tokens_override or self.max_tokens
        if effective_max_tokens is None and model_info:
            effective_max_tokens = model_info.max_output_tokens

        chat_kwargs: dict[str, Any] = {
            "model": api_model,
            "messages": messages,
        }
        if effective_max_tokens is not None:
            chat_kwargs["max_tokens"] = effective_max_tokens
        if self.temperature is not None:
            chat_kwargs["temperature"] = self.temperature

        # Structured output (JSON mode)
        if self.output_schema is not None:
            chat_kwargs["response_format"] = {"type": "json_object"}

        # Tools
        callable_tools = [t for t in self.tools if callable(t)]
        if self.tools:
            chat_kwargs["tools"] = [
                _function_to_tool(t) if callable(t) else t for t in self.tools
            ]
            chat_kwargs["tool_choice"] = self.tool_choice or "auto"
            chat_kwargs["parallel_tool_calls"] = self.parallel_tool_calls
            # Mistral best practice: low temperature for consistent tool calls
            if self.temperature is None:
                chat_kwargs["temperature"] = 0.1

        # Reasoning effort: use override (progressive) or agent setting
        effective_effort = effort_override or self.reasoning_effort or profile_effort_override
        if effective_effort is not None and model_info:
            if model_info.supports_reasoning_effort:
                chat_kwargs["reasoning_effort"] = effective_effort
            else:
                logger.debug(
                    "Model %s doesn't support reasoning_effort, ignoring",
                    model_alias,
                )

        all_tool_calls: list[dict[str, Any]] = []

        for attempt in range(max_retries + 1):
            try:
                coro = client.chat.complete_async(**chat_kwargs)
                if self.max_execution_time:
                    response = await asyncio.wait_for(
                        coro, timeout=float(self.max_execution_time),
                    )
                else:
                    response = await coro
                break  # success

            except asyncio.TimeoutError:
                raise AgentTimeoutError(
                    agent_role=self.role,
                    timeout_seconds=self.max_execution_time or 0,
                )
            except Exception as exc:
                if attempt >= max_retries:
                    logger.error(
                        "[%s] Mistral API failed after %d attempts: %s",
                        rid, attempt + 1, exc,
                    )
                    raise
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "[%s] Mistral API error (attempt %d/%d): %s — retrying in %ds",
                    rid, attempt + 1, max_retries + 1, exc, wait,
                )
                await anyio.sleep(wait)

        # 5. Tool call loop — execute tools until final text response
        if callable_tools:
            tool_iter = 0
            while tool_iter < self.max_iter:
                msg = response.choices[0].message
                if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                    break
                # Append assistant message with tool calls to conversation
                chat_kwargs["messages"].append(msg)
                for tc in msg.tool_calls:
                    tc_result = await _execute_tool(tc, callable_tools)
                    raw_args = tc.function.arguments
                    all_tool_calls.append({
                        "name": tc.function.name,
                        "args": (
                            json.loads(raw_args) if isinstance(raw_args, str)
                            else dict(raw_args)
                        ),
                        "result": tc_result,
                    })
                    chat_kwargs["messages"].append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tc_result,
                    })
                    tool_iter += 1
                # Call again with tool results
                response = await client.chat.complete_async(**chat_kwargs)

        duration = time.monotonic() - start_time

        # 6. Extract output
        choice = response.choices[0]
        output_text = str(choice.message.content or "")
        tool_calls = all_tool_calls

        # 6. Actual cost from real API token counts
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage") and response.usage:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        actual_cost = 0.0
        if model_info:
            actual_cost = (
                (input_tokens / 1_000_000) * model_info.cost_per_1m_input_eur
                + (output_tokens / 1_000_000) * model_info.cost_per_1m_output_eur
            )

        logger.debug(
            "[%s] agent=%s model=%s tokens=%d/%d cost=EUR %.6f dur=%.2fs",
            rid, self.role, model_alias, input_tokens, output_tokens,
            actual_cost, duration,
        )

        # 8. Parse structured output if schema is set
        parsed_output: Any | None = None
        if self.output_schema is not None:
            try:
                parsed_output = self.output_schema.model_validate_json(output_text)
            except Exception as exc:
                logger.warning("Output schema validation failed: %s", exc)

        return AgentResult(
            output=output_text,
            model_used=model_alias,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_eur=actual_cost,
            duration_seconds=round(duration, 3),
            tool_calls=tool_calls,
            reasoning_used=self.reasoning,
            parsed_output=parsed_output,
        )

    async def run_stream(
        self,
        input_text: str,
        *,
        router: Any | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        run_id: str | None = None,
        spent_eur: float = 0.0,
        run_context: RunContext | None = None,
        context: str | None = None,
        on_pattern: dict[str, Callable[..., Any]] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Execute this agent with token-by-token streaming.

        Yields StreamEvent objects as tokens are generated.
        The final event has type="complete" with a full AgentResult.
        Errors are yielded as type="error" events (never raised).

        Args:
            input_text: The user/handoff message to process.
            router: Optional MistralRouter for model="auto" resolution.
            conversation_history: Prior messages to include as context.
            run_id: Trace identifier (generated if not provided).
            spent_eur: Total already spent by Pipeline (for budget check).
            run_context: Shared RunContext for multi-agent cost tracking.
            context: Dynamic per-call context appended to system prompt.
            on_pattern: Dict of {regex: callback(match, text)}. Fires when
                pattern matches accumulated output mid-stream.

        Yields:
            StreamEvent with type in start/token/complete/error/pattern_match/
            validation_retry.
        """
        import anyio
        from mistralai.client import Mistral

        # -- Input validation --
        if not input_text or not input_text.strip():
            yield StreamEvent(
                type="error",
                error=f"Agent '{self.role}': input_text must be a non-empty string",
            )
            return
        effective_budget = (
            run_context.get_effective_budget(self.role, self.budget_eur)
            if run_context and run_context.reallocation == "adaptive"
            else self.budget_eur
        )
        if effective_budget is not None and effective_budget < 0:
            yield StreamEvent(
                type="error",
                error=f"Agent '{self.role}': budget_eur must be >= 0, got {effective_budget}",
            )
            return

        rid = run_id or uuid.uuid4().hex[:12]
        effective_spent = run_context.spent_eur if run_context else spent_eur

        profile_task_type = self.routing_hint if self.routing_hint else None
        profile_model_override: str | None = None
        profile_effort_override: str | None = None
        if self.fleet_profile and self.model == "auto":
            profile_model, profile_effort = apply_profile(
                self.fleet_profile,
                self.model,
                profile_task_type,
            )
            if profile_model != "auto":
                profile_model_override = profile_model
            if profile_effort is not None and self.reasoning_effort is None:
                profile_effort_override = profile_effort

        # 1. Resolve model
        model_alias = profile_model_override or self.model
        if model_alias == "auto" and router is not None:
            try:
                routing_decision = await router.route(
                    prompt=input_text,
                    agent_budget_eur=effective_budget,
                    locale=self.locale,
                    context=self.routing_hint,
                )
                model_alias = routing_decision.primary_model
            except Exception as exc:
                yield StreamEvent(type="error", error=str(exc))
                return

        model_info = MISTRAL_MODELS.get(model_alias)
        api_model = model_info.api_id if model_info else model_alias

        # 2. Build messages (with optional RAG context)
        sys_prompt = self.system_prompt()
        if self.knowledge is not None:
            retrieval = await self.knowledge.retrieve(
                input_text, top_k=self.knowledge_top_k,
            )
            kb_context = self.knowledge.format_context(retrieval)
            if kb_context:
                sys_prompt += f"\n\n{kb_context}"
        if context:
            sys_prompt += f"\n\n## Additional Context\n{context}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": sys_prompt},
        ]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": input_text})

        # 3. Pre-call budget check
        if model_info:
            est_cost = self._estimate_call_cost(messages, model_info)
            try:
                self._check_budget_with_override(
                    est_cost,
                    spent_eur=effective_spent,
                    budget_override=effective_budget,
                )
            except BudgetExceededError as exc:
                yield StreamEvent(type="error", error=str(exc))
                return

        # 4. API key check
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            yield StreamEvent(
                type="error",
                error="MISTRAL_API_KEY environment variable is not set",
            )
            return

        # 5. Compile patterns once
        compiled_patterns: list[tuple[re.Pattern[str], Callable[..., Any]]] = []
        if on_pattern:
            compiled_patterns = [
                (re.compile(pattern), callback)
                for pattern, callback in on_pattern.items()
            ]

        # 6. Build model cascade list: primary model first, then cascade entries
        _CascadeEntry = tuple[str, str, int | None]  # (alias, api_id, max_tok)
        cascade_models: list[_CascadeEntry] = [(model_alias, api_model, None)]
        if self.cascade and self.validate_output:
            for entry in self.cascade:
                if isinstance(entry, str):
                    c_info = MISTRAL_MODELS.get(entry)
                    c_api = c_info.api_id if c_info else entry
                    cascade_models.append((entry, c_api, None))
                else:
                    c_alias = str(entry["model"])
                    c_info = MISTRAL_MODELS.get(c_alias)
                    c_api = c_info.api_id if c_info else c_alias
                    cascade_models.append(
                        (c_alias, c_api, entry.get("max_tokens")),
                    )

        result: AgentResult | None = None
        accepted = False

        for cascade_idx, (cur_alias, cur_api, cur_max_tok) in enumerate(
            cascade_models,
        ):
            if cascade_idx > 0:
                yield StreamEvent(
                    type="cascade_escalation", model_used=cur_alias,
                )

            cur_info = MISTRAL_MODELS.get(cur_alias)

            # Resolve effective max_tokens for this model
            eff_max_tokens = cur_max_tok or self.max_tokens
            if eff_max_tokens is None and cur_info:
                eff_max_tokens = cur_info.max_output_tokens

            stream_kwargs: dict[str, Any] = {
                "model": cur_api,
                "messages": messages,
            }
            if eff_max_tokens is not None:
                stream_kwargs["max_tokens"] = eff_max_tokens
            if self.temperature is not None:
                stream_kwargs["temperature"] = self.temperature

            # Reasoning effort support for this cascade model
            supports_effort = bool(
                cur_info and cur_info.supports_reasoning_effort
            )
            if (
                (self.reasoning_effort is not None or profile_effort_override is not None)
                and supports_effort
                and self.reasoning_strategy == "fixed"
            ):
                stream_kwargs["reasoning_effort"] = (
                    self.reasoning_effort or profile_effort_override
                )

            # Build iteration plan: progressive reasoning or fixed validation
            use_progressive = (
                self.reasoning_strategy == "progressive"
                and supports_effort
                and self.validate_output is not None
            )
            if use_progressive:
                iterations: list[tuple[str | None, bool]] = [
                    (e, i > 0) for i, e in enumerate(["none", "medium", "high"])
                ]
            elif self.validate_output:
                iterations = [(None, False)] * (self.max_validation_retries + 1)
            else:
                iterations = [(None, False)]

            for iter_effort, is_escalation in iterations:
                if is_escalation:
                    yield StreamEvent(
                        type="reasoning_escalation", model_used=cur_alias,
                    )
                elif result is not None and cascade_idx == 0:
                    yield StreamEvent(
                        type="validation_retry", model_used=cur_alias,
                    )

                if iter_effort is not None:
                    stream_kwargs["reasoning_effort"] = iter_effort

                # -- Stream one attempt for this effort/model combo --
                yield StreamEvent(type="start", model_used=cur_alias)

                client = Mistral(api_key=api_key)
                start_time = time.monotonic()
                full_output = ""
                input_tokens = 0
                output_tokens = 0
                tokens_yielded = False
                last_checked: dict[str, int] = {}

                for attempt in range(self.max_retry_limit + 1):
                    try:
                        stream = await client.chat.stream_async(
                            **stream_kwargs,
                        )
                        async with stream as event_stream:
                            async for event in event_stream:
                                if self.max_execution_time:
                                    elapsed = time.monotonic() - start_time
                                    if elapsed > self.max_execution_time:
                                        yield StreamEvent(
                                            type="error",
                                            error=(
                                                f"Agent '{self.role}' timed out"
                                                f" after"
                                                f" {self.max_execution_time}s"
                                            ),
                                        )
                                        return
                                chunk = event.data
                                if chunk.choices:
                                    delta = chunk.choices[0].delta
                                    token_text = (
                                        str(delta.content)
                                        if delta.content
                                        else ""
                                    )
                                    if token_text:
                                        full_output += token_text
                                        tokens_yielded = True
                                        yield StreamEvent(
                                            type="token",
                                            token=token_text,
                                            model_used=cur_alias,
                                        )
                                        for pat, cb in compiled_patterns:
                                            pkey = pat.pattern
                                            sp = last_checked.get(pkey, 0)
                                            sf = max(0, sp - 100)
                                            for m in pat.finditer(
                                                full_output, sf,
                                            ):
                                                if m.start() >= sp:
                                                    cr = cb(m, full_output)
                                                    if asyncio.iscoroutine(cr):
                                                        await cr
                                                    yield StreamEvent(
                                                        type="pattern_match",
                                                        model_used=cur_alias,
                                                        pattern_id=pkey,
                                                    )
                                            last_checked[pkey] = len(
                                                full_output,
                                            )
                                if chunk.usage:
                                    input_tokens = (
                                        chunk.usage.prompt_tokens or 0
                                    )
                                    output_tokens = (
                                        chunk.usage.completion_tokens or 0
                                    )
                        break  # success
                    except Exception as exc:
                        if tokens_yielded or attempt >= self.max_retry_limit:
                            yield StreamEvent(type="error", error=str(exc))
                            return
                        wait = min(2 ** attempt, 30)
                        logger.warning(
                            "[%s] Stream error (attempt %d/%d): %s — retry %ds",
                            rid, attempt + 1, self.max_retry_limit + 1,
                            exc, wait,
                        )
                        await anyio.sleep(wait)

                duration = time.monotonic() - start_time
                actual_cost = 0.0
                if cur_info:
                    actual_cost = (
                        (input_tokens / 1e6) * cur_info.cost_per_1m_input_eur
                        + (output_tokens / 1e6) * cur_info.cost_per_1m_output_eur
                    )
                logger.debug(
                    "[%s] stream agent=%s model=%s tokens=%d/%d cost=EUR %.6f",
                    rid, self.role, cur_alias, input_tokens, output_tokens,
                    actual_cost,
                )
                result = AgentResult(
                    output=full_output,
                    model_used=cur_alias,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_eur=actual_cost,
                    duration_seconds=round(duration, 3),
                    tool_calls=[],
                    reasoning_used=self.reasoning,
                )
                if self.validate_output is None or self.validate_output(result):
                    accepted = True
                    break
                logger.warning(
                    "Stream validation failed for %s (model=%s, effort=%s)",
                    self.role, cur_alias, iter_effort or "fixed",
                )
            # Break outer cascade loop if accepted
            if accepted:
                break

        if not accepted and result is not None:
            logger.warning(
                "All stream attempts failed validation for %s, "
                "accepting last result",
                self.role,
            )

        if run_context and result is not None:
            run_context.record(self.role, result.cost_eur)

        if result is not None:
            yield StreamEvent(
                type="complete",
                model_used=result.model_used,
                result=result,
            )

    def _estimate_call_cost(
        self,
        messages: list[dict[str, str]],
        model_info: MistralModel,
    ) -> float:
        """Estimate cost BEFORE making the API call.

        Uses character-based token estimation (~3.5 chars/token) plus
        conservative output multiplier.  This is a soft guard — the
        real budget enforcement happens post-call with actual token
        counts accumulated by Pipeline.
        """
        # Input tokens: ~3.5 chars per token for mixed en/fr
        total_chars = sum(len(m["content"]) for m in messages)
        est_input_tokens = total_chars / 3.5

        # Output estimate: 1.2x input for reasoning, 0.8x otherwise
        # (most outputs are shorter than inputs)
        output_multiplier = 1.2 if self.reasoning else 0.8
        est_output_tokens = est_input_tokens * output_multiplier

        input_cost = (est_input_tokens / 1_000_000) * model_info.cost_per_1m_input_eur
        output_cost = (est_output_tokens / 1_000_000) * model_info.cost_per_1m_output_eur

        # Reasoning adds modest overhead (1.1x), not 1.4x
        reasoning_overhead = 1.1 if self.reasoning else 1.0
        return (input_cost + output_cost) * reasoning_overhead

    # ── YAML LOADING ─────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> Agent:
        """Load an agent definition from a YAML file.

        The YAML file should contain a mapping of Agent field names to values.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
