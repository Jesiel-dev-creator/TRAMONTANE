"""Tramontane Agent — the core unit of work.

Every agent wraps a Mistral model call with identity (role/goal/backstory),
budget control, GDPR awareness, and automatic model routing.

The ``run()`` method is the single execution entry-point:
resolve model → check budget → call Mistral → track cost → return AgentResult.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from tramontane.core.exceptions import AgentTimeoutError, BudgetExceededError
from tramontane.router.models import MISTRAL_MODELS, MistralModel, get_model

logger = logging.getLogger(__name__)


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

    # ── TOOLS ────────────────────────────────────────────────────────
    tools: list[Any] = Field(default_factory=list)
    allow_code_execution: bool = False
    code_execution_mode: Literal["safe", "unsafe"] = "safe"

    # ── EXECUTION GUARDS (from CrewAI) ───────────────────────────────
    max_iter: int = 20
    max_rpm: int | None = None
    max_execution_time: int | None = None  # seconds
    max_retry_limit: int = 3
    respect_context_window: bool = True

    # ── INTELLIGENCE FLAGS ───────────────────────────────────────────
    reasoning: bool = False
    max_reasoning_attempts: int | None = 3
    streaming: bool = True
    inject_date: bool = False
    allow_delegation: bool = False

    # ── MEMORY (from Agno) ───────────────────────────────────────────
    memory: bool = True
    add_history_to_context: bool = True
    learning: bool = False

    # ── COST CONTROL (Tramontane-unique) ─────────────────────────────
    budget_eur: float | None = None

    # ── GDPR (Tramontane-unique) ─────────────────────────────────────
    gdpr_level: Literal["none", "standard", "strict"] = "none"
    store_on_cloud: bool = True

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
        """Raise BudgetExceededError if estimated_cost would exceed budget.

        Args:
            estimated_cost: The projected cost of the next call.
            spent_eur: Total already spent (passed by Pipeline).
        """
        if self.budget_eur is not None:
            if spent_eur + estimated_cost > self.budget_eur:
                raise BudgetExceededError(
                    budget_eur=self.budget_eur,
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

        Returns:
            AgentResult with output, model, tokens, cost, duration.

        Raises:
            BudgetExceededError: If estimated cost exceeds remaining budget.
            AgentTimeoutError: If max_execution_time is exceeded.
        """
        import anyio
        from mistralai.client import Mistral

        # -- Input validation --
        if not input_text or not input_text.strip():
            msg = f"Agent '{self.role}': input_text must be a non-empty string"
            raise ValueError(msg)
        if self.budget_eur is not None and self.budget_eur < 0:
            msg = f"Agent '{self.role}': budget_eur must be >= 0, got {self.budget_eur}"
            raise ValueError(msg)

        rid = run_id or uuid.uuid4().hex[:12]

        # 1. Resolve model
        model_alias = self.model
        routing_decision = None
        if model_alias == "auto" and router is not None:
            routing_decision = await router.route(
                prompt=input_text,
                agent_budget_eur=self.budget_eur,
                locale=self.locale,
            )
            model_alias = routing_decision.primary_model

        model_info = MISTRAL_MODELS.get(model_alias)
        api_model = model_info.api_id if model_info else model_alias

        # 2. Build messages
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt()},
        ]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": input_text})

        # 3. Pre-call budget check with improved estimation
        if model_info:
            est_cost = self._estimate_call_cost(messages, model_info)
            self.check_budget(est_cost, spent_eur=spent_eur)

        # 4. Call Mistral with retry + exponential backoff
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            msg = "MISTRAL_API_KEY environment variable is not set"
            raise RuntimeError(msg)

        client = Mistral(api_key=api_key)
        start_time = time.monotonic()
        max_retries = self.max_retry_limit
        response: Any = None

        for attempt in range(max_retries + 1):
            try:
                coro = client.chat.complete_async(
                    model=api_model,
                    messages=messages,  # type: ignore[arg-type]
                )
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

        duration = time.monotonic() - start_time

        # 5. Extract output
        choice = response.choices[0]
        output_text = str(choice.message.content or "")
        tool_calls: list[dict[str, Any]] = []
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            tool_calls = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in choice.message.tool_calls
            ]

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

        return AgentResult(
            output=output_text,
            model_used=model_alias,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_eur=actual_cost,
            duration_seconds=round(duration, 3),
            tool_calls=tool_calls,
            reasoning_used=self.reasoning,
        )

    def _estimate_call_cost(
        self,
        messages: list[dict[str, str]],
        model_info: MistralModel,
    ) -> float:
        """Estimate cost BEFORE making the API call.

        Uses character-based token estimation (~3.5 chars/token) plus
        output multiplier based on task type.
        """
        # Input tokens: ~3.5 chars per token for mixed en/fr
        total_chars = sum(len(m["content"]) for m in messages)
        est_input_tokens = total_chars / 3.5

        # Output estimate: 2x input for reasoning, 1.5x otherwise
        output_multiplier = 2.0 if self.reasoning else 1.5
        est_output_tokens = est_input_tokens * output_multiplier

        input_cost = (est_input_tokens / 1_000_000) * model_info.cost_per_1m_input_eur
        output_cost = (est_output_tokens / 1_000_000) * model_info.cost_per_1m_output_eur

        # Reasoning models use more tokens (1.4x overhead)
        reasoning_overhead = 1.4 if self.reasoning else 1.0
        return (input_cost + output_cost) * reasoning_overhead

    # ── YAML LOADING ─────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> Agent:
        """Load an agent definition from a YAML file.

        The YAML file should contain a mapping of Agent field names to values.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
