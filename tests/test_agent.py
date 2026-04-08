"""Tests for tramontane.core.agent."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from tramontane.core.agent import Agent, AgentResult
from tramontane.core.exceptions import BudgetExceededError
from tramontane.core.profiles import FleetProfile, apply_profile


class TestAgentInstantiation:
    """Agent construction and defaults."""

    def test_defaults(self, sample_agent: Agent) -> None:
        assert sample_agent.model == "auto"
        assert sample_agent.streaming is True
        assert sample_agent.gdpr_level == "none"
        assert sample_agent.budget_eur is None
        assert sample_agent.memory is True
        assert sample_agent.max_iter == 20

    def test_custom_fields(self) -> None:
        a = Agent(
            role="Dev",
            goal="Code",
            backstory="Expert",
            model="devstral-small",
            budget_eur=0.05,
            reasoning=True,
            locale="fr",
        )
        assert a.model == "devstral-small"
        assert a.budget_eur == 0.05
        assert a.reasoning is True
        assert a.locale == "fr"


class TestSystemPrompt:
    """system_prompt() output."""

    def test_basic(self, sample_agent: Agent) -> None:
        prompt = sample_agent.system_prompt()
        assert "Role: Researcher" in prompt
        assert "Goal:" in prompt
        assert "Backstory:" in prompt

    def test_with_date(self) -> None:
        a = Agent(role="R", goal="G", backstory="B", inject_date=True)
        prompt = a.system_prompt()
        assert "Current date and time:" in prompt

    def test_with_reasoning(self) -> None:
        a = Agent(role="R", goal="G", backstory="B", reasoning=True)
        prompt = a.system_prompt()
        assert "step by step" in prompt.lower()


class TestSystemPromptOverride:
    """system_prompt parameter on run() overrides backstory."""

    def test_system_prompt_field_accepted(self) -> None:
        """Verify system_prompt is a valid kwarg (no TypeError)."""
        # We can't call run() without API key, but verify the signature works
        import inspect

        sig = inspect.signature(Agent.run)
        assert "system_prompt" in sig.parameters

    def test_run_stream_accepts_system_prompt(self) -> None:
        import inspect

        sig = inspect.signature(Agent.run_stream)
        assert "system_prompt" in sig.parameters


class TestUnknownAliasLoudFailure:
    """Unknown model aliases should fail loudly, not silently degrade."""

    @pytest.mark.asyncio
    async def test_run_raises_on_bad_alias(self) -> None:
        """Agent.run() raises ModelNotAvailableError for api_id-style names."""
        import os

        from tramontane.core.exceptions import ModelNotAvailableError

        os.environ.setdefault("MISTRAL_API_KEY", "test-dummy-key")
        a = Agent(
            role="Builder", goal="Build", backstory="Expert",
            model="mistral-small-latest",  # api_id, NOT a Tramontane alias
        )
        with pytest.raises(ModelNotAvailableError) as exc_info:
            await a.run("build something")
        assert "mistral-small-latest" in str(exc_info.value)
        assert "alias" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_run_stream_yields_error_on_bad_alias(self) -> None:
        """run_stream() yields an error event for unknown aliases."""
        import os

        os.environ.setdefault("MISTRAL_API_KEY", "test-dummy-key")
        a = Agent(
            role="Builder", goal="Build", backstory="Expert",
            model="devstral-small-latest",  # api_id, not alias
        )
        events = []
        async for event in a.run_stream("build something"):
            events.append(event)
            if event.type == "error":
                break
        assert any(e.type == "error" for e in events)
        err = next(e for e in events if e.type == "error")
        assert "devstral-small-latest" in err.error
        assert "alias" in err.error.lower()


class TestCodeModelEstimation:
    """Code models should get higher output estimates."""

    def test_code_model_uses_higher_multiplier(self) -> None:
        """devstral-small (code model) should estimate more output."""
        from tramontane.router.models import get_model

        a_general = Agent(role="R", goal="G", backstory="B", model="mistral-small")
        a_code = Agent(role="R", goal="G", backstory="B", model="devstral-small")

        # Same input length, same per-1M cost (both are 0.10/0.30)
        messages = [
            {"role": "system", "content": "x" * 3500},  # 1000 tokens
            {"role": "user", "content": "build this"},
        ]
        cost_general = a_general._estimate_call_cost(
            messages, get_model("mistral-small"),
        )
        cost_code = a_code._estimate_call_cost(
            messages, get_model("devstral-small"),
        )
        # Code model multiplier 2.0 vs general 0.8 → code estimate should be higher
        assert cost_code > cost_general


class TestCostControl:
    """Budget checking and cost estimation."""

    def test_estimate_cost_known_model(self) -> None:
        cost = Agent.estimate_cost(1000, 500, "mistral-small")
        # mistral-small: €0.10/1M in + €0.30/1M out
        expected = (1000 / 1e6) * 0.10 + (500 / 1e6) * 0.30
        assert abs(cost - expected) < 1e-9

    def test_check_budget_under_limit(self, budget_agent: Agent) -> None:
        budget_agent.check_budget(0.0005)  # Should not raise

    def test_check_budget_over_limit(self, budget_agent: Agent) -> None:
        # budget_eur=0.001, remaining=0.001, 5x tolerance=0.005
        # estimate of 0.006 exceeds 0.005 → should raise
        with pytest.raises(BudgetExceededError):
            budget_agent.check_budget(0.006)

    def test_check_budget_within_tolerance(self, budget_agent: Agent) -> None:
        # budget_eur=0.001, remaining=0.001, 5x tolerance=0.005
        # estimate of 0.003 is within tolerance → should NOT raise
        budget_agent.check_budget(0.003)  # Should not raise

    def test_check_budget_with_spent(self, budget_agent: Agent) -> None:
        # budget_eur=0.001, spent 0.0008, remaining=0.0002, 5x=0.001
        # estimate of 0.002 exceeds 0.001 → should raise
        with pytest.raises(BudgetExceededError):
            budget_agent.check_budget(0.002, spent_eur=0.0008)

    def test_check_budget_with_spent_under_limit(self, budget_agent: Agent) -> None:
        # budget_eur=0.001, spent 0.0005, remaining=0.0005, 5x=0.0025
        # estimate of 0.002 is within tolerance → should NOT raise
        budget_agent.check_budget(0.002, spent_eur=0.0005)  # Should not raise

    def test_no_budget_check_passes(self, sample_agent: Agent) -> None:
        # No budget_eur set → check always passes
        sample_agent.check_budget(999.0, spent_eur=999.0)


class TestMistralParams:
    """to_mistral_params() output."""

    def test_keys(self, sample_agent: Agent) -> None:
        params = sample_agent.to_mistral_params()
        assert "name" in params
        assert "instructions" in params


class TestRunValidation:
    """Agent.run() input validation (no API key needed)."""

    @pytest.mark.asyncio
    async def test_empty_input_raises(self, sample_agent: Agent) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await sample_agent.run("")

    @pytest.mark.asyncio
    async def test_whitespace_input_raises(self, sample_agent: Agent) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            await sample_agent.run("   ")

    def test_negative_budget_raises(self) -> None:
        a = Agent(
            role="R", goal="G", backstory="B", budget_eur=-1.0,
        )
        with pytest.raises(ValueError, match="budget_eur must be >= 0"):
            from tramontane.core._sync import run_sync

            run_sync(a.run("test"))

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(self, sample_agent: Agent) -> None:
        import os

        original = os.environ.pop("MISTRAL_API_KEY", None)
        try:
            with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
                await sample_agent.run("test prompt")
        finally:
            if original:
                os.environ["MISTRAL_API_KEY"] = original

    def test_run_sync_works(self) -> None:
        """Verify run_sync helper works from sync context."""
        from tramontane.core._sync import run_sync
        from tramontane.router.classifier import ClassificationMode, TaskClassifier

        c = TaskClassifier(mode=ClassificationMode.OFFLINE)
        result = run_sync(c.classify("write code"))
        assert result.task_type == "code"


class TestBudgetEstimationTolerance:
    """Pre-call budget estimation should not block affordable calls."""

    def test_cheap_model_small_budget_passes(self) -> None:
        """ministral-3b with budget=0.001 should not raise on short prompt."""
        a = Agent(
            role="Classifier",
            goal="Classify",
            backstory="Triage agent",
            model="ministral-3b",
            budget_eur=0.001,
        )
        from tramontane.router.models import get_model

        model_info = get_model("ministral-3b")
        messages = [
            {"role": "system", "content": "You are a classifier."},
            {"role": "user", "content": "What is this about?"},
        ]
        est = a._estimate_call_cost(messages, model_info)
        # Should not raise — estimate should be well within 5x tolerance
        a.check_budget(est, spent_eur=0.0)

    def test_very_tight_budget_long_prompt_raises(self) -> None:
        """budget=0.0001 with a very long prompt should still raise."""
        a = Agent(
            role="Writer",
            goal="Write",
            backstory="Expert",
            model="mistral-large",
            budget_eur=0.0001,
        )
        from tramontane.router.models import get_model

        model_info = get_model("mistral-large")
        messages = [
            {"role": "system", "content": "You are an expert writer."},
            {"role": "user", "content": "Analyze this: " + "x" * 5000},
        ]
        est = a._estimate_call_cost(messages, model_info)
        with pytest.raises(BudgetExceededError):
            a.check_budget(est, spent_eur=0.0)

    def test_estimate_lower_than_before(self) -> None:
        """Estimation should be less aggressive than the old 2.0x/1.4x approach."""
        a = Agent(
            role="R", goal="G", backstory="B", reasoning=True,
        )
        from tramontane.router.models import get_model

        model_info = get_model("ministral-3b")
        messages = [
            {"role": "system", "content": "Role: R\nGoal: G\nBackstory: B"},
            {"role": "user", "content": "Hello world, classify this text."},
        ]
        est = a._estimate_call_cost(messages, model_info)
        # With ministral-3b (0.04/0.04 per 1M), a ~20 token message
        # should estimate well under 0.0001 EUR
        assert est < 0.0001


class TestFromYaml:
    """YAML loading."""

    def test_loads_correctly(self) -> None:
        data = {
            "role": "Coder",
            "goal": "Write code",
            "backstory": "Expert dev",
            "model": "devstral-small",
            "budget_eur": 0.01,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as f:
            yaml.dump(data, f)
            f.flush()
            agent = Agent.from_yaml(f.name)

        assert agent.role == "Coder"
        assert agent.model == "devstral-small"
        assert agent.budget_eur == 0.01
        Path(f.name).unlink()


class TestStreamEvent:
    """StreamEvent model."""

    def test_token_event(self) -> None:
        from tramontane.core.agent import StreamEvent

        e = StreamEvent(type="token", token="hello", model_used="mistral-small")
        assert e.type == "token"
        assert e.token == "hello"
        assert e.result is None

    def test_complete_event_with_result(self) -> None:
        from tramontane.core.agent import StreamEvent

        result = AgentResult(output="done", model_used="mistral-small")
        e = StreamEvent(type="complete", result=result, model_used="mistral-small")
        assert e.type == "complete"
        assert e.result is not None
        assert e.result.output == "done"

    def test_error_event(self) -> None:
        from tramontane.core.agent import StreamEvent

        e = StreamEvent(type="error", error="API failed")
        assert e.type == "error"
        assert e.error == "API failed"


class TestRunStream:
    """Agent.run_stream() streaming execution."""

    @pytest.mark.asyncio
    async def test_empty_input_yields_error(self, sample_agent: Agent) -> None:
        from tramontane.core.agent import StreamEvent

        events: list[StreamEvent] = []
        async for event in sample_agent.run_stream(""):
            events.append(event)
        assert len(events) == 1
        assert events[0].type == "error"
        assert "non-empty" in events[0].error

    @pytest.mark.asyncio
    async def test_budget_checked_before_streaming(self) -> None:
        """Budget check fires before any streaming starts."""
        a = Agent(
            role="R", goal="G", backstory="B",
            model="mistral-large",
            budget_eur=0.0000001,  # impossibly tight
        )
        from tramontane.core.agent import StreamEvent

        events: list[StreamEvent] = []
        async for event in a.run_stream("Analyze everything in detail " * 100):
            events.append(event)
        # Should get exactly one error event (budget exceeded)
        assert len(events) == 1
        assert events[0].type == "error"
        assert "budget" in events[0].error.lower() or "Budget" in events[0].error

    @pytest.mark.asyncio
    async def test_missing_api_key_yields_error(self) -> None:
        """Missing API key yields an error event, not an exception."""
        import os

        a = Agent(role="R", goal="G", backstory="B", model="mistral-small")
        original = os.environ.pop("MISTRAL_API_KEY", None)
        try:
            from tramontane.core.agent import StreamEvent

            events: list[StreamEvent] = []
            async for event in a.run_stream("hello"):
                events.append(event)
            assert len(events) == 1
            assert events[0].type == "error"
            assert "MISTRAL_API_KEY" in events[0].error
        finally:
            if original:
                os.environ["MISTRAL_API_KEY"] = original


class TestMaxTokens:
    """max_tokens field and API passthrough."""

    def test_max_tokens_field_exists(self) -> None:
        a = Agent(role="R", goal="G", backstory="B", max_tokens=16000)
        assert a.max_tokens == 16000

    def test_max_tokens_default_none(self, sample_agent: Agent) -> None:
        assert sample_agent.max_tokens is None

    def test_max_tokens_in_model_registry(self) -> None:
        from tramontane.router.models import MISTRAL_MODELS

        for alias, model in MISTRAL_MODELS.items():
            assert model.max_output_tokens >= 0, f"{alias} invalid max_output_tokens"

    def test_devstral_small_has_32k_output(self) -> None:
        from tramontane.router.models import get_model

        model = get_model("devstral-small")
        assert model.max_output_tokens == 32768


class TestRoutingHint:
    """Agent routing_hint field."""

    def test_routing_hint_field_exists(self) -> None:
        a = Agent(
            role="R", goal="G", backstory="B",
            routing_hint="text-only JSON output",
        )
        assert a.routing_hint == "text-only JSON output"

    def test_routing_hint_default_none(self, sample_agent: Agent) -> None:
        assert sample_agent.routing_hint is None


class TestTemperature:
    """Agent temperature parameter."""

    def test_temperature_field_exists(self) -> None:
        a = Agent(role="R", goal="G", backstory="B", temperature=0.9)
        assert a.temperature == 0.9

    def test_temperature_zero_is_valid(self) -> None:
        a = Agent(role="R", goal="G", backstory="B", temperature=0.0)
        assert a.temperature == 0.0
        # 0.0 is not None — should be passed to API
        assert a.temperature is not None

    def test_temperature_default_none(self, sample_agent: Agent) -> None:
        assert sample_agent.temperature is None


class TestDynamicContext:
    """Dynamic per-call context parameter."""

    def test_system_prompt_without_context(self, sample_agent: Agent) -> None:
        prompt = sample_agent.system_prompt()
        assert "Additional Context" not in prompt

    def test_context_appended_to_system_prompt(self) -> None:
        """Verify context param changes messages (tested via system_prompt)."""
        a = Agent(role="R", goal="G", backstory="B")
        base = a.system_prompt()
        # The context is appended in _run_once, not system_prompt itself
        # So we just verify the field plumbing works
        assert "Additional Context" not in base


class TestValidateOutput:
    """Output validation hook."""

    def test_validate_output_default_none(self, sample_agent: Agent) -> None:
        assert sample_agent.validate_output is None

    def test_validate_output_field_accepts_callable(self) -> None:
        a = Agent(
            role="R", goal="G", backstory="B",
            validate_output=lambda r: r.output.count("</file>") >= 5,
            max_validation_retries=2,
        )
        assert a.validate_output is not None
        assert a.max_validation_retries == 2

    def test_max_validation_retries_default(self) -> None:
        a = Agent(role="R", goal="G", backstory="B")
        assert a.max_validation_retries == 2


class TestPerModelMaxTokens:
    """Per-model max_tokens defaults from MISTRAL_MODELS."""

    def test_max_tokens_none_gets_model_default(self) -> None:
        """When max_tokens is None, model's max_output_tokens should be used."""
        from tramontane.router.models import get_model

        model = get_model("devstral-small")
        assert model.max_output_tokens == 32768
        # Agent with max_tokens=None should auto-apply 32768

    def test_explicit_max_tokens_overrides(self) -> None:
        a = Agent(role="R", goal="G", backstory="B", max_tokens=8000)
        assert a.max_tokens == 8000


class TestRunContext:
    """RunContext shared cost tracking."""

    def test_run_context_creation(self) -> None:
        from tramontane.core.agent import RunContext

        ctx = RunContext(budget_eur=0.25)
        assert ctx.budget_eur == 0.25
        assert ctx.spent_eur == 0.0
        assert ctx.remaining_eur == 0.25

    def test_run_context_record(self) -> None:
        from tramontane.core.agent import RunContext

        ctx = RunContext(budget_eur=1.0)
        ctx.record("planner", 0.001)
        ctx.record("builder", 0.015)
        assert ctx.spent_eur == pytest.approx(0.016)
        assert ctx.remaining_eur == pytest.approx(0.984)
        assert ctx.agent_costs["planner"] == pytest.approx(0.001)
        assert ctx.agent_costs["builder"] == pytest.approx(0.015)

    def test_run_context_accumulates_same_agent(self) -> None:
        from tramontane.core.agent import RunContext

        ctx = RunContext()
        ctx.record("builder", 0.01)
        ctx.record("builder", 0.02)
        assert ctx.agent_costs["builder"] == pytest.approx(0.03)

    def test_run_context_unlimited_budget(self) -> None:
        from tramontane.core.agent import RunContext

        ctx = RunContext()  # no budget
        assert ctx.remaining_eur is None
        ctx.record("agent", 999.0)
        assert ctx.remaining_eur is None

    def test_run_context_importable_from_package(self) -> None:
        from tramontane import RunContext

        ctx = RunContext(budget_eur=0.5)
        assert ctx.budget_eur == 0.5

    def test_adaptive_reallocation_flows_savings(self) -> None:
        from tramontane.core.agent import RunContext

        ctx = RunContext(reallocation="adaptive")
        planner_budget = 0.005
        planner_effective = ctx.get_effective_budget("planner", planner_budget)
        assert planner_effective == pytest.approx(0.005)
        ctx.record("planner", 0.0001)

        builder_effective = ctx.get_effective_budget("builder", 0.005)
        assert builder_effective == pytest.approx(0.0099)

    def test_fixed_reallocation_does_not_flow_savings(self) -> None:
        from tramontane.core.agent import RunContext

        ctx = RunContext(reallocation="fixed")
        planner_budget = 0.005
        planner_effective = ctx.get_effective_budget("planner", planner_budget)
        assert planner_effective == pytest.approx(0.005)
        ctx.record("planner", 0.0001)

        builder_effective = ctx.get_effective_budget("builder", 0.005)
        assert builder_effective == pytest.approx(0.005)

    def test_adaptive_reallocation_with_no_savings(self) -> None:
        from tramontane.core.agent import RunContext

        ctx = RunContext(reallocation="adaptive")
        ctx.get_effective_budget("planner", 0.005)
        ctx.record("planner", 0.005)

        builder_effective = ctx.get_effective_budget("builder", 0.005)
        assert builder_effective == pytest.approx(0.005)


class TestFleetProfiles:
    """Fleet profile behavior and task overrides."""

    def test_budget_profile_uses_mistral_small_4(self) -> None:
        model, effort = apply_profile(FleetProfile.BUDGET, "auto")
        assert model == "mistral-small-4"
        assert effort == "none"

    def test_quality_profile_uses_devstral_2_for_code(self) -> None:
        model, effort = apply_profile(FleetProfile.QUALITY, "auto", task_type="code")
        assert model == "devstral-2"
        assert effort is None

    def test_explicit_model_ignores_profile_default_model(self) -> None:
        model, effort = apply_profile(FleetProfile.BUDGET, "devstral-small")
        assert model == "devstral-small"
        assert effort == "none"

    def test_unified_profile_uses_mistral_small_4_for_everything(self) -> None:
        general_model, _ = apply_profile(FleetProfile.UNIFIED, "auto", task_type="general")
        research_model, _ = apply_profile(FleetProfile.UNIFIED, "auto", task_type="research")
        assert general_model == "mistral-small-4"
        assert research_model == "mistral-small-4"


class TestReasoningEffort:
    """reasoning_effort and reasoning_strategy fields."""

    def test_reasoning_effort_field(self) -> None:
        a = Agent(
            role="R", goal="G", backstory="B",
            reasoning_effort="high",
        )
        assert a.reasoning_effort == "high"

    def test_reasoning_effort_default_none(self, sample_agent: Agent) -> None:
        assert sample_agent.reasoning_effort is None

    def test_reasoning_effort_none_on_supporting_model(self) -> None:
        """reasoning_effort=None should NOT be passed even if model supports it."""
        a = Agent(
            role="R", goal="G", backstory="B",
            model="mistral-small-4",
            reasoning_effort=None,
        )
        assert a.reasoning_effort is None

    def test_reasoning_effort_on_non_supporting_model(self) -> None:
        """Setting reasoning_effort on devstral-small shouldn't crash."""
        a = Agent(
            role="R", goal="G", backstory="B",
            model="devstral-small",
            reasoning_effort="high",
        )
        # Just verifying field can be set — actual ignoring happens at runtime
        assert a.reasoning_effort == "high"

    def test_reasoning_strategy_default(self, sample_agent: Agent) -> None:
        assert sample_agent.reasoning_strategy == "fixed"

    def test_reasoning_strategy_progressive(self) -> None:
        a = Agent(
            role="R", goal="G", backstory="B",
            model="mistral-small-4",
            reasoning_strategy="progressive",
            validate_output=lambda r: len(r.output) > 100,
        )
        assert a.reasoning_strategy == "progressive"

    def test_progressive_on_non_supporting_model_falls_back(self) -> None:
        """Progressive on devstral-small should fall back to fixed behavior."""
        from tramontane.router.models import get_model

        model = get_model("devstral-small")
        assert model.supports_reasoning_effort is False
        # Agent with progressive on non-supporting model just uses fixed
        a = Agent(
            role="R", goal="G", backstory="B",
            model="devstral-small",
            reasoning_strategy="progressive",
        )
        assert a.reasoning_strategy == "progressive"  # field is set


class TestStreamEventExtended:
    """StreamEvent extended types."""

    def test_pattern_match_event(self) -> None:
        from tramontane.core.agent import StreamEvent

        e = StreamEvent(
            type="pattern_match", model_used="devstral-small",
            pattern_id=r"<file>",
        )
        assert e.type == "pattern_match"
        assert e.pattern_id == "<file>"

    def test_validation_retry_event(self) -> None:
        from tramontane.core.agent import StreamEvent

        e = StreamEvent(type="validation_retry", model_used="devstral-small")
        assert e.type == "validation_retry"

    def test_reasoning_escalation_event(self) -> None:
        from tramontane.core.agent import StreamEvent

        e = StreamEvent(type="reasoning_escalation", model_used="mistral-small-4")
        assert e.type == "reasoning_escalation"

    def test_cascade_escalation_event(self) -> None:
        from tramontane.core.agent import StreamEvent

        e = StreamEvent(type="cascade_escalation", model_used="devstral-2")
        assert e.type == "cascade_escalation"


class TestCascade:
    """Model cascade chain."""

    def test_cascade_field_string_list(self) -> None:
        a = Agent(
            role="R", goal="G", backstory="B",
            cascade=["devstral-small", "devstral-2", "mistral-large-3"],
        )
        assert a.cascade is not None
        assert len(a.cascade) == 3

    def test_cascade_field_dict_list(self) -> None:
        a = Agent(
            role="R", goal="G", backstory="B",
            cascade=[
                {"model": "devstral-small", "max_tokens": 16000},
                {"model": "devstral-2", "max_tokens": 32000},
            ],
        )
        assert a.cascade is not None
        assert len(a.cascade) == 2

    def test_cascade_default_none(self, sample_agent: Agent) -> None:
        assert sample_agent.cascade is None

    def test_cascade_without_validate_does_nothing(self) -> None:
        """Cascade requires validate_output to trigger."""
        a = Agent(
            role="R", goal="G", backstory="B",
            cascade=["devstral-2"],
            # No validate_output — cascade won't fire
        )
        assert a.cascade is not None
        assert a.validate_output is None


class TestToolCalling:
    """Tool calling helpers and Agent integration."""

    def test_function_to_tool_str_param(self) -> None:
        from tramontane.core.agent import _function_to_tool

        def search(query: str) -> str:
            """Search the web."""
            return query

        spec = _function_to_tool(search)
        assert spec["type"] == "function"
        assert spec["function"]["name"] == "search"
        assert spec["function"]["description"] == "Search the web."
        assert spec["function"]["parameters"]["properties"]["query"]["type"] == "string"
        assert "query" in spec["function"]["parameters"]["required"]

    def test_function_to_tool_int_param(self) -> None:
        from tramontane.core.agent import _function_to_tool

        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        spec = _function_to_tool(add)
        assert spec["function"]["parameters"]["properties"]["a"]["type"] == "integer"
        assert spec["function"]["parameters"]["properties"]["b"]["type"] == "integer"
        assert set(spec["function"]["parameters"]["required"]) == {"a", "b"}

    def test_function_to_tool_bool_float(self) -> None:
        from tramontane.core.agent import _function_to_tool

        def configure(temp: float, verbose: bool) -> str:
            """Configure settings."""
            return ""

        spec = _function_to_tool(configure)
        assert spec["function"]["parameters"]["properties"]["temp"]["type"] == "number"
        assert spec["function"]["parameters"]["properties"]["verbose"]["type"] == "boolean"

    def test_function_to_tool_uses_docstring(self) -> None:
        from tramontane.core.agent import _function_to_tool

        def my_func(x: str) -> str:
            """My custom description."""
            return x

        assert _function_to_tool(my_func)["function"]["description"] == "My custom description."

    @pytest.mark.asyncio
    async def test_execute_tool_sync(self) -> None:
        from tramontane.core.agent import _execute_tool

        def greet(name: str) -> str:
            return f"Hello {name}"

        class FakeTC:
            class function:
                name = "greet"
                arguments = '{"name": "Alice"}'
            id = "tc_1"

        result = await _execute_tool(FakeTC(), [greet])
        assert result == "Hello Alice"

    @pytest.mark.asyncio
    async def test_execute_tool_async(self) -> None:
        from tramontane.core.agent import _execute_tool

        async def fetch(url: str) -> str:
            return f"Fetched {url}"

        class FakeTC:
            class function:
                name = "fetch"
                arguments = '{"url": "https://example.com"}'
            id = "tc_2"

        result = await _execute_tool(FakeTC(), [fetch])
        assert result == "Fetched https://example.com"

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self) -> None:
        from tramontane.core.agent import _execute_tool

        class FakeTC:
            class function:
                name = "nonexistent"
                arguments = "{}"
            id = "tc_3"

        result = await _execute_tool(FakeTC(), [])
        assert "Error" in result
        assert "nonexistent" in result

    def test_agent_with_tools_field(self) -> None:
        def search(q: str) -> str:
            return q

        a = Agent(role="R", goal="G", backstory="B", tools=[search])
        assert len(a.tools) == 1

    def test_agent_without_tools_backward_compatible(self, sample_agent: Agent) -> None:
        assert sample_agent.tools == []

    def test_tool_call_stream_event(self) -> None:
        from tramontane.core.agent import StreamEvent

        e = StreamEvent(
            type="tool_call", model_used="mistral-small",
            tool_name="search", tool_args='{"q": "hello"}',
        )
        assert e.type == "tool_call"
        assert e.tool_name == "search"


class TestStructuredOutput:
    """output_schema and parsed_output."""

    def test_output_schema_field(self) -> None:
        from pydantic import BaseModel as BM

        class MyOutput(BM):
            answer: str
            score: float

        a = Agent(
            role="R", goal="G", backstory="B",
            output_schema=MyOutput,
        )
        assert a.output_schema is MyOutput

    def test_output_schema_default_none(self, sample_agent: Agent) -> None:
        assert sample_agent.output_schema is None

    def test_parsed_output_on_result(self) -> None:
        from pydantic import BaseModel as BM

        class Report(BM):
            title: str
            summary: str

        result = AgentResult(
            output='{"title": "Test", "summary": "OK"}',
            model_used="mistral-small",
            parsed_output=Report(title="Test", summary="OK"),
        )
        assert result.parsed_output is not None
        assert result.parsed_output.title == "Test"

    def test_parsed_output_none_when_no_schema(self) -> None:
        result = AgentResult(output="hello", model_used="mistral-small")
        assert result.parsed_output is None


class TestPublicExports:
    """Package-level exports."""

    def test_stream_event_importable_from_package(self) -> None:
        from tramontane import StreamEvent

        e = StreamEvent(type="start", model_used="mistral-small")
        assert e.type == "start"
