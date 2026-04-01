"""Tests for tramontane.core.agent."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from tramontane.core.agent import Agent, AgentResult
from tramontane.core.exceptions import BudgetExceededError


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
        # budget_eur=0.001, remaining=0.001, 2x tolerance=0.002
        # estimate of 0.003 exceeds 0.002 → should raise
        with pytest.raises(BudgetExceededError):
            budget_agent.check_budget(0.003)

    def test_check_budget_within_tolerance(self, budget_agent: Agent) -> None:
        # budget_eur=0.001, remaining=0.001, 2x tolerance=0.002
        # estimate of 0.0015 is within tolerance → should NOT raise
        budget_agent.check_budget(0.0015)  # Should not raise

    def test_check_budget_with_spent(self, budget_agent: Agent) -> None:
        # budget_eur=0.001, spent 0.0008, remaining=0.0002, 2x=0.0004
        # estimate of 0.001 exceeds 0.0004 → should raise
        with pytest.raises(BudgetExceededError):
            budget_agent.check_budget(0.001, spent_eur=0.0008)

    def test_check_budget_with_spent_under_limit(self, budget_agent: Agent) -> None:
        # budget_eur=0.001, spent 0.0005, remaining=0.0005, 2x=0.001
        # estimate of 0.0003 is within tolerance → should NOT raise
        budget_agent.check_budget(0.0003, spent_eur=0.0005)  # Should not raise

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
        # Should not raise — estimate should be well within 2x tolerance
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
