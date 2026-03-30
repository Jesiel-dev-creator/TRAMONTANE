"""Tests for tramontane.core.agent."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from tramontane.core.agent import Agent
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
        with pytest.raises(BudgetExceededError):
            budget_agent.check_budget(0.002)

    def test_check_budget_with_spent(self, budget_agent: Agent) -> None:
        # budget_eur=0.001, already spent 0.0008 → 0.0003 more should fail
        with pytest.raises(BudgetExceededError):
            budget_agent.check_budget(0.0003, spent_eur=0.0008)

    def test_check_budget_with_spent_under_limit(self, budget_agent: Agent) -> None:
        # budget_eur=0.001, spent 0.0005, estimating 0.0003 → total 0.0008 < 0.001
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
