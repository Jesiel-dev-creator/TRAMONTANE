"""Tests for tramontane.core.parallel — parallel agent execution."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tramontane.core.agent import Agent, AgentResult
from tramontane.core.parallel import ParallelGroup, ParallelResult


def _make_agent(role: str) -> Agent:
    return Agent(role=role, goal="Test", backstory="Test agent")


def _make_result(role: str, output: str, cost: float = 0.001) -> AgentResult:
    return AgentResult(output=output, model_used="mistral-small", cost_eur=cost)


class TestParallelResult:
    """ParallelResult data operations."""

    def test_merge_concatenates_outputs(self) -> None:
        pr = ParallelResult(
            results={
                "A": _make_result("A", "Output A"),
                "B": _make_result("B", "Output B"),
            },
        )
        merged = pr.merge()
        assert "Output A" in merged
        assert "Output B" in merged

    def test_merge_with_custom_separator(self) -> None:
        pr = ParallelResult(
            results={
                "A": _make_result("A", "A"),
                "B": _make_result("B", "B"),
            },
        )
        assert pr.merge(separator=" | ") == "A | B"

    def test_get_by_role(self) -> None:
        result_a = _make_result("A", "Output A")
        pr = ParallelResult(results={"A": result_a})
        assert pr.get("A") is result_a
        assert pr.get("Missing") is None


class TestParallelGroup:
    """ParallelGroup concurrent execution."""

    @pytest.mark.asyncio
    async def test_runs_two_agents(self) -> None:
        result_a = _make_result("Designer", "Colors: warm palette")
        result_b = _make_result("Architect", "Components: Header, Main, Footer")

        mock_run = AsyncMock(side_effect=[result_a, result_b])
        with patch.object(Agent, "run", mock_run):
            group = ParallelGroup([_make_agent("Designer"), _make_agent("Architect")])
            pr = await group.run(input_text="Design a bakery website")

        assert len(pr.results) == 2
        assert mock_run.call_count == 2

    @pytest.mark.asyncio
    async def test_shared_input(self) -> None:
        result = _make_result("Worker", "Done")
        mock_run = AsyncMock(return_value=result)

        with patch.object(Agent, "run", mock_run):
            group = ParallelGroup([_make_agent("Worker")])
            await group.run(input_text="shared prompt")

        mock_run.assert_called_once()
        assert mock_run.call_args[0][0] == "shared prompt"

    @pytest.mark.asyncio
    async def test_per_agent_inputs(self) -> None:
        calls: list[str] = []

        async def capture_run(self_agent: Agent, input_text: str, **_kw: object) -> AgentResult:
            calls.append(f"{self_agent.role}:{input_text}")
            return _make_result(self_agent.role, "ok")

        with patch.object(Agent, "run", capture_run):
            group = ParallelGroup([_make_agent("Designer"), _make_agent("Architect")])
            await group.run(inputs={
                "Designer": "Pick colors",
                "Architect": "Plan structure",
            })

        assert "Designer:Pick colors" in calls
        assert "Architect:Plan structure" in calls

    @pytest.mark.asyncio
    async def test_handles_one_error(self) -> None:
        call_count = 0

        async def sometimes_fail(self_agent: Agent, input_text: str, **_kw: object) -> AgentResult:
            nonlocal call_count
            call_count += 1
            if self_agent.role == "Fail":
                msg = "API down"
                raise RuntimeError(msg)
            return _make_result(self_agent.role, "Success")

        with patch.object(Agent, "run", sometimes_fail):
            group = ParallelGroup([_make_agent("OK"), _make_agent("Fail")])
            pr = await group.run(input_text="test")

        assert len(pr.results) == 1
        assert pr.get("OK") is not None
        assert "Fail" in pr.errors

    @pytest.mark.asyncio
    async def test_total_cost_summed(self) -> None:
        results_iter = iter([
            _make_result("A", "a", cost=0.002),
            _make_result("B", "b", cost=0.003),
        ])
        mock_run = AsyncMock(side_effect=lambda *a, **kw: next(results_iter))

        with patch.object(Agent, "run", mock_run):
            group = ParallelGroup([_make_agent("A"), _make_agent("B")])
            pr = await group.run(input_text="x")

        assert pr.total_cost_eur == pytest.approx(0.005)

    @pytest.mark.asyncio
    async def test_actually_concurrent(self) -> None:
        """Verify parallel execution is faster than sequential."""
        import asyncio

        async def slow_run(_self: Agent, _input: str, **_kw: object) -> AgentResult:
            await asyncio.sleep(0.1)
            return _make_result("x", "done")

        with patch.object(Agent, "run", slow_run):
            group = ParallelGroup([_make_agent("Slow1"), _make_agent("Slow2")])
            pr = await group.run(input_text="x")

        # Parallel: ~0.1s. Sequential would be ~0.2s.
        assert pr.total_duration_s < 0.18
        assert len(pr.results) == 2

    def test_agents_property(self) -> None:
        agents = [_make_agent("A"), _make_agent("B")]
        group = ParallelGroup(agents)
        assert group.agents == agents


class TestParallelImports:
    """Package-level imports."""

    def test_parallel_group_importable(self) -> None:
        from tramontane import ParallelGroup

        assert ParallelGroup is not None

    def test_parallel_result_importable(self) -> None:
        from tramontane import ParallelResult

        assert ParallelResult is not None
