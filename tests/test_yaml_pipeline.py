"""Tests for tramontane.core.yaml_pipeline — YAML pipeline definitions."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tramontane.core.yaml_pipeline import (
    AgentSpec,
    PipelineSpec,
    create_agents_from_spec,
    load_pipeline_spec,
)


def _write_yaml(data: dict, tmp_path: Path) -> Path:
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


VALID_YAML = {
    "name": "Test Pipeline",
    "version": "1.0",
    "budget_eur": 0.01,
    "agents": {
        "researcher": {
            "role": "Researcher",
            "goal": "Research topics",
            "backstory": "Expert researcher",
            "model": "mistral-small",
            "temperature": 0.5,
        },
        "writer": {
            "role": "Writer",
            "goal": "Write content",
            "backstory": "Expert writer",
            "model": "devstral-small",
            "budget_eur": 0.005,
        },
    },
    "flow": ["researcher", "writer"],
}


class TestLoadPipelineSpec:
    """load_pipeline_spec() YAML loading."""

    def test_loads_valid_yaml(self, tmp_path: Path) -> None:
        path = _write_yaml(VALID_YAML, tmp_path)
        spec = load_pipeline_spec(path)
        assert spec.name == "Test Pipeline"
        assert spec.version == "1.0"
        assert spec.budget_eur == 0.01
        assert len(spec.agents) == 2
        assert len(spec.flow) == 2

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_pipeline_spec("/nonexistent/pipeline.yaml")

    def test_invalid_flow_raises(self, tmp_path: Path) -> None:
        data = {**VALID_YAML, "flow": ["researcher", "nonexistent"]}
        path = _write_yaml(data, tmp_path)
        with pytest.raises(ValueError, match="nonexistent"):
            load_pipeline_spec(path)


class TestPipelineSpecValidation:
    """PipelineSpec.validate_flow()."""

    def test_valid_flow(self) -> None:
        spec = PipelineSpec(**VALID_YAML)
        assert spec.validate_flow() == []

    def test_undefined_agent_in_flow(self) -> None:
        data = {**VALID_YAML, "flow": ["researcher", "ghost"]}
        spec = PipelineSpec(**data)
        errors = spec.validate_flow()
        assert len(errors) == 1
        assert "ghost" in errors[0]


class TestAgentSpec:
    """AgentSpec.to_agent() conversion."""

    def test_basic_conversion(self) -> None:
        spec = AgentSpec(
            role="Writer",
            goal="Write",
            backstory="Expert",
            model="devstral-small",
        )
        agent = spec.to_agent()
        assert agent.role == "Writer"
        assert agent.model == "devstral-small"

    def test_temperature_passed(self) -> None:
        spec = AgentSpec(
            role="R", goal="G", backstory="B",
            temperature=0.8,
        )
        agent = spec.to_agent()
        assert agent.temperature == 0.8

    def test_budget_passed(self) -> None:
        spec = AgentSpec(
            role="R", goal="G", backstory="B",
            budget_eur=0.005,
        )
        agent = spec.to_agent()
        assert agent.budget_eur == 0.005

    def test_reasoning_effort_passed(self) -> None:
        spec = AgentSpec(
            role="R", goal="G", backstory="B",
            reasoning_effort="high",
        )
        agent = spec.to_agent()
        assert agent.reasoning_effort == "high"


class TestCreateAgents:
    """create_agents_from_spec() flow-ordered creation."""

    def test_creates_in_flow_order(self) -> None:
        spec = PipelineSpec(**VALID_YAML)
        agents = create_agents_from_spec(spec)
        assert len(agents) == 2
        assert agents[0].role == "Researcher"
        assert agents[1].role == "Writer"


class TestYamlPipelineImports:
    """Package-level imports."""

    def test_load_pipeline_spec_importable(self) -> None:
        from tramontane import load_pipeline_spec

        assert load_pipeline_spec is not None

    def test_pipeline_spec_importable(self) -> None:
        from tramontane import PipelineSpec

        assert PipelineSpec is not None

    def test_run_yaml_pipeline_importable(self) -> None:
        from tramontane import run_yaml_pipeline

        assert run_yaml_pipeline is not None
