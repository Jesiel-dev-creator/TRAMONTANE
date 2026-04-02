"""Tests for tramontane.skills.loader — SkillLoader + MarkdownSkill + YamlSkill."""

from __future__ import annotations

from pathlib import Path

import pytest

from tramontane.skills.loader import MarkdownSkill, SkillLoader, YamlSkill


class TestMarkdownSkill:
    @pytest.mark.asyncio
    async def test_execute_returns_instructions(self) -> None:
        skill = MarkdownSkill(
            name="test", description="Test",
            instructions="Do this specific thing.",
            triggers=["test"],
        )
        result = await skill.execute("hello")
        assert result.output == "Do this specific thing."
        assert result.success is True
        assert result.metadata["type"] == "markdown_skill"


class TestLoadFromSkillMd:
    def test_parses_frontmatter(self, tmp_path: Path) -> None:
        md_file = tmp_path / "SKILL.md"
        md_file.write_text(
            "---\nname: my_skill\ndescription: Does stuff\n"
            "triggers: [do, make]\n---\n\nInstructions here.",
        )
        skill = SkillLoader.load_from_skill_md(str(md_file))
        assert skill.name == "my_skill"
        assert skill.description == "Does stuff"
        assert skill.triggers == ["do", "make"]

    def test_no_frontmatter(self, tmp_path: Path) -> None:
        md_file = tmp_path / "plain.md"
        md_file.write_text("Just instructions, no YAML.")
        skill = SkillLoader.load_from_skill_md(str(md_file))
        assert skill.name == "plain"
        assert "Just instructions" in skill._instructions


class TestYamlSkill:
    def test_loads_config(self) -> None:
        config = {
            "name": "qualifier",
            "description": "Score leads",
            "triggers": ["qualify", "score"],
            "preferred_model": "ministral-3b-latest",
            "temperature": 0.1,
            "budget_eur": 0.001,
            "prompt": "You are a lead qualifier.",
        }
        skill = YamlSkill(config)
        assert skill.name == "qualifier"
        assert skill.preferred_model == "ministral-3b-latest"
        assert skill.budget_eur == 0.001

    def test_load_from_yaml_file(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "skill.yaml"
        yaml_file.write_text(
            "name: test\ndescription: Test skill\n"
            "triggers: [test]\nprompt: Do the thing\n",
        )
        skill = SkillLoader.load_from_yaml(str(yaml_file))
        assert skill.name == "test"


class TestImports:
    def test_loader_importable(self) -> None:
        from tramontane import SkillLoader
        assert SkillLoader is not None

    def test_markdown_skill_importable(self) -> None:
        from tramontane import MarkdownSkill
        assert MarkdownSkill is not None
