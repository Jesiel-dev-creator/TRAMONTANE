"""Skill loader — load skills from Python modules, directories, YAML, SKILL.md."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any

import yaml

from tramontane.skills.base import Skill, SkillResult, track_skill

logger = logging.getLogger(__name__)


class MarkdownSkill(Skill):
    """Skill loaded from a SKILL.md file.

    Compatible with OpenClaw/Claude Code SKILL.md format.
    Returns instructions as context for the calling agent.
    """

    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        triggers: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self._instructions = instructions
        self.triggers = triggers or []

    @track_skill
    async def execute(
        self, input_text: str, context: dict[str, Any] | None = None,
    ) -> SkillResult:
        """Return skill instructions as output."""
        return SkillResult(
            output=self._instructions,
            success=True,
            metadata={"type": "markdown_skill"},
        )


class YamlSkill(Skill):
    """Skill loaded from a YAML definition file.

    Inspired by NVIDIA NeMo Agent Toolkit workflow.yml.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.name = str(config.get("name", ""))
        self.description = str(config.get("description", ""))
        self.version = str(config.get("version", "1.0"))
        self.triggers = list(config.get("triggers", []))
        self.preferred_model = str(config.get("preferred_model", "auto"))
        self.preferred_temperature = config.get("temperature")
        self.budget_eur = config.get("budget_eur")
        self.memory_tags = list(config.get("memory_tags", []))
        self.tags = list(config.get("tags", []))
        self._prompt = str(config.get("prompt", ""))
        self._output_format = str(config.get("output_format", "text"))

    @track_skill
    async def execute(
        self, input_text: str, context: dict[str, Any] | None = None,
    ) -> SkillResult:
        """Execute by creating a temporary Agent with YAML config."""
        from tramontane.core.agent import Agent

        kwargs: dict[str, Any] = {
            "role": self.name,
            "goal": self.description,
            "backstory": self._prompt,
            "model": self.preferred_model,
        }
        if self.preferred_temperature is not None:
            kwargs["temperature"] = self.preferred_temperature
        if self.budget_eur is not None:
            kwargs["budget_eur"] = self.budget_eur
        if self._output_format == "json":
            kwargs["output_schema"] = None  # JSON mode without schema

        agent = Agent(**kwargs)
        result = await agent.run(input_text)

        return SkillResult(
            output=result.output,
            cost_eur=result.cost_eur,
            model_used=result.model_used,
            success=True,
            metadata={"type": "yaml_skill"},
        )


class SkillLoader:
    """Load skills from various sources."""

    @staticmethod
    def load_from_module(module_path: str) -> list[Skill]:
        """Import a Python module and find all Skill subclasses."""
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            logger.warning("Could not import module: %s", module_path)
            return []

        skills: list[Skill] = []
        for _name, obj in inspect.getmembers(mod, inspect.isclass):
            if (
                issubclass(obj, Skill)
                and obj is not Skill
                and not inspect.isabstract(obj)
            ):
                skills.append(obj())
        return skills

    @staticmethod
    def load_from_directory(path: str) -> list[Skill]:
        """Scan directory for .py files containing Skill subclasses."""
        dir_path = Path(path)
        if not dir_path.is_dir():
            logger.warning("Not a directory: %s", path)
            return []

        skills: list[Skill] = []
        for py_file in dir_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = py_file.stem
            spec = importlib.util.spec_from_file_location(
                module_name, py_file,
            )
            if not spec or not spec.loader:
                continue
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
            except Exception:
                logger.warning("Failed to load %s", py_file)
                continue

            for _name, obj in inspect.getmembers(mod, inspect.isclass):
                if (
                    issubclass(obj, Skill)
                    and obj is not Skill
                    and not inspect.isabstract(obj)
                ):
                    skills.append(obj())
        return skills

    @staticmethod
    def load_from_skill_md(path: str) -> MarkdownSkill:
        """Parse a SKILL.md file into a MarkdownSkill.

        Expects optional YAML frontmatter (--- delimited) with:
        name, description, triggers. Body is the instructions.
        """
        text = Path(path).read_text(encoding="utf-8")

        name = Path(path).stem
        description = ""
        triggers: list[str] = []
        instructions = text

        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                frontmatter = yaml.safe_load(parts[1]) or {}
                name = str(frontmatter.get("name", name))
                description = str(frontmatter.get("description", ""))
                triggers = list(frontmatter.get("triggers", []))
                instructions = parts[2].strip()

        return MarkdownSkill(
            name=name,
            description=description,
            instructions=instructions,
            triggers=triggers,
        )

    @staticmethod
    def load_from_yaml(path: str) -> YamlSkill:
        """Load a skill definition from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return YamlSkill(data)
