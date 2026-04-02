"""Tests for tramontane.skills.builtin — built-in skills metadata."""

from __future__ import annotations

from tramontane.skills.builtin import (
    ALL_BUILTIN_SKILLS,
    CodeGenerationSkill,
    DataExtractionSkill,
    EmailDraftSkill,
    TextAnalysisSkill,
    WebSearchSkill,
)
from tramontane.skills.registry import SkillRegistry


class TestBuiltinSkillMetadata:
    def test_text_analysis_triggers(self) -> None:
        s = TextAnalysisSkill()
        assert "analyze" in s.triggers
        assert s.preferred_model == "mistral-small-4"

    def test_code_generation_triggers(self) -> None:
        s = CodeGenerationSkill()
        assert "code" in s.triggers
        assert s.preferred_model == "devstral-small"

    def test_email_draft_triggers(self) -> None:
        s = EmailDraftSkill()
        assert "email" in s.triggers
        assert s.preferred_temperature == 0.7

    def test_data_extraction_triggers(self) -> None:
        s = DataExtractionSkill()
        assert "extract" in s.triggers
        assert s.preferred_model == "ministral-3b"

    def test_web_search_triggers(self) -> None:
        s = WebSearchSkill()
        assert "search" in s.triggers
        assert "research" in s.triggers

    def test_all_have_names(self) -> None:
        for cls in ALL_BUILTIN_SKILLS:
            s = cls()
            assert s.name, f"{cls.__name__} has no name"
            assert s.description, f"{cls.__name__} has no description"

    def test_all_registerable(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/reg.db"  # type: ignore[operator]
        reg = SkillRegistry(db_path=db)
        for cls in ALL_BUILTIN_SKILLS:
            reg.register(cls())
        assert len(reg.list_all()) == 5

    def test_five_builtin_skills(self) -> None:
        assert len(ALL_BUILTIN_SKILLS) == 5
