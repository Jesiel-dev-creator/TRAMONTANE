"""Tests for tramontane.skills.registry — SkillRegistry."""

from __future__ import annotations

from typing import Any

from tramontane.skills.base import Skill, SkillResult
from tramontane.skills.registry import SkillRegistry


class _TestSkill(Skill):
    name = "test_skill"
    description = "A test skill for registry tests"
    triggers = ["test", "hello"]
    tags = ["testing"]

    async def execute(
        self, input_text: str, context: dict[str, Any] | None = None,
    ) -> SkillResult:
        return SkillResult(output="ok", success=True)


class _OtherSkill(Skill):
    name = "other_skill"
    description = "Another test skill"
    triggers = ["other"]
    tags = ["other"]

    async def execute(
        self, input_text: str, context: dict[str, Any] | None = None,
    ) -> SkillResult:
        return SkillResult(output="ok", success=True)


class TestRegister:
    def test_register_stores_skill(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/reg.db"  # type: ignore[operator]
        reg = SkillRegistry(db_path=db)
        reg.register(_TestSkill())
        assert reg.get("test_skill") is not None

    def test_get_returns_none_for_unknown(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/reg.db"  # type: ignore[operator]
        reg = SkillRegistry(db_path=db)
        assert reg.get("nonexistent") is None


class TestUnregister:
    def test_unregister_removes(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/reg.db"  # type: ignore[operator]
        reg = SkillRegistry(db_path=db)
        reg.register(_TestSkill())
        assert reg.unregister("test_skill") is True
        assert reg.get("test_skill") is None

    def test_unregister_nonexistent(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/reg.db"  # type: ignore[operator]
        reg = SkillRegistry(db_path=db)
        assert reg.unregister("nope") is False


class TestSearch:
    def test_search_by_trigger(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/reg.db"  # type: ignore[operator]
        reg = SkillRegistry(db_path=db)
        reg.register(_TestSkill())
        reg.register(_OtherSkill())
        results = reg.search("test something")
        assert len(results) >= 1
        assert results[0][0].name == "test_skill"

    def test_search_no_match(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/reg.db"  # type: ignore[operator]
        reg = SkillRegistry(db_path=db)
        reg.register(_TestSkill())
        assert reg.search("xyz completely unrelated") == []


class TestListAndFilter:
    def test_list_all(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/reg.db"  # type: ignore[operator]
        reg = SkillRegistry(db_path=db)
        reg.register(_TestSkill())
        reg.register(_OtherSkill())
        assert len(reg.list_all()) == 2

    def test_get_by_tag(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/reg.db"  # type: ignore[operator]
        reg = SkillRegistry(db_path=db)
        reg.register(_TestSkill())
        reg.register(_OtherSkill())
        testing = reg.get_by_tag("testing")
        assert len(testing) == 1
        assert testing[0].name == "test_skill"


class TestVerify:
    def test_verify_valid_skill(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/reg.db"  # type: ignore[operator]
        reg = SkillRegistry(db_path=db)
        check = reg.verify_skill(_TestSkill())
        assert check["verified"] is True
        assert check["hash"]  # non-empty SHA-256

    def test_verify_empty_name_warns(self, tmp_path: object) -> None:
        class NoName(Skill):
            name = ""
            description = ""
            async def execute(self, i: str, c: Any = None) -> SkillResult:
                return SkillResult()

        db = str(tmp_path) + "/reg.db"  # type: ignore[operator]
        reg = SkillRegistry(db_path=db)
        check = reg.verify_skill(NoName())
        assert check["verified"] is False
        assert any("no name" in w.lower() for w in check["warnings"])


class TestRecordExecution:
    def test_updates_stats(self, tmp_path: object) -> None:
        db = str(tmp_path) + "/reg.db"  # type: ignore[operator]
        reg = SkillRegistry(db_path=db)
        reg.register(_TestSkill())
        reg.record_execution("test_skill", True, 0.001, 1.5, 0.9)
        row = reg._conn.execute(
            "SELECT total_executions, successful_executions FROM skill_registry WHERE name=?",
            ("test_skill",),
        ).fetchone()
        assert row["total_executions"] == 1
        assert row["successful_executions"] == 1


class TestImports:
    def test_registry_importable(self) -> None:
        from tramontane import SkillRegistry
        assert SkillRegistry is not None
