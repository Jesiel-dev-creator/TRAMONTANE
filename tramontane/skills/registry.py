"""Skill registry — discover, register, search, and track skills."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import sqlite3
from typing import Any

from tramontane.skills.base import Skill

logger = logging.getLogger(__name__)

DEFAULT_DB = "tramontane_skills.db"


class SkillRegistry:
    """Registry for discovering, storing, and tracking skills."""

    def __init__(self, db_path: str = DEFAULT_DB) -> None:
        self._db_path = db_path
        self._skills: dict[str, Skill] = {}
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Create registry table."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS skill_registry (
                name TEXT PRIMARY KEY,
                description TEXT,
                version TEXT,
                triggers TEXT,
                preferred_model TEXT,
                memory_tags TEXT,
                author TEXT,
                tags TEXT,
                embedding BLOB,
                hash TEXT,
                total_executions INTEGER DEFAULT 0,
                successful_executions INTEGER DEFAULT 0,
                avg_cost_eur REAL DEFAULT 0,
                avg_duration_s REAL DEFAULT 0,
                avg_quality_score REAL DEFAULT 0,
                best_model TEXT,
                best_temperature REAL,
                last_used_at DATETIME,
                registered_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._conn.commit()

    def verify_skill(self, skill: Skill) -> dict[str, Any]:
        """Security verification before registration."""
        warnings: list[str] = []

        if not skill.name:
            warnings.append("Skill has no name")
        if not skill.description:
            warnings.append("Skill has no description")

        skill_hash = ""
        try:
            source = inspect.getsource(type(skill))
            skill_hash = hashlib.sha256(source.encode()).hexdigest()
            dangerous = [
                "os.system", "subprocess", "eval(", "exec(",
                "__import__", "shutil.rmtree",
            ]
            for pattern in dangerous:
                if pattern in source:
                    warnings.append(f"Dangerous pattern: {pattern}")
        except (OSError, TypeError):
            warnings.append("Could not inspect skill source")

        return {
            "verified": len(warnings) == 0,
            "hash": skill_hash,
            "warnings": warnings,
        }

    def register(self, skill: Skill, verify: bool = True) -> None:
        """Register a skill."""
        if verify:
            check = self.verify_skill(skill)
            if not check["verified"]:
                logger.warning(
                    "Skill '%s' warnings: %s", skill.name, check["warnings"],
                )
            skill_hash = check["hash"]
        else:
            skill_hash = ""

        self._skills[skill.name] = skill
        self._conn.execute(
            """INSERT OR REPLACE INTO skill_registry
               (name, description, version, triggers, preferred_model,
                memory_tags, author, tags, hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                skill.name,
                skill.description,
                skill.version,
                json.dumps(skill.triggers),
                skill.preferred_model,
                json.dumps(skill.memory_tags),
                skill.author,
                json.dumps(skill.tags),
                skill_hash,
            ),
        )
        self._conn.commit()
        logger.info("Registered skill '%s' v%s", skill.name, skill.version)

    def unregister(self, name: str) -> bool:
        """Remove a skill."""
        if name not in self._skills:
            return False
        del self._skills[name]
        self._conn.execute(
            "DELETE FROM skill_registry WHERE name = ?", (name,),
        )
        self._conn.commit()
        return True

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def search(
        self, query: str, top_k: int = 5,
    ) -> list[tuple[Skill, float]]:
        """Search skills by keyword matching."""
        scored: list[tuple[Skill, float]] = []
        for skill in self._skills.values():
            score = skill.matches(query)
            if score > 0:
                scored.append((skill, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def list_all(self) -> list[Skill]:
        """List all registered skills."""
        return list(self._skills.values())

    def get_by_tag(self, tag: str) -> list[Skill]:
        """Filter skills by tag."""
        return [s for s in self._skills.values() if tag in s.tags]

    def record_execution(
        self,
        name: str,
        success: bool,
        cost: float,
        duration: float,
        quality_score: float = 0.0,
    ) -> None:
        """Update execution stats for a skill."""
        row = self._conn.execute(
            "SELECT total_executions, successful_executions, "
            "avg_cost_eur, avg_duration_s, avg_quality_score "
            "FROM skill_registry WHERE name = ?",
            (name,),
        ).fetchone()
        if not row:
            return

        total = int(row["total_executions"]) + 1
        succ = int(row["successful_executions"]) + (1 if success else 0)
        # Running average
        prev_cost = float(row["avg_cost_eur"] or 0)
        prev_dur = float(row["avg_duration_s"] or 0)
        prev_qual = float(row["avg_quality_score"] or 0)
        avg_cost = prev_cost + (cost - prev_cost) / total
        avg_dur = prev_dur + (duration - prev_dur) / total
        avg_qual = prev_qual + (quality_score - prev_qual) / total

        self._conn.execute(
            """UPDATE skill_registry
               SET total_executions=?, successful_executions=?,
                   avg_cost_eur=?, avg_duration_s=?, avg_quality_score=?,
                   last_used_at=CURRENT_TIMESTAMP
               WHERE name=?""",
            (total, succ, avg_cost, avg_dur, avg_qual, name),
        )
        self._conn.commit()

    async def semantic_search(
        self, query: str, top_k: int = 5,
    ) -> list[tuple[Skill, float]]:
        """Find skills by semantic similarity. Falls back to keyword."""
        rows = self._conn.execute(
            "SELECT name, embedding FROM skill_registry "
            "WHERE embedding IS NOT NULL",
        ).fetchall()

        if not rows:
            return self.search(query, top_k=top_k)

        import math
        import os

        from mistralai.client import Mistral

        client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY", ""))
        resp = await client.embeddings.create_async(
            model="mistral-embed", inputs=[query],
        )
        q_emb = resp.data[0].embedding if resp.data[0].embedding else []
        if not q_emb:
            return self.search(query, top_k=top_k)

        scored: list[tuple[str, float]] = []
        for row in rows:
            stored = json.loads(row["embedding"]) if row["embedding"] else []
            if stored:
                dot = sum(a * b for a, b in zip(q_emb, stored))
                na = math.sqrt(sum(a * a for a in q_emb))
                nb = math.sqrt(sum(b * b for b in stored))
                sim = dot / (na * nb) if na and nb else 0.0
                scored.append((row["name"], sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        results: list[tuple[Skill, float]] = []
        for name, score in scored[:top_k]:
            skill = self._skills.get(name)
            if skill:
                results.append((skill, score))
        return results
