"""Smoke tests for tramontane CLI commands."""

from __future__ import annotations

from typer.testing import CliRunner

from tramontane.cli.main import app

runner = CliRunner()


class TestCLISmoke:
    """CLI commands don't crash."""

    def test_version(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "TRAMONTANE" in result.output

    def test_models(self) -> None:
        result = runner.invoke(app, ["models"])
        assert result.exit_code == 0
        assert "MISTRAL MODEL FLEET" in result.output

    def test_doctor(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 0
        assert "Version" in result.output

    def test_fleet(self) -> None:
        result = runner.invoke(app, ["fleet"])
        assert result.exit_code == 0
        assert "mistral-small-4" in result.output

    def test_watch(self) -> None:
        result = runner.invoke(app, ["watch"])
        assert result.exit_code == 0

    def test_simulate_missing_file(self) -> None:
        result = runner.invoke(app, ["simulate", "/nonexistent.yaml"])
        assert result.exit_code == 1

    def test_telemetry_stats_empty(self) -> None:
        result = runner.invoke(app, ["telemetry", "stats", "--db", ":memory:"])
        # Empty telemetry exits 0 with a warning
        assert result.exit_code == 0

    def test_knowledge_search_empty_db(self) -> None:
        result = runner.invoke(
            app, ["knowledge", "search", "test", "--db", ":memory:"],
        )
        assert result.exit_code == 1
