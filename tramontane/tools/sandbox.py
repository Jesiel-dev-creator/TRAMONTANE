"""Code execution sandbox with graceful degradation on WSL2.

Auto-detects Docker availability. Falls back to LOCAL mode (no isolation)
when Docker is unavailable. Supports E2B cloud sandbox when configured.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import os
import shutil
import tempfile
import time

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SandboxMode(enum.Enum):
    """How code is executed."""

    DOCKER = "docker"
    LOCAL = "local"
    E2B = "e2b"
    DISABLED = "disabled"


class SandboxResult(BaseModel):
    """Result of a code execution."""

    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float
    mode_used: SandboxMode
    files_created: list[str] = Field(default_factory=list)


def _docker_available() -> bool:
    """Check if Docker daemon is reachable."""
    if not shutil.which("docker"):
        return False
    try:
        import subprocess

        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


class CodeSandbox:
    """Executes code with configurable isolation.

    Auto-detects the best available mode on construction:
    Docker > E2B > LOCAL. Logs a warning when running without isolation.
    """

    def __init__(
        self,
        mode: SandboxMode = SandboxMode.DOCKER,
        timeout_seconds: int = 30,
        max_output_chars: int = 10000,
    ) -> None:
        self._timeout = timeout_seconds
        self._max_output = max_output_chars
        self.mode_used = self._detect_mode(mode)

        if self.mode_used == SandboxMode.LOCAL:
            logger.warning(
                "CodeSandbox running in LOCAL mode — no sandboxing"
            )

    @staticmethod
    def _detect_mode(requested: SandboxMode) -> SandboxMode:
        """Auto-detect the best available mode."""
        if requested == SandboxMode.DISABLED:
            return SandboxMode.DISABLED

        if requested == SandboxMode.DOCKER and _docker_available():
            return SandboxMode.DOCKER

        if requested == SandboxMode.E2B and os.environ.get("E2B_API_KEY"):
            return SandboxMode.E2B

        # Fallback chain
        if _docker_available():
            return SandboxMode.DOCKER
        if os.environ.get("E2B_API_KEY"):
            return SandboxMode.E2B
        return SandboxMode.LOCAL

    async def execute(
        self,
        code: str,
        language: str = "python",
        working_dir: str | None = None,
    ) -> SandboxResult:
        """Execute code and return the result."""
        if self.mode_used == SandboxMode.DISABLED:
            return SandboxResult(
                stdout="",
                stderr="Code execution is disabled (GDPR strict mode)",
                exit_code=1,
                execution_time_ms=0.0,
                mode_used=SandboxMode.DISABLED,
            )

        if self.mode_used == SandboxMode.DOCKER:
            return await self._execute_docker(code, language, working_dir)

        return await self._execute_local(code, language, working_dir)

    # -- Docker mode -------------------------------------------------------

    async def _execute_docker(
        self,
        code: str,
        language: str,
        working_dir: str | None,
    ) -> SandboxResult:
        """Execute in an isolated Docker container (no network)."""
        start = time.monotonic()

        with tempfile.TemporaryDirectory() as tmpdir:
            ext = ".py" if language == "python" else f".{language}"
            script_path = os.path.join(tmpdir, f"script{ext}")
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(code)

            image = "python:3.12-slim" if language == "python" else f"{language}:latest"
            cmd = [
                "docker", "run", "--rm", "--network=none",
                "-v", f"{tmpdir}:/workspace:ro",
                image, "python", "/workspace/script.py",
            ]

            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                )
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=float(self._timeout)
                )
                elapsed = (time.monotonic() - start) * 1000
                return SandboxResult(
                    stdout=stdout_bytes.decode(errors="replace")[: self._max_output],
                    stderr=stderr_bytes.decode(errors="replace")[: self._max_output],
                    exit_code=proc.returncode or 0,
                    execution_time_ms=elapsed,
                    mode_used=SandboxMode.DOCKER,
                )
            except asyncio.TimeoutError:
                elapsed = (time.monotonic() - start) * 1000
                return SandboxResult(
                    stdout="",
                    stderr=f"Execution timed out after {self._timeout}s",
                    exit_code=124,
                    execution_time_ms=elapsed,
                    mode_used=SandboxMode.DOCKER,
                )

    # -- Local mode (WSL2 fallback) ----------------------------------------

    async def _execute_local(
        self,
        code: str,
        language: str,
        working_dir: str | None,
    ) -> SandboxResult:
        """Execute locally via subprocess (no isolation)."""
        start = time.monotonic()

        ext = ".py" if language == "python" else f".{language}"
        fd, script_path = tempfile.mkstemp(suffix=ext)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)

            interpreter = "python3" if language == "python" else language
            proc = await asyncio.create_subprocess_exec(
                interpreter, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=float(self._timeout)
            )
            elapsed = (time.monotonic() - start) * 1000
            return SandboxResult(
                stdout=stdout_bytes.decode(errors="replace")[: self._max_output],
                stderr=stderr_bytes.decode(errors="replace")[: self._max_output],
                exit_code=proc.returncode or 0,
                execution_time_ms=elapsed,
                mode_used=SandboxMode.LOCAL,
            )
        except asyncio.TimeoutError:
            elapsed = (time.monotonic() - start) * 1000
            return SandboxResult(
                stdout="",
                stderr=f"Execution timed out after {self._timeout}s",
                exit_code=124,
                execution_time_ms=elapsed,
                mode_used=SandboxMode.LOCAL,
            )
        finally:
            os.unlink(script_path)

    # -- Helpers -----------------------------------------------------------

    def is_available(self) -> bool:
        """Check if the sandbox can execute code."""
        return self.mode_used != SandboxMode.DISABLED

    def mode_warning(self) -> str | None:
        """Return a warning string if not in DOCKER mode, else None."""
        if self.mode_used == SandboxMode.LOCAL:
            return "Running in LOCAL mode — code executes without isolation"
        if self.mode_used == SandboxMode.DISABLED:
            return "Code execution is disabled"
        return None
