"""Tests for tramontane.voice.tts — VoicePipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from tramontane.core.agent import Agent, AgentResult
from tramontane.voice.tts import VoicePipeline


def _make_agent() -> Agent:
    return Agent(role="Support", goal="Help users", backstory="Expert")


def _make_result(output: str = "Hello!") -> AgentResult:
    return AgentResult(output=output, model_used="mistral-small", cost_eur=0.001)


class TestVoicePipeline:
    """VoicePipeline execution."""

    @pytest.mark.asyncio
    async def test_text_input_skips_transcription(self) -> None:
        agent = _make_agent()
        mock_run = AsyncMock(return_value=_make_result("Response"))

        with patch.object(Agent, "run", mock_run):
            vp = VoicePipeline(agent=agent, enable_tts=False)
            result = await vp.run(text_input="Hello")

        assert result.transcript == "Hello"
        assert result.agent_output == "Response"
        assert result.audio_bytes is None

    @pytest.mark.asyncio
    async def test_tts_disabled_no_audio(self) -> None:
        agent = _make_agent()

        with patch.object(Agent, "run", AsyncMock(return_value=_make_result())):
            vp = VoicePipeline(agent=agent, enable_tts=False)
            result = await vp.run(text_input="test")

        assert result.audio_bytes is None
        assert "voxtral-tts" not in result.models_used

    @pytest.mark.asyncio
    async def test_cost_includes_agent(self) -> None:
        agent = _make_agent()

        with patch.object(
            Agent, "run",
            AsyncMock(return_value=_make_result()),
        ):
            vp = VoicePipeline(agent=agent, enable_tts=False)
            result = await vp.run(text_input="test")

        assert result.cost_eur >= 0.001

    @pytest.mark.asyncio
    async def test_run_text_convenience(self) -> None:
        agent = _make_agent()

        with patch.object(Agent, "run", AsyncMock(return_value=_make_result())):
            vp = VoicePipeline(agent=agent, enable_tts=False)
            result = await vp.run_text("test")

        assert result.transcript == "test"

    @pytest.mark.asyncio
    async def test_no_input_raises(self) -> None:
        agent = _make_agent()
        vp = VoicePipeline(agent=agent)

        with pytest.raises(ValueError, match="audio_input or text_input"):
            await vp.run()

    @pytest.mark.asyncio
    async def test_tts_failure_graceful(self) -> None:
        agent = _make_agent()

        with (
            patch.object(Agent, "run", AsyncMock(return_value=_make_result())),
            patch.object(
                VoicePipeline, "_speak",
                AsyncMock(side_effect=RuntimeError("TTS down")),
            ),
        ):
            vp = VoicePipeline(agent=agent, enable_tts=True)
            result = await vp.run(text_input="test")

        # Should still return text, just no audio
        assert result.agent_output == "Hello!"
        assert result.audio_bytes is None


class TestVoiceImports:
    """Package-level imports."""

    def test_voice_pipeline_importable(self) -> None:
        from tramontane import VoicePipeline

        assert VoicePipeline is not None

    def test_voice_result_importable(self) -> None:
        from tramontane import VoiceResult

        assert VoiceResult is not None
