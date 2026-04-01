"""Voice pipeline — speech-to-text -> agent -> text-to-speech."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class VoiceResult:
    """Result from a voice pipeline execution."""

    transcript: str
    agent_output: str
    audio_bytes: bytes | None = None
    cost_eur: float = 0.0
    models_used: list[str] = field(default_factory=list)


class VoicePipeline:
    """Full voice-to-voice agent pipeline.

    Speech -> Voxtral Transcription -> Agent -> Voxtral TTS -> Audio
    """

    def __init__(
        self,
        agent: Any,
        transcribe_model: str = "voxtral-mini-latest",
        speak_model: str = "voxtral-tts-2603",
        voice: str = "alloy",
        enable_tts: bool = True,
    ) -> None:
        self._agent = agent
        self._transcribe_model = transcribe_model
        self._speak_model = speak_model
        self._voice = voice
        self._enable_tts = enable_tts

    async def _transcribe(self, audio_bytes: bytes) -> str:
        """Transcribe audio to text using Voxtral."""
        from tramontane.voice.gateway import VoiceGateway

        gw = VoiceGateway()
        result = await gw.transcribe_bytes(audio_bytes)
        return str(result.transcript)

    async def _speak(self, text: str) -> bytes:
        """Convert text to speech using Voxtral TTS via REST API."""
        import httpx

        api_key = os.environ.get("MISTRAL_API_KEY", "")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.mistral.ai/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._speak_model,
                    "input": text,
                    "voice": self._voice,
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.content

    async def run(
        self,
        audio_input: bytes | None = None,
        text_input: str | None = None,
        *,
        router: Any | None = None,
    ) -> VoiceResult:
        """Run the voice pipeline.

        Provide either audio_input (transcribed first) or text_input.
        """
        cost = 0.0
        models: list[str] = []

        # Step 1: Transcribe (if audio input)
        if audio_input is not None:
            transcript = await self._transcribe(audio_input)
            models.append(self._transcribe_model)
        elif text_input is not None:
            transcript = text_input
        else:
            msg = "Provide either audio_input or text_input"
            raise ValueError(msg)

        # Step 2: Agent processing
        result = await self._agent.run(transcript, router=router)
        cost += result.cost_eur
        models.append(result.model_used)

        # Step 3: TTS (if enabled)
        audio_out = None
        if self._enable_tts:
            try:
                audio_out = await self._speak(result.output)
                models.append(self._speak_model)
                # Estimate TTS cost: $0.016 per 1K chars, convert to EUR
                tts_cost = len(result.output) / 1000 * 0.016 * 0.92
                cost += tts_cost
            except Exception as exc:
                logger.warning("TTS failed: %s (returning text only)", exc)

        return VoiceResult(
            transcript=transcript,
            agent_output=result.output,
            audio_bytes=audio_out,
            cost_eur=cost,
            models_used=models,
        )

    async def run_text(
        self,
        text: str,
        *,
        router: Any | None = None,
    ) -> VoiceResult:
        """Convenience: text in -> agent -> speech out."""
        return await self.run(text_input=text, router=router)
