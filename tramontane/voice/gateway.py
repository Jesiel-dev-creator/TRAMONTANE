"""Voxtral-Mini voice input gateway.

Uses the Mistral Audio Transcriptions API (client.audio.transcriptions)
with voxtral-mini-latest. Supports wav, mp3, ogg, m4a, flac, webm.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".webm"}


class VoiceInput(BaseModel):
    """Result of a voice transcription."""

    transcript: str
    confidence: float
    language: str
    duration_seconds: float
    cost_eur: float


class VoiceGateway:
    """Transcribes audio to text using the Mistral Audio Transcriptions API.

    Uses ``client.audio.transcriptions.complete()`` with voxtral-mini-latest.
    Lazy-creates the Mistral client on first use.
    """

    def __init__(
        self,
        api_key: str | None = None,
        language: str = "auto",
    ) -> None:
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self._language = language
        self._client: Any = None

    def _get_client(self) -> Any:
        """Return (and cache) the Mistral client."""
        if self._client is None:
            from mistralai.client import Mistral

            self._client = Mistral(api_key=self._api_key)
        return self._client

    async def transcribe_file(self, audio_path: str) -> VoiceInput:
        """Transcribe an audio file to text via the Audio Transcriptions API."""
        path = Path(audio_path)
        if path.suffix.lower() not in _SUPPORTED_FORMATS:
            msg = (
                f"Unsupported format: {path.suffix}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_FORMATS))}"
            )
            raise ValueError(msg)

        audio_bytes = path.read_bytes()
        file_name = path.name
        return await self._transcribe(audio_bytes, file_name)

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        format: str = "wav",
    ) -> VoiceInput:
        """Transcribe raw audio bytes to text."""
        file_name = f"audio.{format}"
        return await self._transcribe(audio_bytes, file_name)

    async def _transcribe(
        self,
        audio_bytes: bytes,
        file_name: str,
    ) -> VoiceInput:
        """Core transcription using client.audio.transcriptions."""
        client = self._get_client()

        # Estimate duration (rough: wav ~32KB/s at 16kHz mono 16-bit)
        estimated_duration = max(0.1, len(audio_bytes) / 32_000)

        # Language param: None if auto, else pass the code
        lang_param = None if self._language == "auto" else self._language

        try:
            # Use the dedicated Audio Transcriptions API
            response = await client.audio.transcriptions.complete_async(
                model="voxtral-mini-latest",
                file={
                    "content": audio_bytes,
                    "file_name": file_name,
                },
                language=lang_param,
            )

            transcript = response.text or ""

            # Detect language if auto
            detected_lang = self._language
            if detected_lang == "auto":
                detected_lang = self._detect_language_hint(transcript)

            # Estimate cost (Voxtral is Tier 1: EUR 0.04/1M tokens)
            # Rough estimate: ~1 token per 4 chars of transcript
            est_tokens = len(transcript) // 4
            cost = (est_tokens / 1_000_000) * 0.04

            logger.info(
                "Transcribed %.1fs audio -> '%s...'",
                estimated_duration,
                transcript[:50],
            )

            return VoiceInput(
                transcript=transcript,
                confidence=0.90,
                language=detected_lang,
                duration_seconds=round(estimated_duration, 1),
                cost_eur=cost,
            )

        except Exception:
            logger.warning("Voice transcription failed", exc_info=True)
            return VoiceInput(
                transcript="",
                confidence=0.0,
                language=self._language if self._language != "auto" else "unknown",
                duration_seconds=round(estimated_duration, 1),
                cost_eur=0.0,
            )

    def transcribe_file_sync(self, audio_path: str) -> VoiceInput:
        """Synchronous wrapper for transcribe_file()."""
        from tramontane.core._sync import run_sync

        return run_sync(self.transcribe_file(audio_path))

    def is_available(self) -> bool:
        """Check if voice transcription is available."""
        return bool(self._api_key)

    @staticmethod
    def _detect_language_hint(text: str) -> str:
        """Simple language detection from transcript content."""
        from tramontane.router.classifier import _detect_language

        return _detect_language(text)
