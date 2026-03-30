"""PII detection with dual-mode ONLINE/OFFLINE (same pattern as TaskClassifier).

OFFLINE: regex patterns for emails, phones, IBAN, French NIR, passports,
credit cards, IP addresses, and basic name detection.
ONLINE: Ministral-3B for contextual PII that regex misses.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import re

from pydantic import BaseModel

from tramontane.router.classifier import ClassificationMode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class PIIType(enum.Enum):
    """Categories of personally identifiable information."""

    EMAIL = "email"
    PHONE = "phone"
    IBAN = "iban"
    NIR = "nir"
    PASSPORT = "passport"
    NAME = "name"
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"
    CREDIT_CARD = "credit_card"
    CUSTOM = "custom"


class PIIDetection(BaseModel):
    """A single PII detection within text."""

    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float
    redacted_value: str


class PIIResult(BaseModel):
    """Full result of PII scanning on a text."""

    original_text: str
    cleaned_text: str
    detections: list[PIIDetection]
    has_pii: bool
    pii_types_found: list[PIIType]
    mode_used: ClassificationMode


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[PIIType, str, str]] = [
    # (type, regex, redaction_label)
    (
        PIIType.EMAIL,
        r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b",
        "[EMAIL]",
    ),
    (
        PIIType.PHONE,
        # French: +33, 06/07, 0X XX XX XX XX; international with +
        r"(?:\+33[\s.-]?\d[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}[\s.-]?\d{2}"
        r"|0[1-9](?:[\s.-]?\d{2}){4}"
        r"|\+\d{1,3}[\s.-]?\d[\s.-]?\d{2,4}[\s.-]?\d{2,4}[\s.-]?\d{2,4})",
        "[PHONE]",
    ),
    (
        PIIType.IBAN,
        # French IBAN + general European IBAN
        r"\b[A-Z]{2}\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{1,4}\b",
        "[IBAN]",
    ),
    (
        PIIType.NIR,
        # French Numéro de Sécurité Sociale
        r"\b[12]\d{2}(?:0[1-9]|1[0-2])\d{5}\d{3}\d{2}\b",
        "[NIR]",
    ),
    (
        PIIType.PASSPORT,
        # French passport format
        r"\b[0-9]{2}[A-Z]{2}\d{5}\b",
        "[PASSPORT]",
    ),
    (
        PIIType.CREDIT_CARD,
        # 16-digit card numbers (with optional separators)
        r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
        "[CREDIT_CARD]",
    ),
    (
        PIIType.IP_ADDRESS,
        # IPv4
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
        "[IP]",
    ),
]

# Name detection: title-cased word pairs after common prefixes
_NAME_PREFIXES = r"(?:Monsieur|Madame|Mme|M\.|Dr\.?|Prof\.?|Mr\.?|Mrs\.?|Ms\.?)"
_NAME_PATTERN = re.compile(
    rf"{_NAME_PREFIXES}\s+([A-ZÀ-Ü][a-zà-ü]+(?:\s+[A-ZÀ-Ü][a-zà-ü]+)+)",
    re.UNICODE,
)

_CLASSIFIER_PROMPT = (
    "You are a PII detector. Analyze the text and return ONLY a JSON array "
    "of objects with these fields:\n"
    '{"pii_type": "email"|"phone"|"iban"|"nir"|"passport"|"name"|'
    '"address"|"ip_address"|"credit_card"|"custom", '
    '"value": "<detected text>", "start": <int>, "end": <int>}\n'
    "Return [] if no PII found. Return ONLY valid JSON."
)


# ---------------------------------------------------------------------------
# PIIDetector
# ---------------------------------------------------------------------------


class PIIDetector:
    """Detects PII with dual-mode ONLINE/OFFLINE.

    Same fallback pattern as TaskClassifier: if the API key is missing
    or the call fails, drops to OFFLINE regex mode with a warning.
    """

    def __init__(
        self,
        mode: ClassificationMode = ClassificationMode.ONLINE,
        api_key: str | None = None,
        locale: str = "fr",
    ) -> None:
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self._mode = mode
        self._locale = locale

        if self._mode == ClassificationMode.ONLINE and not self._api_key:
            logger.warning(
                "No MISTRAL_API_KEY — switching PII detector to OFFLINE mode"
            )
            self._mode = ClassificationMode.OFFLINE

    @property
    def mode(self) -> ClassificationMode:
        """Current detection mode."""
        return self._mode

    async def detect(self, text: str) -> PIIResult:
        """Scan text for PII and return redacted result."""
        # Always run regex first
        regex_detections = self._detect_offline(text)

        online_detections: list[PIIDetection] = []
        if self._mode == ClassificationMode.ONLINE:
            online_detections = await self._detect_online(text)

        # Merge and deduplicate (prefer higher confidence)
        merged = self._merge_detections(regex_detections, online_detections)
        cleaned = self.redact(text, merged)

        pii_types_found = list({d.pii_type for d in merged})

        return PIIResult(
            original_text=text,
            cleaned_text=cleaned,
            detections=merged,
            has_pii=len(merged) > 0,
            pii_types_found=pii_types_found,
            mode_used=self._mode,
        )

    # -- OFFLINE regex path ------------------------------------------------

    def _detect_offline(self, text: str) -> list[PIIDetection]:
        """Detect PII using regex patterns."""
        detections: list[PIIDetection] = []

        for pii_type, pattern, label in _PATTERNS:
            for match in re.finditer(pattern, text):
                detections.append(
                    PIIDetection(
                        pii_type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.85,
                        redacted_value=label,
                    )
                )

        # Name detection via prefix patterns
        for match in _NAME_PATTERN.finditer(text):
            name = match.group(1)
            full_start = match.start(1)
            detections.append(
                PIIDetection(
                    pii_type=PIIType.NAME,
                    value=name,
                    start=full_start,
                    end=full_start + len(name),
                    confidence=0.65,
                    redacted_value="[NAME]",
                )
            )

        return detections

    # -- ONLINE path (Ministral-3B) ----------------------------------------

    async def _detect_online(self, text: str) -> list[PIIDetection]:
        """Detect PII using Ministral-3B for contextual analysis."""
        try:
            from mistralai.client import Mistral

            client = Mistral(api_key=self._api_key)
            response = await client.chat.complete_async(
                model="ministral-3b-latest",
                messages=[  # type: ignore[arg-type]
                    {"role": "system", "content": _CLASSIFIER_PROMPT},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
            )
            raw = str(response.choices[0].message.content)
            data = json.loads(raw)

            items = data if isinstance(data, list) else data.get("detections", [])
            detections: list[PIIDetection] = []
            for item in items:
                pii_type_str = item.get("pii_type", "custom")
                try:
                    pii_type = PIIType(pii_type_str)
                except ValueError:
                    pii_type = PIIType.CUSTOM
                detections.append(
                    PIIDetection(
                        pii_type=pii_type,
                        value=item.get("value", ""),
                        start=item.get("start", 0),
                        end=item.get("end", 0),
                        confidence=0.92,
                        redacted_value=f"[{pii_type.value.upper()}]",
                    )
                )
            return detections

        except Exception:
            logger.warning(
                "Online PII detection failed — using regex only",
                exc_info=True,
            )
            return []

    # -- Redaction ---------------------------------------------------------

    @staticmethod
    def redact(text: str, detections: list[PIIDetection]) -> str:
        """Apply redactions right-to-left to preserve character positions."""
        # Sort by start position descending
        sorted_dets = sorted(detections, key=lambda d: d.start, reverse=True)
        result = text
        for det in sorted_dets:
            result = result[:det.start] + det.redacted_value + result[det.end:]
        return result

    # -- Merge / dedup -----------------------------------------------------

    @staticmethod
    def _merge_detections(
        regex: list[PIIDetection],
        online: list[PIIDetection],
    ) -> list[PIIDetection]:
        """Merge regex and online detections, deduplicating by position overlap."""
        if not online:
            return regex

        merged = list(regex)
        for o_det in online:
            # Check if any regex detection overlaps
            overlaps = any(
                r_det.start <= o_det.end and o_det.start <= r_det.end
                for r_det in regex
            )
            if not overlaps:
                merged.append(o_det)

        return sorted(merged, key=lambda d: d.start)

    # -- Sync wrapper ------------------------------------------------------

    def detect_sync(self, text: str) -> PIIResult:
        """Synchronous wrapper for detect(). Do not call from async context."""
        from tramontane.core._sync import run_sync

        return run_sync(self.detect(text))
