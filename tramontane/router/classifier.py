"""Task classification for the Tramontane router.

Supports ONLINE mode (real Ministral-3B API call) and OFFLINE mode
(keyword heuristic fallback, zero API cost).  OFFLINE is the default
when MISTRAL_API_KEY is not set — tests and CI never need a key.
"""

from __future__ import annotations

import enum
import json
import logging
import os
import re
from typing import Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

TaskType = Literal[
    "code", "reasoning", "research", "vision",
    "bulk", "general", "classification", "voice",
]

# Valid task types the router can handle
VALID_TASK_TYPES: set[str] = {
    "code", "reasoning", "general", "bulk", "vision",
    "research", "classification", "voice",
}

# Mapping for common invalid types the LLM might return
TASK_TYPE_ALIASES: dict[str, str] = {
    "design": "general",
    "analysis": "reasoning",
    "writing": "general",
    "translation": "general",
    "summarization": "general",
    "extraction": "bulk",
    "coding": "code",
    "programming": "code",
    "math": "reasoning",
    "creative": "general",
    "conversation": "general",
    "search": "research",
}


def _validate_task_type(raw_type: str) -> str:
    """Validate and normalize classifier output to a known task type."""
    normalized = raw_type.lower().strip()
    if normalized in VALID_TASK_TYPES:
        return normalized
    if normalized in TASK_TYPE_ALIASES:
        logger.warning(
            "Classifier returned '%s', remapping to '%s'",
            raw_type, TASK_TYPE_ALIASES[normalized],
        )
        return TASK_TYPE_ALIASES[normalized]
    logger.warning(
        "Classifier returned unknown task type '%s', defaulting to 'general'",
        raw_type,
    )
    return "general"
GDPRSensitivity = Literal["none", "low", "high"]


class ClassificationMode(enum.Enum):
    """How the classifier determines task type."""

    ONLINE = "online"
    OFFLINE = "offline"


class ClassificationResult(BaseModel):
    """Result of classifying a user prompt."""

    task_type: TaskType
    complexity: int
    has_code: bool
    has_vision: bool
    needs_reasoning: bool
    estimated_output_tokens: int
    language: str
    gdpr_sensitivity: GDPRSensitivity
    mode_used: ClassificationMode
    confidence: float


# ---------------------------------------------------------------------------
# Keyword sets for OFFLINE classification
# ---------------------------------------------------------------------------

_CODE_KEYWORDS: set[str] = {
    "write", "build", "code", "function", "class", "fix", "debug",
    "implement", "refactor", "script", "bug", "error", "syntax",
}
_CODE_PATTERNS: list[str] = [
    r"```", r"def\s+\w+", r"class\s+\w+", r"import\s+\w+",
    r"\w+\.\w+\(", r"if\s+.*:", r"for\s+\w+\s+in",
]
_REASONING_KEYWORDS: set[str] = {
    "analyze", "explain", "why", "compare", "evaluate", "assess",
    "plan", "design", "decide", "should", "recommend",
}
_RESEARCH_KEYWORDS: set[str] = {
    "find", "search", "research", "latest",
    "news", "information",
}
_RESEARCH_PHRASES: list[str] = ["what is", "who is", "tell me about"]
_BULK_KEYWORDS: set[str] = {"list", "enumerate", "all", "every", "batch"}
_CREATIVE_KEYWORDS: set[str] = {
    "poem", "essay", "generate", "compose", "draft", "narrative", "fiction",
}
_CREATIVE_PHRASES: list[str] = ["write a", "create a story"]
_VISION_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp", ".tiff",
}

# Language-detection markers (need ≥ 3 hits to trigger)
_LANG_MARKERS: dict[str, set[str]] = {
    "fr": {
        "le", "la", "les", "de", "du", "un", "une", "est", "sont", "avec",
        "pour", "dans", "que", "qui", "je", "nous", "vous", "faire",
    },
    "de": {
        "der", "die", "das", "und", "ist", "ein", "eine", "ich", "wir",
        "sie", "nicht", "auch", "mit", "auf", "für",
    },
    "es": {
        "el", "los", "las", "un", "una", "por", "que", "con", "para",
        "como", "más", "este", "esta", "yo",
    },
    "it": {
        "il", "lo", "le", "di", "una", "sono", "con", "per", "che",
        "non", "questo", "questa", "io", "noi",
    },
    "pt": {
        "os", "as", "em", "um", "uma", "são", "com", "para", "que",
        "não", "este", "esta", "eu", "nós",
    },
}

# System prompt for the ONLINE classifier (ministral-3b)
_CLASSIFIER_SYSTEM_PROMPT = (
    "You are a task classifier for the Tramontane agent framework.\n"
    "Analyze the user's prompt and return ONLY a JSON object with these fields:\n"
    '{\n'
    '  "task_type": "code"|"reasoning"|"research"|"vision"'
    '|"bulk"|"general"|"classification"|"voice",\n'
    '  "complexity": <int 1-5>,\n'
    '  "has_code": <bool>,\n'
    '  "has_vision": <bool>,\n'
    '  "needs_reasoning": <bool>,\n'
    '  "estimated_output_tokens": <int>,\n'
    '  "language": "<ISO 639-1>",\n'
    '  "gdpr_sensitivity": "none"|"low"|"high"\n'
    "}\n"
    "Return ONLY valid JSON, no explanations."
)

# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------


def _detect_language(text: str) -> str:
    """Detect language from common word markers. Returns ISO 639-1 code."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    best_lang, best_score = "en", 0
    for lang, markers in _LANG_MARKERS.items():
        score = len(words & markers)
        if score > best_score:
            best_lang, best_score = lang, score
    return best_lang if best_score >= 3 else "en"


def _has_code_content(text: str) -> bool:
    """Check if text contains code-like patterns."""
    return any(re.search(p, text) for p in _CODE_PATTERNS)


def _has_vision_content(context: str | None) -> bool:
    """Check if context references image files or vision content."""
    if not context:
        return False
    lower = context.lower()
    return any(ext in lower for ext in _VISION_EXTENSIONS) or "image" in lower


def _detect_gdpr_sensitivity(text: str) -> GDPRSensitivity:
    """Simple heuristic for GDPR sensitivity detection."""
    high_patterns = [
        r"\b\d{2,3}[-.\s]?\d{2,3}[-.\s]?\d{4}\b",  # phone-like
        r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",       # SSN-like
        r"\b[A-Z]{2}\d{2}\s?\w{4}",                   # IBAN-like
    ]
    low_patterns = [
        r"\b[\w.-]+@[\w.-]+\.\w+\b",  # email
    ]
    for pattern in high_patterns:
        if re.search(pattern, text):
            return "high"
    for pattern in low_patterns:
        if re.search(pattern, text):
            return "low"
    return "none"


def _phrase_match(text: str, phrases: list[str]) -> int:
    """Count how many phrases appear in text."""
    lower = text.lower()
    return sum(1 for p in phrases if p in lower)


# ---------------------------------------------------------------------------
# TaskClassifier
# ---------------------------------------------------------------------------


class TaskClassifier:
    """Classifies user prompts for model routing.

    Supports ONLINE mode (Ministral-3B API) and OFFLINE mode (keyword heuristics).
    Auto-switches to OFFLINE when no API key is available.
    """

    def __init__(
        self,
        mode: ClassificationMode = ClassificationMode.ONLINE,
        api_key: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self._mode = mode

        if self._mode == ClassificationMode.ONLINE and not self._api_key:
            logger.warning(
                "No MISTRAL_API_KEY found — switching classifier to OFFLINE mode"
            )
            self._mode = ClassificationMode.OFFLINE

        logger.info("TaskClassifier initialized in %s mode", self._mode.value)

    @property
    def mode(self) -> ClassificationMode:
        """Current classification mode."""
        return self._mode

    async def classify(
        self,
        prompt: str,
        context: str | None = None,
    ) -> ClassificationResult:
        """Classify a prompt to determine task type, complexity, and routing hints."""
        if self._mode == ClassificationMode.ONLINE:
            return await self._classify_online(prompt, context)
        return self._classify_offline(prompt, context)

    # -- ONLINE path -------------------------------------------------------

    async def _classify_online(
        self,
        prompt: str,
        context: str | None = None,
    ) -> ClassificationResult:
        """Classify using a real Ministral-3B API call."""
        try:
            from mistralai.client import Mistral

            client = Mistral(api_key=self._api_key)
            user_content = prompt
            if context:
                user_content = f"{prompt}\n\nContext: {context}"

            response = await client.chat.complete_async(
                model="ministral-3b-latest",
                messages=[  # type: ignore[arg-type]
                    {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            data = json.loads(str(content))
            # Validate/normalize task_type before Pydantic validation
            if "task_type" in data:
                data["task_type"] = _validate_task_type(str(data["task_type"]))
            data["mode_used"] = ClassificationMode.ONLINE
            data["confidence"] = 0.95
            return ClassificationResult.model_validate(data)
        except Exception:
            logger.warning(
                "Online classification failed — falling back to OFFLINE",
                exc_info=True,
            )
            return self._classify_offline(prompt, context)

    # -- OFFLINE path (keyword heuristics) ---------------------------------

    def _classify_offline(
        self,
        prompt: str,
        context: str | None = None,
    ) -> ClassificationResult:
        """Classify using keyword heuristics (zero API cost)."""
        lower = prompt.lower()
        words = set(re.findall(r"\b\w+\b", lower))

        has_vision = _has_vision_content(context)
        has_code = _has_code_content(prompt) or bool(words & _CODE_KEYWORDS)

        # Score each task category
        code_score = len(words & _CODE_KEYWORDS) + (3 if has_code else 0)
        reasoning_score = len(words & _REASONING_KEYWORDS)
        research_score = (
            len(words & _RESEARCH_KEYWORDS) + _phrase_match(prompt, _RESEARCH_PHRASES)
        )
        bulk_score = len(words & _BULK_KEYWORDS)
        creative_score = (
            len(words & _CREATIVE_KEYWORDS) + _phrase_match(prompt, _CREATIVE_PHRASES)
        )

        # Decide task_type via explicit branches (mypy-safe Literal assignment)
        task_type: TaskType
        if has_vision:
            task_type = "vision"
        elif code_score >= 2:
            task_type = "code"
        elif reasoning_score >= 2:
            task_type = "reasoning"
        elif research_score >= 2:
            task_type = "research"
        elif creative_score >= 2:
            task_type = "general"
        elif bulk_score >= 2 or len(prompt) < 50:
            task_type = "bulk"
        else:
            task_type = "general"

        complexity = min(5, max(1, len(prompt) // 200 + 1))

        needs_reasoning = (
            task_type == "reasoning"
            or reasoning_score >= 2
            or complexity >= 4
        )

        estimated_output_tokens = complexity * 250

        return ClassificationResult(
            task_type=task_type,
            complexity=complexity,
            has_code=has_code,
            has_vision=has_vision,
            needs_reasoning=needs_reasoning,
            estimated_output_tokens=estimated_output_tokens,
            language=_detect_language(prompt),
            gdpr_sensitivity=_detect_gdpr_sensitivity(prompt),
            mode_used=ClassificationMode.OFFLINE,
            confidence=0.70,
        )

    # -- Sync wrapper ------------------------------------------------------

    def classify_sync(
        self,
        prompt: str,
        context: str | None = None,
    ) -> ClassificationResult:
        """Synchronous wrapper for classify(). Do not call from async context."""
        from tramontane.core._sync import run_sync

        return run_sync(self.classify(prompt, context))
