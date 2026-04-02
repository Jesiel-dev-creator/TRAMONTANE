"""Fact extraction — auto-extract structured facts from agent output."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """Extract all factual statements from this text.
Return a JSON object with a "facts" key containing an array of objects:
{{"facts": [{{"content": "...", "entity": "...", "category": "fact|..."}}]}}

Rules:
- Each fact should be atomic (ONE piece of information)
- Deduplicate obvious repetitions
- Ignore meta-commentary, only extract substantive facts
- If no facts found, return {{"facts": []}}

Text: {text}

JSON only:"""


@dataclass
class ExtractedFact:
    """A single extracted fact."""

    content: str
    entity: str = ""
    category: str = "fact"


class FactExtractor:
    """Extracts structured facts from text using ministral-3b."""

    def __init__(self, model: str = "ministral-3b-latest") -> None:
        self._model = model

    async def extract(self, text: str) -> list[ExtractedFact]:
        """Extract facts from text via LLM."""
        from mistralai.client import Mistral

        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

        try:
            resp = await client.chat.complete_async(
                model=self._model,
                messages=[  # type: ignore[arg-type]
                    {"role": "user", "content": EXTRACTION_PROMPT.format(text=text)},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw = str(resp.choices[0].message.content or "{}")
            data = json.loads(raw)

            facts_list = data.get("facts", [])
            if isinstance(facts_list, list):
                return [
                    ExtractedFact(
                        content=str(f.get("content", "")),
                        entity=str(f.get("entity", "")),
                        category=str(f.get("category", "fact")),
                    )
                    for f in facts_list
                    if isinstance(f, dict) and f.get("content")
                ]
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.warning("Fact extraction failed: %s", exc)
        except Exception as exc:
            logger.warning("Fact extraction API error: %s", exc)

        return []
