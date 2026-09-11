"""Extracts durable facts about the user from their messages.

One small LLM call per message, asking "is there anything worth
remembering here?" — a name, a preference, an ongoing project. Merges
into the existing fact dict rather than replacing it, so facts learned
across many different messages/sessions accumulate instead of overwriting
each other.
"""
import json
import logging

from app.core.llm import GeminiClient

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """Look at this message and decide if it contains any durable fact worth \
remembering long-term about the person (their name, a stated preference, an ongoing project, \
their role/profession, or similar). Ignore anything temporary, casual, or already obvious.

Message: "{message}"

Respond with ONLY a JSON object. If there's a fact worth storing, use short snake_case keys, \
e.g. {{"name": "Alex", "role": "backend developer"}}. If there's nothing worth storing, \
respond with exactly: {{}}

JSON:"""


class MemoryExtractor:
    def __init__(self, llm_client: GeminiClient, model_name: str | None = None):
        self._llm = llm_client
        self._model_name = model_name

    def extract(self, message: str) -> dict:
        """Returns a dict of new facts found in this message, or {} if none.
        Never raises — a failed extraction just means nothing new was
        learned this turn, not a broken chat response.
        """
        try:
            response = self._llm.generate(
                system_prompt="You extract durable facts as JSON. Output only valid JSON, nothing else.",
                user_message=_EXTRACT_PROMPT.format(message=message),
                temperature=0.0,
                model=self._model_name,
            )
            logger.debug("Memory extraction raw response: %s", response)
            cleaned = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            facts = json.loads(cleaned)
            logger.debug("Memory extraction parsed facts: %s", facts)
            return facts if isinstance(facts, dict) else {}
        except Exception as exc:
            logger.warning("Memory extraction failed, skipping this turn: %s", exc)
            return {}

    @staticmethod
    def merge(existing: dict, new_facts: dict) -> dict:
        """New facts win on conflict (assume the person corrected themselves),
        but nothing already known is dropped."""
        merged = dict(existing)
        merged.update(new_facts)
        return merged
