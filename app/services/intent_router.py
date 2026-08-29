"""Lightweight intent classification — replaces brittle keyword regex.

One cheap, fast LLM call reads the actual message and decides how to
route it. This is deliberately NOT the full multi-step agent — it's a
single classification call, not a reasoning loop. Think "receptionist
pointing you to a department" vs "caseworker doing an investigation."
That fuller version is a later upgrade (the real agent workflow).
"""

import logging
from enum import Enum

from app.core.llm import GeminiClient

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    CASUAL = "casual"          # greeting / thanks / small talk — no real question
    NEEDS_PDF = "needs_pdf"    # answerable from the uploaded document
    NEEDS_WEB = "needs_web"    # needs current/live internet info
    NEEDS_BOTH = "needs_both"  # needs the document AND live web info together


_CLASSIFY_PROMPT = """Classify the user's message into exactly ONE label based on its underlying requirement.

Labels:
- casual: Greetings, small talk, OR requests for advice, brainstorming, writing, coding, capability explanations, and general knowledge where internal AI knowledge is sufficient (no live news or document lookup needed).
- needs_pdf: Questions specifically asking about uploaded documents, file contents, summaries, or specific file data.
- needs_web: Questions strictly requiring real-time internet data, breaking news, live prices, current events, or post-cutoff information.
- needs_both: Questions explicitly asking to compare facts inside the uploaded document against live internet news/data.

Examples (follow this pattern exactly, even when a message looks borderline):
- "can you write a piece of code for me to reverse a string" -> casual
- "explain how binary search works" -> casual
- "what's the time complexity of quicksort" -> casual
- "how are you doing today" -> casual
- "what does my document say about the deadline" -> needs_pdf
- "what's the current price of gold" -> needs_web
- "compare the numbers in my document with the latest figures" -> needs_both

Has a document been uploaded this session: {has_pdf}

Message: "{query}"

Respond with ONLY the label word in lowercase (casual, needs_pdf, needs_web, needs_both).

Label:"""

_VALID_LABELS = {i.value for i in Intent}


class IntentRouter:
    def __init__(self, llm_client: GeminiClient):
        self._llm = llm_client

    def classify(self, query: str, has_pdf: bool) -> Intent:
        """Classify a message. Falls back to a safe default if the LLM call
        fails or returns something unparseable — never raises, since a
        classification failure shouldn't take down the whole chat response.
        """
        prompt = _CLASSIFY_PROMPT.format(has_pdf=has_pdf, query=query)
        try:
            raw = self._llm.generate(
                system_prompt="You are a precise classifier. Output only the label word.",
                user_message=prompt,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Intent classification failed, using fallback: %s", exc)
            return Intent.NEEDS_PDF if has_pdf else Intent.NEEDS_WEB

        label = raw.strip().lower().strip(".\"'")
        if label not in _VALID_LABELS:
            logger.warning("Unrecognized intent label '%s', using fallback", label)
            return Intent.NEEDS_PDF if has_pdf else Intent.NEEDS_WEB

        return Intent(label)
