"""Reranking using Gemini API.

v2: API-based reranking using Gemini as the judge. Takes the query + up to
25 candidate chunks (already fused by RRF), asks Gemini to return the top 5
chunk numbers ordered by actual relevance. No new dependency, no new API key,
uses the same Gemini client already configured in the app.

Fallback behavior: on any parse/API failure, returns the original RRF order
untouched — never let reranking failure break retrieval.
"""
import logging
import re

from app.core.llm import GeminiClient

logger = logging.getLogger(__name__)


_RERANK_PROMPT = """You are a relevance judge. Given a question and candidate text passages numbered [1] through [{max_num}], return ONLY the top {top_n} passage numbers that are most relevant to answering the question.

Rules:
- Consider both direct relevance and usefulness for answering the specific question
- Return exactly {top_n} numbers, separated by spaces, in order of most relevant to least relevant
- If fewer than {top_n} passages exist, return all available passage numbers
- Output ONLY the numbers, no explanation, no extra text

Question: {query}

Passages:
{passages}

Top {top_n} passage numbers:"""


class Reranker:
    def __init__(self, llm_client: GeminiClient, model_name: str = "gemini-3.1-flash-lite"):
        self._llm = llm_client
        self._model_name = model_name

    def rerank(self, query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
        """Rerank candidates using Gemini as the relevance judge.

        Args:
            query: The user's question
            candidates: List of chunk dicts with 'text', 'source', 'chunk_id' keys
            top_n: Number of top chunks to return

        Returns:
            Reranked list of top_n chunks, or original candidates if reranking fails
        """
        if not candidates:
            return []

        # If we have fewer candidates than top_n, just return all of them
        if len(candidates) <= top_n:
            return candidates

        try:
            # Build passages with numbered labels
            passages = "\n".join(
                f"[{i+1}] {c['text']}" for i, c in enumerate(candidates)
            )

            # Build prompt
            prompt = _RERANK_PROMPT.format(
                max_num=len(candidates),
                top_n=top_n,
                query=query,
                passages=passages
            )

            # Call Gemini for reranking
            response = self._llm.generate(
                system_prompt="You are a precise relevance judge. Output only numbers.",
                user_message=prompt,
                temperature=0.1,  # Low temperature for consistent ranking
                model=self._model_name
            )

            # Parse response to extract numbers
            selected_indices = self._parse_rerank_response(response, len(candidates), top_n)

            if not selected_indices:
                logger.warning("Reranking failed to parse response, using original order")
                return candidates[:top_n]

            # Map back to original candidates (convert to 0-based indices)
            reranked = []
            for idx in selected_indices:
                if 0 <= idx < len(candidates):
                    reranked.append(candidates[idx])

            logger.info("Reranked %d candidates to top %d", len(candidates), len(reranked))
            return reranked

        except Exception as exc:
            logger.warning("Reranking failed with error: %s, using original order", exc)
            return candidates[:top_n]

    def _parse_rerank_response(self, response: str, total_candidates: int, top_n: int) -> list[int]:
        """Parse the reranking response to extract chunk indices.

        Args:
            response: Raw LLM response
            total_candidates: Total number of candidates (for validation)
            top_n: Expected number of results

        Returns:
            List of 0-based indices, or empty list if parsing fails
        """
        try:
            # Extract all numbers from the response
            numbers = re.findall(r'\d+', response.strip())
            
            if not numbers:
                logger.warning("No numbers found in reranking response")
                return []

            # Convert to integers and adjust to 0-based indices
            indices = []
            for num_str in numbers[:top_n]:  # Take only top_n numbers
                try:
                    # LLM returns 1-based indices, convert to 0-based
                    idx = int(num_str) - 1
                    if 0 <= idx < total_candidates:
                        indices.append(idx)
                    else:
                        logger.warning("Index %d out of range [0, %d), skipping", idx, total_candidates)
                except ValueError:
                    logger.warning("Could not parse number: %s", num_str)
                    continue

            # Remove duplicates while preserving order
            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)

            return unique_indices

        except Exception as exc:
            logger.warning("Failed to parse reranking response: %s", exc)
            return []


def get_reranker(llm_client: GeminiClient, model_name: str = "gemini-3.1-flash-lite") -> Reranker:
    return Reranker(llm_client, model_name)
