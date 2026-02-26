"""
openeval/scorers/faithfulness.py — FaithfulnessScorer for hallucination detection.

Checks if LLM output is faithful to provided context (RAG hallucination detection).
Useful for evaluating retrieval-augmented generation (RAG) systems.
"""

import json
from typing import TYPE_CHECKING

from openeval.scorers.base import BaseScorer
from openeval.cost import calculate_cost

if TYPE_CHECKING:
    from openeval.test_case import TestCase
    from openeval.types import ScoreResult


class FaithfulnessScorer(BaseScorer):
    """
    Checks if output is faithful to context (RAG hallucination detection).

    This scorer evaluates whether an LLM's output is grounded in the provided context.
    It detects hallucinations by checking if all claims in the output are supported
    by the context. This is critical for RAG systems where the LLM should only use
    information from retrieved documents.

    Example:
        >>> scorer = FaithfulnessScorer(client=client, threshold=0.7)
        >>> result = scorer.score(rag_test_case)
        >>> print(f"Faithfulness: {result.value:.2f}")
        >>> if not result.passed:
        ...     print("Potential hallucination detected!")
    """

    name = "Faithfulness"

    def __init__(
        self, client=None, model: str = "gpt-4o-mini", threshold: float = 0.7
    ):
        """
        Initialize FaithfulnessScorer.

        Args:
            client: OpenAI client instance (for testing/injection)
            model: Model to use for evaluation (default: gpt-4o-mini)
            threshold: Minimum faithfulness score to pass (default: 0.7)
        """
        self.client = client
        self.model = model
        self.threshold = threshold

    def score(self, test_case: "TestCase") -> "ScoreResult":
        """
        Score test case by checking faithfulness to context.

        Args:
            test_case: Test case with context and actual_output

        Returns:
            ScoreResult with faithfulness score, reasoning, pass/fail status, and cost tracking

        Raises:
            ValueError: If context is None
        """
        from openeval.types import ScoreResult

        if test_case.context is None:
            raise ValueError("FaithfulnessScorer requires test_case.context")

        if not test_case.actual_output:
            return ScoreResult(
                name=self.name,
                value=0.0,
                reason="actual_output is empty",
                passed=False,
                threshold=self.threshold,
            )

        # Join context into a single string
        context_str = "\n".join(test_case.context)

        # Build the faithfulness prompt
        prompt = self._build_prompt(context_str, test_case.actual_output)

        # Call the LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content or ""

        # Parse JSON response
        score_val, reason = self._parse_response(content)

        # Clamp score to [0.0, 1.0]
        score_val = max(0.0, min(1.0, score_val))

        # Calculate cost
        cost = calculate_cost(
            self.model,
            response.usage.prompt_tokens if response.usage else 0,
            response.usage.completion_tokens if response.usage else 0,
        )

        return ScoreResult(
            name=self.name,
            value=score_val,
            reason=reason,
            passed=score_val >= self.threshold,
            threshold=self.threshold,
            token_usage=response.usage.total_tokens if response.usage else 0,
            cost_usd=cost,
        )

    def _build_prompt(self, context: str, output: str) -> str:
        """
        Build the faithfulness evaluation prompt.

        Args:
            context: Retrieved context documents
            output: LLM output to evaluate

        Returns:
            Prompt string for the LLM
        """
        prompt = f"""You are a faithfulness evaluator.
Given the context and the output, score from 0.0 to 1.0
how well the output is supported by ONLY the information in the context.
Score 0.0 if the output contains fabricated information.
Score 1.0 if every claim in the output is directly supported by context.

Context:
{context}

Output:
{output}

Return ONLY valid JSON: {{"score": <float>, "reason": "<explanation>"}}"""

        return prompt

    def _parse_response(self, content: str) -> tuple[float, str]:
        """
        Parse LLM JSON response.

        Args:
            content: Raw response content from LLM

        Returns:
            Tuple of (score, reason). Returns (0.0, error_message) on parse failure.
        """
        try:
            parsed = json.loads(content)
            score_val = float(parsed.get("score", 0.0))
            reason = parsed.get("reason", "")
            return (score_val, reason)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            # Handle malformed JSON gracefully
            return (0.0, f"Failed to parse response: {content[:100]}")
