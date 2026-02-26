"""
openeval/scorers/similarity.py — SimilarityScorer for embedding-based comparison.

Uses OpenAI embeddings to compute cosine similarity between actual and expected outputs.
Useful for semantic similarity evaluation when exact wording may differ.
"""

import numpy as np
from typing import List, Tuple, TYPE_CHECKING

from openeval.scorers.base import BaseScorer
from openeval.cost import calculate_cost

if TYPE_CHECKING:
    from openeval.test_case import TestCase
    from openeval.types import ScoreResult


class SimilarityScorer(BaseScorer):
    """
    Computes cosine similarity between embeddings of actual and expected output.

    This scorer uses OpenAI embeddings to convert text into vector representations,
    then calculates the cosine similarity between them. This is useful for
    evaluating semantic similarity when exact wording may differ but meaning is the same.

    Example:
        >>> scorer = SimilarityScorer(client=client, threshold=0.7)
        >>> result = scorer.score(test_case)
        >>> print(f"Similarity: {result.value:.2f}")
    """

    name = "Similarity"

    def __init__(self, client=None, model: str = "text-embedding-3-small", threshold: float = 0.7):
        """
        Initialize SimilarityScorer.

        Args:
            client: OpenAI client instance (for testing/injection)
            model: Embedding model to use (default: text-embedding-3-small)
            threshold: Minimum similarity to pass (default: 0.7)
        """
        self.client = client
        self.model = model
        self.threshold = threshold

    def score(self, test_case: "TestCase") -> "ScoreResult":
        """
        Score test case by computing cosine similarity of embeddings.

        Args:
            test_case: Test case with actual_output and expected_output

        Returns:
            ScoreResult with similarity value, pass/fail status, and cost tracking

        Raises:
            ValueError: If expected_output is None
        """
        from openeval.types import ScoreResult

        if test_case.expected_output is None:
            raise ValueError("SimilarityScorer requires test_case.expected_output")

        if not test_case.actual_output:
            return ScoreResult(
                name=self.name,
                value=0.0,
                reason="actual_output is empty",
                passed=False,
                threshold=self.threshold,
            )

        # Get embeddings for both texts
        expected_embedding, expected_tokens = self._get_embedding(test_case.expected_output)
        actual_embedding, actual_tokens = self._get_embedding(test_case.actual_output)

        # Calculate cosine similarity
        similarity = self._cosine_similarity(expected_embedding, actual_embedding)

        # Calculate total cost (embeddings only have prompt tokens)
        total_tokens = expected_tokens + actual_tokens
        cost = calculate_cost(self.model, total_tokens, 0)

        return ScoreResult(
            name=self.name,
            value=similarity,
            reason=f"Cosine similarity: {similarity:.4f}",
            passed=similarity >= self.threshold,
            threshold=self.threshold,
            token_usage=total_tokens,
            cost_usd=cost,
        )

    def _get_embedding(self, text: str) -> Tuple[List[float], int]:
        """
        Get embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Tuple of (embedding vector, token count)
        """
        response = self.client.embeddings.create(model=self.model, input=text)
        embedding = response.data[0].embedding
        tokens = response.usage.total_tokens if response.usage else 0
        return (embedding, tokens)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        a_arr, b_arr = np.array(a), np.array(b)
        return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr)))
