"""
openeval/scorers/exact_match.py — ExactMatchScorer for string comparison.

This scorer performs exact string matching between actual_output and expected_output,
with optional case-insensitive comparison.
"""

from openeval.scorers.base import BaseScorer
from openeval.types import ScoreResult
from openeval.test_case import TestCase
from typing import Optional


class ExactMatchScorer(BaseScorer):
    """
    String equality scorer.

    Compares actual_output with expected_output for exact match.
    Returns 1.0 if strings match, 0.0 otherwise.

    Attributes:
        name: Scorer name ("ExactMatch")
        case_sensitive: Whether comparison should be case-sensitive (default: True)
        threshold: Minimum score to pass (default: 0.5)
    """

    name = "ExactMatch"

    def __init__(self, case_sensitive: bool = True, threshold: float = 0.5):
        """
        Initialize ExactMatchScorer.

        Args:
            case_sensitive: If True, comparison respects case. If False, case-insensitive.
            threshold: Minimum score to pass (default: 0.5)
        """
        self.case_sensitive = case_sensitive
        self.threshold = threshold

    def score(self, test_case: TestCase) -> ScoreResult:
        """
        Score a test case by exact string comparison.

        Args:
            test_case: The test case to score. Must have expected_output.

        Returns:
            ScoreResult with value 1.0 if match, 0.0 otherwise

        Raises:
            ValueError: If test_case.expected_output is None
        """
        if test_case.expected_output is None:
            raise ValueError("ExactMatchScorer requires test_case.expected_output")

        if test_case.actual_output is None:
            return ScoreResult(
                name=self.name,
                value=0.0,
                reason="actual_output is None",
                passed=False,
                threshold=self.threshold,
            )

        if self.case_sensitive:
            matches = test_case.actual_output == test_case.expected_output
        else:
            matches = test_case.actual_output.lower() == test_case.expected_output.lower()

        value = 1.0 if matches else 0.0
        reason = (
            "Outputs match exactly"
            if matches
            else f"Expected '{test_case.expected_output}', got '{test_case.actual_output}'"
        )

        return ScoreResult(
            name=self.name,
            value=value,
            reason=reason,
            passed=value >= self.threshold,
            threshold=self.threshold,
        )
