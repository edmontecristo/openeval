"""
openeval/scorers/contains.py — Keyword-based scorers.

Contains scorers check if actual_output contains specific keywords:
- ContainsAnyScorer: Returns 1.0 if ANY keyword is found
- ContainsAllScorer: Returns fraction of keywords found (0.0 to 1.0)
"""

from openeval.scorers.base import BaseScorer
from openeval.types import ScoreResult
from openeval.test_case import TestCase
from typing import List


class ContainsAnyScorer(BaseScorer):
    """
    Checks if ANY keyword is present in actual_output.

    Returns 1.0 if at least one keyword is found, 0.0 otherwise.

    Attributes:
        name: Scorer name ("ContainsAny")
        keywords: List of keywords to search for
        threshold: Minimum score to pass (default: 0.5)
    """

    name = "ContainsAny"

    def __init__(self, keywords: List[str], threshold: float = 0.5):
        """
        Initialize ContainsAnyScorer.

        Args:
            keywords: List of keyword strings to search for
            threshold: Minimum score to pass (default: 0.5)
        """
        self.keywords = keywords
        self.threshold = threshold

    def score(self, test_case: TestCase) -> ScoreResult:
        """
        Score a test case by checking if any keyword is present.

        Args:
            test_case: The test case to score

        Returns:
            ScoreResult with value 1.0 if any keyword found, 0.0 otherwise
        """
        if not test_case.actual_output:
            return ScoreResult(
                name=self.name,
                value=0.0,
                reason="actual_output is empty",
                passed=False,
                threshold=self.threshold,
            )

        found = []
        for kw in self.keywords:
            if kw.lower() in test_case.actual_output.lower():
                found.append(kw)

        value = 1.0 if found else 0.0
        reason = (
            f"Found keywords: {', '.join(found)}"
            if found
            else f"None of {self.keywords} found in output"
        )

        return ScoreResult(
            name=self.name,
            value=value,
            reason=reason,
            passed=value >= self.threshold,
            threshold=self.threshold,
        )


class ContainsAllScorer(BaseScorer):
    """
    Checks what fraction of keywords are present in actual_output.

    Returns the proportion of keywords found (from 0.0 to 1.0).
    For example, if 2 out of 3 keywords are found, returns 0.667.

    Attributes:
        name: Scorer name ("ContainsAll")
        keywords: List of keywords to search for
        threshold: Minimum score to pass (default: 0.5)
    """

    name = "ContainsAll"

    def __init__(self, keywords: List[str], threshold: float = 0.5):
        """
        Initialize ContainsAllScorer.

        Args:
            keywords: List of keyword strings to search for
            threshold: Minimum score to pass (default: 0.5)
        """
        self.keywords = keywords
        self.threshold = threshold

    def score(self, test_case: TestCase) -> ScoreResult:
        """
        Score a test case by checking what fraction of keywords are present.

        Args:
            test_case: The test case to score

        Returns:
            ScoreResult with value equal to fraction of keywords found
        """
        if not self.keywords:
            return ScoreResult(
                name=self.name,
                value=1.0,
                reason="No keywords specified",
                passed=True,
                threshold=self.threshold,
            )

        if not test_case.actual_output:
            return ScoreResult(
                name=self.name,
                value=0.0,
                reason="actual_output is empty",
                passed=False,
                threshold=self.threshold,
            )

        found = []
        for kw in self.keywords:
            if kw.lower() in test_case.actual_output.lower():
                found.append(kw)

        value = len(found) / len(self.keywords)
        reason = f"Found {len(found)}/{len(self.keywords)} keywords: {', '.join(found)}"

        return ScoreResult(
            name=self.name,
            value=value,
            reason=reason,
            passed=value >= self.threshold,
            threshold=self.threshold,
        )
