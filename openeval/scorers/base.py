"""
openeval/scorers/base.py — Base scorer interface and function wrapper.

All scorers in OpenEval implement the BaseScorer interface, which provides
a consistent API for evaluating test cases.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from openeval.test_case import TestCase
    from openeval.types import ScoreResult


class BaseScorer(ABC):
    """
    Abstract base class for all scorers.

    All scorer implementations must inherit from this class and implement
    the score() method.
    """

    name: str

    @abstractmethod
    def score(self, test_case: "TestCase") -> "ScoreResult":
        """
        Score a test case and return a ScoreResult.

        Args:
            test_case: The test case to score

        Returns:
            ScoreResult containing the score value and optional metadata
        """
        pass


class FunctionScorer(BaseScorer):
    """
    Wrapper that converts a plain function into a scorer.

    This allows users to create custom scorers using simple functions
    without needing to create a full class.

    Example:
        def my_scorer(tc: TestCase) -> float:
            return 1.0 if "hello" in tc.actual_output.lower() else 0.0

        scorer = FunctionScorer(name="MyScorer", fn=my_scorer, threshold=0.5)
    """

    def __init__(self, name: str, fn: Callable[["TestCase"], float], threshold: float = 0.5):
        """
        Initialize a FunctionScorer.

        Args:
            name: Name for this scorer
            fn: Function that takes a TestCase and returns a float score
            threshold: Minimum score to pass (default: 0.5)
        """
        self.name = name
        self.fn = fn
        self.threshold = threshold

    def score(self, test_case: "TestCase") -> "ScoreResult":
        """
        Score a test case using the wrapped function.

        Args:
            test_case: The test case to score

        Returns:
            ScoreResult with the score value and pass/fail status
        """
        # Import here to avoid circular dependency
        from openeval.types import ScoreResult

        # Call the function to get the raw score
        raw_value = self.fn(test_case)

        # Create and return the ScoreResult
        return ScoreResult(
            name=self.name,
            value=raw_value,
            passed=raw_value >= self.threshold,
            threshold=self.threshold,
        )
