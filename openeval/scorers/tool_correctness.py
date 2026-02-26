"""
openeval/scorers/tool_correctness.py — ToolCorrectnessScorer for agent validation.

This scorer validates that an agent called the expected tools,
with optional order checking.
"""

from openeval.scorers.base import BaseScorer
from openeval.types import ScoreResult
from openeval.test_case import TestCase
from typing import Optional


class ToolCorrectnessScorer(BaseScorer):
    """
    Checks if agent called the expected tools.

    Validates agent tool calls by comparing tools_called against expected_tools.
    Can check both presence and order of tool calls.

    Attributes:
        name: Scorer name ("ToolCorrectness")
        check_order: If True, validates both presence and order of tools
        threshold: Minimum score to pass (default: 0.5)
    """

    name = "ToolCorrectness"

    def __init__(self, check_order: bool = False, threshold: float = 0.5):
        """
        Initialize ToolCorrectnessScorer.

        Args:
            check_order: If True, checks that tools are called in the correct order.
                        Extra tools are allowed but all expected tools must be present in order.
            threshold: Minimum score to pass (default: 0.5)
        """
        self.check_order = check_order
        self.threshold = threshold

    def score(self, test_case: TestCase) -> ScoreResult:
        """
        Score a test case by validating tool calls.

        Args:
            test_case: The test case to score. Must have expected_tools.

        Returns:
            ScoreResult with value based on how many expected tools were called

        Raises:
            ValueError: If test_case.expected_tools is None
        """
        if test_case.expected_tools is None:
            raise ValueError("ToolCorrectnessScorer requires test_case.expected_tools")

        if test_case.tools_called is None:
            return ScoreResult(
                name=self.name,
                value=0.0,
                reason="tools_called is None",
                passed=False,
                threshold=self.threshold,
            )

        expected = test_case.expected_tools
        actual = test_case.tools_called

        if self.check_order:
            # Check both presence and order
            # Extra tools are OK, but all expected must be present in order
            actual_iter = iter(actual)
            matched = 0
            for exp_tool in expected:
                for act_tool in actual_iter:
                    if act_tool == exp_tool:
                        matched += 1
                        break

            value = matched / len(expected) if expected else 1.0
            reason = f"Matched {matched}/{len(expected)} tools in order"
        else:
            # Set-based comparison: check if all expected tools are present
            actual_set = set(actual)
            matched = sum(1 for tool in expected if tool in actual_set)
            value = matched / len(expected) if expected else 1.0
            reason = f"Matched {matched}/{len(expected)} expected tools (extra tools: {len(actual) - matched})"

        return ScoreResult(
            name=self.name,
            value=value,
            reason=reason,
            passed=value >= self.threshold,
            threshold=self.threshold,
        )
