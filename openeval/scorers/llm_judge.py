"""
openeval/scorers/llm_judge.py — LLMJudgeScorer for LLM-as-a-Judge evaluation.

Uses GPT-4 to evaluate the quality of an LLM output based on custom criteria.
This is the G-Eval pattern: one LLM evaluates another's output.
"""

import json
from typing import TYPE_CHECKING

from openeval.scorers.base import BaseScorer
from openeval.cost import calculate_cost

if TYPE_CHECKING:
    from openeval.test_case import TestCase
    from openeval.types import ScoreResult


class LLMJudgeScorer(BaseScorer):
    """
    Uses LLM to score output quality based on custom criteria.

    This scorer implements the G-Eval pattern where one LLM evaluates another's output.
    You provide custom evaluation criteria, and the LLM scores the output from 0.0 to 1.0.

    Example:
        >>> scorer = LLMJudgeScorer(
        ...     name="Correctness",
        ...     criteria="Is the output factually correct?",
        ...     client=client,
        ...     threshold=0.7
        ... )
        >>> result = scorer.score(test_case)
        >>> print(f"Score: {result.value:.2f} - {result.reason}")
    """

    def __init__(
        self,
        name: str,
        criteria: str,
        client=None,
        model: str = "gpt-4o-mini",
        threshold: float = 0.5,
    ):
        """
        Initialize LLMJudgeScorer.

        Args:
            name: Name for this scorer (e.g., "Correctness", "Politeness")
            criteria: Custom evaluation criteria for the judge LLM
            client: OpenAI client instance (for testing/injection)
            model: Model to use for judging (default: gpt-4o-mini)
            threshold: Minimum score to pass (default: 0.5)
        """
        self.name = name
        self.criteria = criteria
        self.client = client
        self.model = model
        self.threshold = threshold

    def score(self, test_case: "TestCase") -> "ScoreResult":
        """
        Score test case using LLM-as-a-Judge.

        Args:
            test_case: Test case with input, actual_output, and optionally expected_output

        Returns:
            ScoreResult with LLM judgment, reasoning, pass/fail status, and cost tracking
        """
        from openeval.types import ScoreResult

        if not test_case.actual_output:
            return ScoreResult(
                name=self.name,
                value=0.0,
                reason="No actual output to judge",
                passed=False,
                threshold=self.threshold,
            )

        # Build the judge prompt
        prompt = self._build_prompt(test_case)

        # Call the LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        content = response.choices[0].message.content or ""

        # Parse JSON response — handle malformed responses gracefully
        score_val, reason = self._parse_response(content)

        # Clamp score to [0.0, 1.0]
        score_val = max(0.0, min(1.0, score_val))

        # Track token usage and cost
        tokens = response.usage.total_tokens if response.usage else 0
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
            token_usage=tokens,
            cost_usd=cost,
        )

    def _build_prompt(self, test_case: "TestCase") -> str:
        """
        Build the judge prompt for the LLM.

        Args:
            test_case: Test case to evaluate

        Returns:
            Prompt string for the judge LLM
        """
        expected_str = test_case.expected_output or "None provided"

        prompt = f"""You are an evaluation judge. Score the output from 0.0 to 1.0.

Criteria: {self.criteria}
Input: {test_case.input}
Expected output: {expected_str}
Actual output: {test_case.actual_output}

Return ONLY valid JSON: {{"score": <float 0.0-1.0>, "reason": "<explanation>"}}"""

        return prompt

    def _parse_response(self, content: str) -> tuple[float, str]:
        """
        Parse LLM JSON response.

        Handles raw JSON and markdown-fenced JSON (```json...```).
        Falls back to regex extraction if JSON parsing fails.

        Args:
            content: Raw response content from LLM

        Returns:
            Tuple of (score, reason). Returns (0.0, error_message) on parse failure.
        """
        import re
        # Strip markdown code fences if present (common with Ollama models)
        cleaned = content.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ``` etc)
            cleaned = re.sub(r'^```[a-zA-Z]*\n?', '', cleaned)
            # Remove closing fence
            cleaned = re.sub(r'```\s*$', '', cleaned).strip()

        try:
            parsed = json.loads(cleaned)
            score_val = float(parsed.get("score", 0.0))
            reason = parsed.get("reason", "")
            return (score_val, reason)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            pass

        # Last resort: regex extract score and reason from anywhere in the text
        score_match = re.search(r'"score"\s*:\s*([0-9.]+)', content)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', content)
        if score_match:
            try:
                score_val = float(score_match.group(1))
                reason = reason_match.group(1) if reason_match else "Extracted via regex"
                return (score_val, reason)
            except ValueError:
                pass

        return (0.0, f"Failed to parse judge response: {content[:200]}")
