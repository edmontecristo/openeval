"""
test_cost_tracking.py — Tests for token usage and cost calculation.

Every LLM call costs money. Users need to know:
- How many tokens were used
- How much each eval cost in USD
- Cost breakdown by scorer
"""

import pytest


class TestTokenCounter:
    """Track token usage across all LLM calls."""

    def test_count_from_openai_response(self):
        """Should extract tokens from OpenAI-style response."""
        from openeval.cost import TokenUsage

        usage = TokenUsage.from_openai_usage(
            prompt_tokens=150,
            completion_tokens=50,
            model="gpt-4o-mini",
        )
        assert usage.prompt_tokens == 150
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 200

    def test_sum_token_usages(self):
        """Should sum up multiple token usages."""
        from openeval.cost import TokenUsage

        u1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        u2 = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300)
        total = u1 + u2
        assert total.prompt_tokens == 300
        assert total.completion_tokens == 150
        assert total.total_tokens == 450


class TestCostCalculator:
    """Calculate USD cost from token usage + model pricing."""

    def test_gpt4o_mini_cost(self):
        """GPT-4o-mini: $0.15/1M input, $0.60/1M output."""
        from openeval.cost import calculate_cost

        cost = calculate_cost(
            model="gpt-4o-mini",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        expected = (1000 * 0.15 / 1_000_000) + (500 * 0.60 / 1_000_000)
        assert abs(cost - expected) < 0.0001

    def test_gpt4o_cost(self):
        """GPT-4o: $2.50/1M input, $10.00/1M output."""
        from openeval.cost import calculate_cost

        cost = calculate_cost(
            model="gpt-4o",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        expected = (1000 * 2.50 / 1_000_000) + (500 * 10.00 / 1_000_000)
        assert abs(cost - expected) < 0.001

    def test_unknown_model_returns_zero(self):
        """Unknown model should return 0 cost (not crash)."""
        from openeval.cost import calculate_cost

        cost = calculate_cost(
            model="unknown-model-xyz",
            prompt_tokens=1000,
            completion_tokens=500,
        )
        assert cost == 0.0

    def test_total_eval_cost(self):
        """Sum cost across entire evaluation run."""
        from openeval.cost import CostTracker

        tracker = CostTracker()
        tracker.add(model="gpt-4o-mini", prompt_tokens=100, completion_tokens=50)
        tracker.add(model="gpt-4o-mini", prompt_tokens=200, completion_tokens=100)

        assert tracker.total_cost > 0
        assert tracker.total_tokens == 450
        assert len(tracker.breakdown) == 2


class TestCostBreakdown:
    """Detailed cost breakdown by component."""

    def test_breakdown_by_scorer(self):
        """Show cost per scorer (e.g., LLM-Judge vs Similarity)."""
        from openeval.cost import CostTracker

        tracker = CostTracker()
        tracker.add(
            model="gpt-4o-mini",
            prompt_tokens=1000,
            completion_tokens=500,
            label="LLMJudge",
        )
        tracker.add(
            model="text-embedding-3-small",
            prompt_tokens=100,
            completion_tokens=0,
            label="Similarity",
        )

        by_label = tracker.breakdown_by_label()
        assert "LLMJudge" in by_label
        assert "Similarity" in by_label
        assert by_label["LLMJudge"]["total_tokens"] == 1500
