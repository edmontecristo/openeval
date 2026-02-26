"""
cost.py — LLM usage cost tracking.

Tracks token usage and calculates USD costs for LLM API calls.
Supports OpenAI, Claude, and embedding models with per-model pricing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Model pricing (USD per 1M tokens)
# Source: OpenAI and Anthropic pricing as of 2024
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
}


@dataclass
class TokenUsage:
    """
    Token usage statistics for a single API call.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt
        completion_tokens: Number of tokens in the generated response
        total_tokens: Total tokens used (prompt + completion)
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_openai_usage(
        cls, prompt_tokens: int, completion_tokens: int, model: str = None
    ):
        """
        Create TokenUsage from OpenAI-style response format.

        Args:
            prompt_tokens: Input tokens from API response
            completion_tokens: Output tokens from API response
            model: Model name (optional, not used in calculation but kept for API compatibility)

        Returns:
            TokenUsage instance with calculated total
        """
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def __add__(self, other):
        """
        Sum two TokenUsage instances.

        Useful for aggregating usage across multiple API calls.

        Example:
            >>> total = usage1 + usage2
            >>> total.total_tokens
            450
        """
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate USD cost for an LLM API call.

    Args:
        model: Model name (must exist in MODEL_PRICING)
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Cost in USD. Returns 0.0 for unknown models (never crashes).

    Example:
        >>> cost = calculate_cost("gpt-4o-mini", 1000, 500)
        >>> print(f"${cost:.6f}")
        $0.000450
    """
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        return 0.0

    input_cost = (prompt_tokens * pricing["input"]) / 1_000_000
    output_cost = (completion_tokens * pricing["output"]) / 1_000_000
    return input_cost + output_cost


@dataclass
class CostEntry:
    """
    Single cost entry for one API call.

    Attributes:
        model: Model name used
        prompt_tokens: Input tokens
        completion_tokens: Output tokens
        cost: Calculated USD cost
        label: Optional label (e.g., "LLMJudge", "Similarity") for grouping
    """

    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    label: Optional[str] = None


class CostTracker:
    """
    Track and aggregate LLM costs across an evaluation run.

    Features:
    - Add individual API call costs
    - Get total cost and token count
    - Breakdown by model or by label (scorer type)

    Example:
        >>> tracker = CostTracker()
        >>> tracker.add("gpt-4o-mini", 1000, 500, label="LLMJudge")
        >>> tracker.add("text-embedding-3-small", 100, 0, label="Similarity")
        >>> print(f"Total: ${tracker.total_cost:.6f}")
        >>> print(tracker.breakdown_by_label())
    """

    def __init__(self):
        """Initialize empty cost tracker."""
        self._entries: List[CostEntry] = []

    def add(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        label: Optional[str] = None,
    ):
        """
        Add a cost entry for an API call.

        Args:
            model: Model name
            prompt_tokens: Input tokens
            completion_tokens: Output tokens
            label: Optional label for grouping (e.g., scorer name)
        """
        cost = calculate_cost(model, prompt_tokens, completion_tokens)
        self._entries.append(
            CostEntry(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=cost,
                label=label,
            )
        )

    @property
    def total_cost(self) -> float:
        """Total USD cost across all tracked API calls."""
        return sum(entry.cost for entry in self._entries)

    @property
    def total_tokens(self) -> int:
        """Total tokens used across all tracked API calls."""
        return sum(
            entry.prompt_tokens + entry.completion_tokens for entry in self._entries
        )

    @property
    def breakdown(self) -> List[dict]:
        """
        Detailed breakdown of all cost entries.

        Returns:
            List of dicts, one per entry, with full details
        """
        return [
            {
                "model": entry.model,
                "prompt_tokens": entry.prompt_tokens,
                "completion_tokens": entry.completion_tokens,
                "total_tokens": entry.prompt_tokens + entry.completion_tokens,
                "cost_usd": entry.cost,
                "label": entry.label,
            }
            for entry in self._entries
        ]

    def breakdown_by_label(self) -> Dict[str, dict]:
        """
        Aggregate costs by label (e.g., by scorer type).

        Useful for understanding which scorers are most expensive.

        Returns:
            Dict mapping label name to aggregated stats:
            {
                "label_name": {
                    "prompt_tokens": int,
                    "completion_tokens": int,
                    "total_tokens": int,
                    "cost_usd": float
                }
            }

        Example:
            >>> tracker = CostTracker()
            >>> tracker.add("gpt-4o-mini", 1000, 500, label="LLMJudge")
            >>> tracker.add("gpt-4o-mini", 500, 200, label="LLMJudge")
            >>> tracker.add("text-embedding-3-small", 100, 0, label="Similarity")
            >>> by_label = tracker.breakdown_by_label()
            >>> by_label["LLMJudge"]["total_tokens"]
            2200
        """
        result: Dict[str, dict] = {}

        for entry in self._entries:
            label = entry.label or "default"

            if label not in result:
                result[label] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                }

            result[label]["prompt_tokens"] += entry.prompt_tokens
            result[label]["completion_tokens"] += entry.completion_tokens
            result[label]["total_tokens"] += (
                entry.prompt_tokens + entry.completion_tokens
            )
            result[label]["cost_usd"] += entry.cost

        return result
