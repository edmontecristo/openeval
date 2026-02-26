"""
openeval/eval.py — Main evaluation orchestrator.

The Eval() function is the primary API for running LLM evaluations.
It takes test data, runs tasks (LLM calls), applies scorers, and returns
aggregated results with statistics and cost tracking.
"""

import time
from typing import List, Callable, Optional, Dict, Any, Union

from openeval.types import ExperimentResult, EvalResult, ScoreResult
from openeval.test_case import TestCase
from openeval.dataset import Dataset


def Eval(
    name: str,
    data: Union[List[Dict], List[TestCase], Dataset],
    task: Optional[Callable[[str], str]] = None,
    scorers: Optional[List] = None,
) -> ExperimentResult:
    """
    Main evaluation function — runs tasks, applies scorers, returns results.

    This is the primary API for OpenEval. It orchestrates the entire evaluation
    process:
    1. Normalizes input data to TestCase objects
    2. Runs the task function (if provided) to generate actual_output
    3. Applies all scorers to each test case
    4. Aggregates results and computes summary statistics
    5. Tracks timing and cost

    Args:
        name: Experiment name/identifier (e.g., "chatbot-v2", "experiment-42")
        data: Input test data in one of these formats:
            - List of dictionaries (e.g., [{"input": "...", "expected_output": "..."}])
            - List of TestCase objects
            - Dataset object
        task: Optional callable that takes input string and returns output string.
              If None, test cases must already have actual_output populated.
              Function signature: task(input: str) -> str
        scorers: List of scorer instances to apply to each test case.
                 All scorers must implement BaseScorer interface with score() method.

    Returns:
        ExperimentResult containing:
            - results: List of EvalResult (one per test case)
            - summary: Dict with mean, min, max, pass_rate per scorer
            - duration_ms: Total evaluation time in milliseconds
            - total_cost_usd: Sum of all LLM API costs

    Raises:
        ValueError: If data is empty or scorers list is empty

    Example:
        >>> from openeval import Eval
        >>> from openeval.scorers.exact_match import ExactMatchScorer
        >>>
        >>> result = Eval(
        ...     name="my-experiment",
        ...     data=[{"input": "hello", "expected_output": "world"}],
        ...     task=lambda input: "world",
        ...     scorers=[ExactMatchScorer()],
        ... )
        >>> print(result.summary["ExactMatch"]["mean"])
        1.0
    """
    # Validate inputs
    if not data:
        raise ValueError("data cannot be empty")

    if isinstance(data, list) and len(data) == 0:
        raise ValueError("data cannot be empty")

    if not scorers:
        raise ValueError("scorers cannot be empty")

    if isinstance(data, list) and len(scorers) == 0:
        raise ValueError("scorers cannot be empty")

    start_time = time.time()
    results: List[EvalResult] = []
    total_cost = 0.0

    # Normalize data → list of TestCase
    test_cases = _normalize_data(data)

    # Evaluate each test case
    for tc in test_cases:
        # Run task if provided (get actual_output)
        error = None
        if task is not None:
            try:
                tc.actual_output = task(tc.input)
            except Exception as e:
                error = str(e)

        # Run all scorers
        scores: Dict[str, ScoreResult] = {}
        if error is None:
            for scorer in scorers:
                score_result = scorer.score(tc)
                scores[score_result.name] = score_result
                if score_result.cost_usd:
                    total_cost += score_result.cost_usd

        # Build EvalResult
        results.append(
            EvalResult(
                test_case_id=tc.id,
                input=tc.input,
                actual_output=tc.actual_output or "",
                expected_output=tc.expected_output,
                scores=scores,
                error=error,
            )
        )

    # Compute metrics
    duration = (time.time() - start_time) * 1000  # Convert to milliseconds
    summary = _compute_summary(results, scorers)

    return ExperimentResult(
        name=name,
        results=results,
        summary=summary,
        duration_ms=duration,
        total_cost_usd=total_cost,
    )


def _normalize_data(
    data: Union[List[Dict], List[TestCase], Dataset],
) -> List[TestCase]:
    """
    Convert various data formats to a list of TestCase objects.

    Supports:
    - Dataset object: extracts items and converts to TestCase
    - List of TestCase: returns as-is
    - List of dict: converts each dict to TestCase using from_dict()

    Args:
        data: Input data in any supported format

    Returns:
        List of TestCase objects

    Raises:
        ValueError: If data format is unsupported
    """
    if isinstance(data, Dataset):
        # Convert Dataset items to TestCase objects
        return [
            TestCase.from_dict(item) if isinstance(item, dict) else item
            for item in data
        ]

    if isinstance(data, list):
        result = []
        for item in data:
            if isinstance(item, TestCase):
                result.append(item)
            elif isinstance(item, dict):
                result.append(TestCase.from_dict(item))
            else:
                raise ValueError(
                    f"Unsupported data type in list: {type(item)}. "
                    "Expected dict or TestCase"
                )
        return result

    raise ValueError(
        f"Unsupported data format: {type(data)}. "
        "Expected list[dict], list[TestCase], or Dataset"
    )


def _compute_summary(results: List[EvalResult], scorers: List) -> Dict:
    """
    Compute aggregate statistics (mean, min, max, pass_rate) per scorer.

    For each scorer, calculates:
    - mean: Average score across all test cases
    - min: Minimum score
    - max: Maximum score
    - pass_rate: Fraction of scores >= scorer threshold

    Args:
        results: List of EvalResult from evaluation
        scorers: List of scorer instances used in evaluation

    Returns:
        Dictionary mapping scorer names to their statistics
        Example: {"ExactMatch": {"mean": 0.85, "min": 0.0, "max": 1.0, "pass_rate": 0.9}}
    """
    summary = {}

    for scorer in scorers:
        name = scorer.name

        # Extract all scores for this scorer
        scores = [
            r.scores[name].value
            for r in results
            if name in r.scores and r.error is None
        ]

        if scores:
            threshold = getattr(scorer, "threshold", 0.0)
            passed_count = sum(1 for score in scores if score >= threshold)

            summary[name] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "pass_rate": passed_count / len(scores),
            }

    return summary
