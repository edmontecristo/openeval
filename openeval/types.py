"""
openeval/types.py — Core data models for OpenEval framework.

This module defines the foundational types used throughout the evaluation system:
- ScoreResult: Individual scorer output with value, reasoning, and metadata
- EvalResult: Results for a single test case across all scorers
- ExperimentResult: Complete evaluation results with summary statistics
"""

from pydantic import BaseModel
from typing import Optional, Dict, List


class ScoreResult(BaseModel):
    """
    Result from scoring a single test case.

    Attributes:
        name: Scorer name (e.g., "ExactMatch", "LLMJudge")
        value: Score from 0.0 to 1.0
        reason: Optional explanation for the score
        passed: Whether score meets threshold (if threshold provided)
        threshold: Minimum score to pass (optional)
        token_usage: Number of tokens used (for LLM-based scorers)
        cost_usd: Cost of the LLM call in USD (for LLM-based scorers)
    """
    name: str
    value: float
    reason: Optional[str] = None
    passed: Optional[bool] = None
    threshold: Optional[float] = None
    token_usage: Optional[int] = None
    cost_usd: Optional[float] = None


class EvalResult(BaseModel):
    """
    Evaluation results for a single test case.

    Aggregates scores from all scorers for one test case,
    along with input/output data and any errors.

    Attributes:
        test_case_id: Unique identifier for the test case
        input: The original input prompt
        actual_output: The actual output from the system being evaluated
        expected_output: The ground truth expected output (if available)
        scores: Dictionary mapping scorer names to ScoreResult objects
        error: Error message if the task function raised an exception
    """
    test_case_id: str
    input: str
    actual_output: str
    expected_output: Optional[str] = None
    scores: Dict[str, ScoreResult] = {}
    error: Optional[str] = None


class ExperimentResult(BaseModel):
    """
    Complete results from an evaluation experiment.

    Contains all test case results, aggregated summary statistics,
    timing information, and cost tracking.

    Attributes:
        name: Name/identifier for this experiment
        results: List of individual test case results
        summary: Aggregated statistics (e.g., {"ExactMatch": {"mean": 0.85, "pass_rate": 0.9}})
        duration_ms: Total experiment duration in milliseconds
        total_cost_usd: Total cost of all LLM API calls in USD
    """
    name: str
    results: List[EvalResult] = []
    summary: Dict = {}
    duration_ms: Optional[float] = None
    total_cost_usd: Optional[float] = None
