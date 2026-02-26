"""
openeval/experiment.py — Experiment comparison and regression detection.

Provides tools to compare multiple experiment runs and detect performance
regressions or improvements across different model versions or configurations.
"""

from openeval.types import ExperimentResult
from typing import Dict, List


def compare_experiments(
    baseline: ExperimentResult,
    candidate: ExperimentResult,
) -> Dict[str, Dict]:
    """
    Compare two experiment runs to detect performance changes.

    Computes the delta (candidate - baseline) for each scorer's mean score,
    allowing you to detect improvements or regressions between different
    model versions, prompts, or configurations.

    Args:
        baseline: Baseline experiment results (e.g., current production model)
        candidate: New experiment results to compare against baseline
                   (e.g., new model version or prompt template)

    Returns:
        Dictionary mapping scorer names to comparison dictionaries:
        {
            "scorer_name": {
                "baseline": float,      # baseline mean score
                "candidate": float,     # candidate mean score
                "delta": float,         # candidate - baseline
                "improved": bool,       # True if delta > 0
            }
        }

    Example:
        >>> from openeval import Eval
        >>> from openeval.experiment import compare_experiments
        >>> from openeval.scorers.exact_match import ExactMatchScorer
        >>>
        >>> # Run baseline experiment
        >>> baseline = Eval(
        ...     name="baseline",
        ...     data=[{"input": "a", "expected_output": "a"}],
        ...     task=lambda input: "wrong",
        ...     scorers=[ExactMatchScorer()],
        ... )
        >>>
        >>> # Run candidate experiment (improved version)
        >>> candidate = Eval(
        ...     name="candidate",
        ...     data=[{"input": "a", "expected_output": "a"}],
        ...     task=lambda input: input,
        ...     scorers=[ExactMatchScorer()],
        ... )
        >>>
        >>> # Compare
        >>> comparison = compare_experiments(baseline, candidate)
        >>> print(comparison["ExactMatch"]["delta"])  # e.g., 1.0
        >>> print(comparison["ExactMatch"]["improved"])  # True
    """
    comparison = {}

    # Iterate through all scorers in the candidate experiment
    for scorer_name in candidate.summary:
        baseline_stats = baseline.summary.get(scorer_name, {})
        candidate_stats = candidate.summary[scorer_name]

        base_mean = baseline_stats.get("mean", 0.0)
        cand_mean = candidate_stats.get("mean", 0.0)
        delta = cand_mean - base_mean

        comparison[scorer_name] = {
            "baseline": base_mean,
            "candidate": cand_mean,
            "delta": delta,
            "improved": delta > 0,
        }

    return comparison
