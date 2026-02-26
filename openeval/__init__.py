"""
OpenEval — Open-source LLM evaluation framework.

Evaluate, compare, and ship better AI applications with a scorer-based system,
CLI-first workflow, and CI/CD-native quality gates.
"""

__version__ = "0.1.1"

from openeval.test_case import TestCase
from openeval.types import ScoreResult, ExperimentResult, EvalResult
from openeval.scorers.base import BaseScorer, FunctionScorer
from openeval.tracing import trace, Span, TraceSession, get_current_traces
from openeval.eval import Eval
from openeval.experiment import compare_experiments

__all__ = [
    "TestCase",
    "ScoreResult",
    "ExperimentResult",
    "EvalResult",
    "BaseScorer",
    "FunctionScorer",
    "trace",
    "Span",
    "TraceSession",
    "get_current_traces",
    "Eval",
    "compare_experiments",
]
