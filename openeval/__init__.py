"""
OpenEval — Open-source LLM evaluation framework.

A Braintrust.dev alternative designed for CLI-first, developer-friendly usage.
"""

from openeval.test_case import TestCase
from openeval.types import ScoreResult, ExperimentResult, EvalResult
from openeval.scorers.base import BaseScorer, FunctionScorer
from openeval.tracing import trace, Span, TraceSession, get_current_traces

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
]
