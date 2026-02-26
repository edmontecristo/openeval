"""
OpenEval Scorers — Evaluation metrics and scorers.

This package contains all scorer implementations:
- base: BaseScorer abstract class and FunctionScorer wrapper
- exact_match: ExactMatchScorer for string comparison
- contains: ContainsAnyScorer and ContainsAllScorer for keyword matching
- similarity: SimilarityScorer for embedding-based comparison
- llm_judge: LLMJudgeScorer for LLM-as-a-Judge evaluation
- faithfulness: FaithfulnessScorer for hallucination detection
- tool_correctness: ToolCorrectnessScorer for agent tool validation
"""

from openeval.scorers.base import BaseScorer, FunctionScorer
from openeval.scorers.exact_match import ExactMatchScorer
from openeval.scorers.contains import ContainsAnyScorer, ContainsAllScorer
from openeval.scorers.llm_judge import LLMJudgeScorer
from openeval.scorers.similarity import SimilarityScorer
from openeval.scorers.faithfulness import FaithfulnessScorer
from openeval.scorers.tool_correctness import ToolCorrectnessScorer

__all__ = [
    "BaseScorer",
    "FunctionScorer",
    "ExactMatchScorer",
    "ContainsAnyScorer",
    "ContainsAllScorer",
    "LLMJudgeScorer",
    "SimilarityScorer",
    "FaithfulnessScorer",
    "ToolCorrectnessScorer",
]
