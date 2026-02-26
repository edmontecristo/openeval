"""
test_scorers.py — Tests for all scoring/metric functions.

This is the HEART of the evaluation engine. Scorers determine
whether LLM output is good or bad. Must be bulletproof.

Scorer types:
  1. ExactMatch — deterministic, no API
  2. ContainsAny / ContainsAll — pattern matching
  3. SimilarityScorer — embedding cosine distance
  4. LLMJudge (G-Eval) — LLM judges LLM
  5. Faithfulness — checks hallucination against context
  6. ToolCorrectness — agent tool validation
"""

import pytest
from unittest.mock import MagicMock, patch


# ═══════════════════════════════════════════════════════════════════
# 1. ExactMatch Scorer
# ═══════════════════════════════════════════════════════════════════


class TestExactMatchScorer:
    """Simple string comparison — baseline scorer."""

    def test_identical_strings_score_1(self):
        from openeval.scorers.exact_match import ExactMatchScorer
        from openeval.test_case import TestCase

        scorer = ExactMatchScorer()
        tc = TestCase(
            input="x",
            actual_output="Hello world",
            expected_output="Hello world",
        )
        result = scorer.score(tc)
        assert result.value == 1.0

    def test_different_strings_score_0(self):
        from openeval.scorers.exact_match import ExactMatchScorer
        from openeval.test_case import TestCase

        scorer = ExactMatchScorer()
        tc = TestCase(
            input="x",
            actual_output="Hello world",
            expected_output="Goodbye world",
        )
        result = scorer.score(tc)
        assert result.value == 0.0

    def test_case_insensitive_option(self):
        """Should support case-insensitive matching."""
        from openeval.scorers.exact_match import ExactMatchScorer
        from openeval.test_case import TestCase

        scorer = ExactMatchScorer(case_sensitive=False)
        tc = TestCase(
            input="x",
            actual_output="Hello World",
            expected_output="hello world",
        )
        result = scorer.score(tc)
        assert result.value == 1.0

    def test_missing_expected_output_raises(self):
        """Cannot score without expected_output."""
        from openeval.scorers.exact_match import ExactMatchScorer
        from openeval.test_case import TestCase

        scorer = ExactMatchScorer()
        tc = TestCase(input="x", actual_output="hello")
        with pytest.raises((ValueError, Exception)):
            scorer.score(tc)

    def test_returns_score_result_object(self):
        """Score should return a ScoreResult, not raw float."""
        from openeval.scorers.exact_match import ExactMatchScorer
        from openeval.test_case import TestCase
        from openeval.types import ScoreResult

        scorer = ExactMatchScorer()
        tc = TestCase(input="x", actual_output="a", expected_output="a")
        result = scorer.score(tc)
        assert isinstance(result, ScoreResult)
        assert hasattr(result, "value")
        assert hasattr(result, "name")
        assert hasattr(result, "reason")


# ═══════════════════════════════════════════════════════════════════
# 2. Contains Scorer
# ═══════════════════════════════════════════════════════════════════


class TestContainsScorer:
    """Check if output contains specific keywords."""

    def test_contains_any_match(self):
        from openeval.scorers.contains import ContainsAnyScorer
        from openeval.test_case import TestCase

        scorer = ContainsAnyScorer(keywords=["refund", "return", "exchange"])
        tc = TestCase(
            input="x",
            actual_output="You can get a full refund within 30 days.",
        )
        result = scorer.score(tc)
        assert result.value == 1.0

    def test_contains_any_no_match(self):
        from openeval.scorers.contains import ContainsAnyScorer
        from openeval.test_case import TestCase

        scorer = ContainsAnyScorer(keywords=["refund", "return"])
        tc = TestCase(
            input="x",
            actual_output="The weather is sunny today.",
        )
        result = scorer.score(tc)
        assert result.value == 0.0

    def test_contains_all_partial(self):
        """ContainsAll should return fraction when some match."""
        from openeval.scorers.contains import ContainsAllScorer
        from openeval.test_case import TestCase

        scorer = ContainsAllScorer(keywords=["refund", "30 days", "free shipping"])
        tc = TestCase(
            input="x",
            actual_output="You can get a refund within 30 days.",
        )
        result = scorer.score(tc)
        # 2 out of 3 keywords found
        assert 0.5 < result.value < 1.0


# ═══════════════════════════════════════════════════════════════════
# 3. Similarity Scorer (Embeddings)
# ═══════════════════════════════════════════════════════════════════


class TestSimilarityScorer:
    """Cosine similarity via embeddings."""

    def test_identical_texts_high_score(self, mock_openai_client):
        """Same text should have cosine similarity = 1.0."""
        from openeval.scorers.similarity import SimilarityScorer
        from openeval.test_case import TestCase

        scorer = SimilarityScorer(client=mock_openai_client)
        tc = TestCase(
            input="x",
            actual_output="The sky is blue.",
            expected_output="The sky is blue.",
        )
        result = scorer.score(tc)
        # With identical embeddings, cosine = 1.0
        assert result.value >= 0.99

    def test_returns_score_between_0_and_1(self, mock_openai_client):
        from openeval.scorers.similarity import SimilarityScorer
        from openeval.test_case import TestCase

        scorer = SimilarityScorer(client=mock_openai_client)
        tc = TestCase(
            input="x",
            actual_output="Hello",
            expected_output="World",
        )
        result = scorer.score(tc)
        assert 0.0 <= result.value <= 1.0

    def test_threshold_pass_fail(self, mock_openai_client):
        """Scorer should indicate pass/fail based on threshold."""
        from openeval.scorers.similarity import SimilarityScorer
        from openeval.test_case import TestCase

        scorer = SimilarityScorer(client=mock_openai_client, threshold=0.8)
        tc = TestCase(
            input="x",
            actual_output="same text",
            expected_output="same text",
        )
        result = scorer.score(tc)
        assert result.passed is True

    def test_tracks_token_usage(self, mock_openai_client):
        """Should track how many tokens embedding used."""
        from openeval.scorers.similarity import SimilarityScorer
        from openeval.test_case import TestCase

        scorer = SimilarityScorer(client=mock_openai_client)
        tc = TestCase(
            input="x",
            actual_output="hello",
            expected_output="hello",
        )
        result = scorer.score(tc)
        assert result.token_usage is not None
        assert result.token_usage > 0


# ═══════════════════════════════════════════════════════════════════
# 4. LLM-as-a-Judge (G-Eval style)
# ═══════════════════════════════════════════════════════════════════


class TestLLMJudgeScorer:
    """The most powerful scorer — one LLM evaluates another."""

    def test_basic_judgment(self, mock_openai_client, simple_test_case):
        """LLM judge should return a score and reason."""
        from openeval.scorers.llm_judge import LLMJudgeScorer

        scorer = LLMJudgeScorer(
            name="Correctness",
            criteria="Is the actual output factually correct based on expected output?",
            client=mock_openai_client,
        )
        result = scorer.score(simple_test_case)
        assert 0.0 <= result.value <= 1.0
        assert result.reason is not None
        assert len(result.reason) > 0

    def test_custom_criteria(self, mock_openai_client, simple_test_case):
        """Should accept custom evaluation criteria."""
        from openeval.scorers.llm_judge import LLMJudgeScorer

        scorer = LLMJudgeScorer(
            name="Politeness",
            criteria="Is the response polite and professional?",
            client=mock_openai_client,
        )
        result = scorer.score(simple_test_case)
        assert result.name == "Politeness"

    def test_threshold_determines_pass(self, mock_openai_client, simple_test_case):
        """Should use threshold to determine pass/fail."""
        from openeval.scorers.llm_judge import LLMJudgeScorer

        scorer = LLMJudgeScorer(
            name="test",
            criteria="test",
            client=mock_openai_client,
            threshold=0.5,
        )
        result = scorer.score(simple_test_case)
        # Mock returns 0.85, threshold is 0.5 → should pass
        assert result.passed is True

    def test_high_threshold_fails(self, mock_openai_client, simple_test_case):
        """Score below threshold should fail."""
        from openeval.scorers.llm_judge import LLMJudgeScorer

        scorer = LLMJudgeScorer(
            name="test",
            criteria="test",
            client=mock_openai_client,
            threshold=0.95,
        )
        result = scorer.score(simple_test_case)
        # Mock returns 0.85, threshold is 0.95 → should fail
        assert result.passed is False

    def test_tracks_cost(self, mock_openai_client, simple_test_case):
        """Should track how much the judgment LLM call cost."""
        from openeval.scorers.llm_judge import LLMJudgeScorer

        scorer = LLMJudgeScorer(
            name="test",
            criteria="test",
            client=mock_openai_client,
        )
        result = scorer.score(simple_test_case)
        assert result.token_usage is not None
        assert result.cost_usd is not None
        assert result.cost_usd >= 0.0

    def test_handles_malformed_llm_response(self, simple_test_case):
        """If judge LLM returns garbage, should handle gracefully."""
        from openeval.scorers.llm_judge import LLMJudgeScorer

        bad_client = MagicMock()
        bad_response = MagicMock()
        bad_response.choices = [MagicMock()]
        bad_response.choices[0].message.content = "This is not JSON at all!"
        bad_response.usage.prompt_tokens = 100
        bad_response.usage.completion_tokens = 20
        bad_response.usage.total_tokens = 120
        bad_response.model = "gpt-4o-mini"
        bad_client.chat.completions.create.return_value = bad_response

        scorer = LLMJudgeScorer(
            name="test", criteria="test", client=bad_client
        )
        # Should NOT crash — should return error score or retry
        result = scorer.score(simple_test_case)
        assert result is not None
        assert isinstance(result.value, float)


# ═══════════════════════════════════════════════════════════════════
# 5. Faithfulness/Hallucination Scorer
# ═══════════════════════════════════════════════════════════════════


class TestFaithfulnessScorer:
    """Checks if output is faithful to provided context (no hallucination)."""

    def test_faithful_output_scores_high(self, mock_openai_client, rag_test_case):
        """Output grounded in context should score high."""
        from openeval.scorers.faithfulness import FaithfulnessScorer

        scorer = FaithfulnessScorer(client=mock_openai_client)
        result = scorer.score(rag_test_case)
        assert result.value > 0.5

    def test_hallucinated_output_scores_low(
        self, mock_openai_client_low_score, hallucination_test_case
    ):
        """Output with fabricated info should score low."""
        from openeval.scorers.faithfulness import FaithfulnessScorer

        scorer = FaithfulnessScorer(client=mock_openai_client_low_score)
        result = scorer.score(hallucination_test_case)
        assert result.value < 0.5

    def test_requires_context(self, mock_openai_client):
        """Faithfulness scorer needs context to compare against."""
        from openeval.scorers.faithfulness import FaithfulnessScorer
        from openeval.test_case import TestCase

        scorer = FaithfulnessScorer(client=mock_openai_client)
        tc = TestCase(input="x", actual_output="something")  # No context!
        with pytest.raises((ValueError, Exception)):
            scorer.score(tc)


# ═══════════════════════════════════════════════════════════════════
# 6. Tool Correctness Scorer (for agents)
# ═══════════════════════════════════════════════════════════════════


class TestToolCorrectnessScorer:
    """Validates that an agent called the right tools in the right order."""

    def test_correct_tools_score_1(self, agent_test_case):
        """All expected tools called → score 1.0."""
        from openeval.scorers.tool_correctness import ToolCorrectnessScorer

        scorer = ToolCorrectnessScorer()
        result = scorer.score(agent_test_case)
        assert result.value == 1.0

    def test_missing_tool_reduces_score(self):
        """Missing a tool should reduce score proportionally."""
        from openeval.scorers.tool_correctness import ToolCorrectnessScorer
        from openeval.test_case import TestCase

        scorer = ToolCorrectnessScorer()
        tc = TestCase(
            input="x",
            actual_output="booked",
            tools_called=["search_flights"],  # Missing 2 tools!
            expected_tools=["search_flights", "check_availability", "book_flight"],
        )
        result = scorer.score(tc)
        assert result.value < 1.0
        assert result.value > 0.0  # At least 1 tool matched

    def test_wrong_order_detected(self):
        """Tools called in wrong order should be detected."""
        from openeval.scorers.tool_correctness import ToolCorrectnessScorer
        from openeval.test_case import TestCase

        scorer = ToolCorrectnessScorer(check_order=True)
        tc = TestCase(
            input="x",
            actual_output="booked",
            tools_called=["book_flight", "search_flights", "check_availability"],
            expected_tools=["search_flights", "check_availability", "book_flight"],
        )
        result = scorer.score(tc)
        # Tools are correct but order is wrong
        assert result.value < 1.0

    def test_extra_tools_allowed(self):
        """Extra tools (not in expected) should not penalize by default."""
        from openeval.scorers.tool_correctness import ToolCorrectnessScorer
        from openeval.test_case import TestCase

        scorer = ToolCorrectnessScorer()
        tc = TestCase(
            input="x",
            actual_output="done",
            tools_called=[
                "search_flights",
                "check_weather",
                "check_availability",
                "book_flight",
            ],
            expected_tools=["search_flights", "check_availability", "book_flight"],
        )
        result = scorer.score(tc)
        assert result.value == 1.0  # All expected tools present


# ═══════════════════════════════════════════════════════════════════
# 7. Scorer Interface
# ═══════════════════════════════════════════════════════════════════


class TestScorerInterface:
    """All scorers should implement the same interface."""

    def test_all_scorers_have_name(self):
        """Every scorer has a .name property."""
        from openeval.scorers.exact_match import ExactMatchScorer
        from openeval.scorers.contains import ContainsAnyScorer

        assert ExactMatchScorer().name == "ExactMatch"
        assert ContainsAnyScorer(keywords=["x"]).name == "ContainsAny"

    def test_all_scorers_have_score_method(self):
        """Every scorer implements .score(test_case) → ScoreResult."""
        from openeval.scorers.exact_match import ExactMatchScorer

        scorer = ExactMatchScorer()
        assert callable(getattr(scorer, "score", None))

    def test_custom_scorer_function(self):
        """Users should be able to pass a plain function as scorer."""
        from openeval.scorers.base import FunctionScorer
        from openeval.test_case import TestCase

        def my_scorer(tc: TestCase) -> float:
            return 1.0 if "hello" in tc.actual_output.lower() else 0.0

        scorer = FunctionScorer(name="MyScorer", fn=my_scorer)
        tc = TestCase(input="x", actual_output="Hello there!")
        result = scorer.score(tc)
        assert result.value == 1.0
