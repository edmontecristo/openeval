"""
test_eval_runner.py — Tests for the core Eval() function.

This is the main orchestrator: takes data, runs tasks, applies scorers,
returns results. Must work end-to-end reliably.
"""

import pytest
from unittest.mock import MagicMock


class TestEvalBasic:
    """Core Eval() function — the primary user-facing API."""

    def test_eval_runs_with_minimal_args(self):
        """Eval should work with just data + task + scorer."""
        from openeval import Eval
        from openeval.test_case import TestCase
        from openeval.scorers.exact_match import ExactMatchScorer

        result = Eval(
            name="basic-test",
            data=[
                {"input": "hello", "expected_output": "world"},
            ],
            task=lambda input: "world",
            scorers=[ExactMatchScorer()],
        )

        assert result is not None
        assert result.name == "basic-test"

    def test_eval_returns_experiment_result(self):
        """Eval should return an ExperimentResult with scores."""
        from openeval import Eval
        from openeval.scorers.exact_match import ExactMatchScorer
        from openeval.types import ExperimentResult

        result = Eval(
            name="test",
            data=[{"input": "a", "expected_output": "b"}],
            task=lambda input: "b",
            scorers=[ExactMatchScorer()],
        )

        assert isinstance(result, ExperimentResult)
        assert len(result.results) == 1
        assert result.results[0].scores["ExactMatch"].value == 1.0

    def test_eval_runs_multiple_data_points(self):
        """Should evaluate all data points, not just the first."""
        from openeval import Eval
        from openeval.scorers.exact_match import ExactMatchScorer

        data = [
            {"input": f"q{i}", "expected_output": f"a{i}"}
            for i in range(10)
        ]

        result = Eval(
            name="multi-test",
            data=data,
            task=lambda input: input.replace("q", "a"),
            scorers=[ExactMatchScorer()],
        )

        assert len(result.results) == 10
        assert all(r.scores["ExactMatch"].value == 1.0 for r in result.results)

    def test_eval_with_multiple_scorers(self):
        """Should apply ALL scorers to each test case."""
        from openeval import Eval
        from openeval.scorers.exact_match import ExactMatchScorer
        from openeval.scorers.contains import ContainsAnyScorer

        result = Eval(
            name="multi-scorer",
            data=[{"input": "test", "expected_output": "refund available"}],
            task=lambda input: "refund available",
            scorers=[
                ExactMatchScorer(),
                ContainsAnyScorer(keywords=["refund"]),
            ],
        )

        scores = result.results[0].scores
        assert "ExactMatch" in scores
        assert "ContainsAny" in scores
        assert scores["ExactMatch"].value == 1.0
        assert scores["ContainsAny"].value == 1.0


class TestEvalDataInput:
    """Eval should accept data in multiple formats."""

    def test_accepts_list_of_dicts(self):
        """Standard format — list of dicts."""
        from openeval import Eval
        from openeval.scorers.exact_match import ExactMatchScorer

        result = Eval(
            name="test",
            data=[{"input": "a", "expected_output": "b"}],
            task=lambda input: "b",
            scorers=[ExactMatchScorer()],
        )
        assert len(result.results) == 1

    def test_accepts_list_of_test_cases(self):
        """Should also accept pre-built TestCase objects."""
        from openeval import Eval
        from openeval.test_case import TestCase
        from openeval.scorers.exact_match import ExactMatchScorer

        data = [
            TestCase(input="hello", actual_output="world", expected_output="world"),
        ]
        result = Eval(
            name="test",
            data=data,
            task=None,  # actual_output already set
            scorers=[ExactMatchScorer()],
        )
        assert len(result.results) == 1

    def test_accepts_dataset_object(self):
        """Should accept a Dataset object with versioning."""
        from openeval import Eval
        from openeval.dataset import Dataset
        from openeval.scorers.exact_match import ExactMatchScorer

        ds = Dataset(name="test-dataset")
        ds.add({"input": "q1", "expected_output": "a1"})
        ds.add({"input": "q2", "expected_output": "a2"})

        result = Eval(
            name="test",
            data=ds,
            task=lambda input: input.replace("q", "a"),
            scorers=[ExactMatchScorer()],
        )
        assert len(result.results) == 2

    def test_empty_data_raises(self):
        """Cannot run eval with no data."""
        from openeval import Eval
        from openeval.scorers.exact_match import ExactMatchScorer

        with pytest.raises((ValueError, Exception)):
            Eval(name="test", data=[], task=lambda x: x, scorers=[ExactMatchScorer()])


class TestEvalTask:
    """The task function (the LLM call being evaluated)."""

    def test_task_receives_input_returns_output(self):
        """Task function gets input string, returns output string."""
        from openeval import Eval
        from openeval.scorers.exact_match import ExactMatchScorer

        def my_chatbot(input: str) -> str:
            return f"Response to: {input}"

        result = Eval(
            name="test",
            data=[{"input": "hi", "expected_output": "Response to: hi"}],
            task=my_chatbot,
            scorers=[ExactMatchScorer()],
        )
        assert result.results[0].scores["ExactMatch"].value == 1.0

    def test_task_exception_captured_not_crashed(self):
        """If task function throws, eval should capture error, not crash."""
        from openeval import Eval
        from openeval.scorers.exact_match import ExactMatchScorer

        def buggy_task(input: str) -> str:
            raise RuntimeError("LLM API is down!")

        result = Eval(
            name="test",
            data=[{"input": "hi", "expected_output": "hello"}],
            task=buggy_task,
            scorers=[ExactMatchScorer()],
        )
        # Should complete with error status, not crash
        assert result is not None
        assert result.results[0].error is not None

    def test_task_none_when_actual_output_preset(self):
        """If actual_output already in data, task can be None."""
        from openeval import Eval
        from openeval.test_case import TestCase
        from openeval.scorers.exact_match import ExactMatchScorer

        data = [
            TestCase(input="x", actual_output="hello", expected_output="hello"),
        ]
        result = Eval(
            name="test", data=data, task=None, scorers=[ExactMatchScorer()]
        )
        assert result.results[0].scores["ExactMatch"].value == 1.0


class TestEvalMetrics:
    """Eval should compute aggregate metrics."""

    def test_computes_average_score(self):
        """Should calculate mean score across all test cases."""
        from openeval import Eval
        from openeval.scorers.exact_match import ExactMatchScorer

        data = [
            {"input": "a", "expected_output": "a"},  # match
            {"input": "b", "expected_output": "b"},  # match
            {"input": "c", "expected_output": "X"},  # no match
        ]

        result = Eval(
            name="test",
            data=data,
            task=lambda input: input,
            scorers=[ExactMatchScorer()],
        )

        avg = result.summary["ExactMatch"]["mean"]
        assert abs(avg - 0.6667) < 0.01

    def test_tracks_total_duration(self):
        """Should measure how long the entire eval took."""
        from openeval import Eval
        from openeval.scorers.exact_match import ExactMatchScorer

        result = Eval(
            name="test",
            data=[{"input": "a", "expected_output": "a"}],
            task=lambda input: input,
            scorers=[ExactMatchScorer()],
        )
        assert result.duration_ms is not None
        assert result.duration_ms > 0

    def test_tracks_total_cost(self, mock_openai_client):
        """Should sum up cost across all scorer LLM calls."""
        from openeval import Eval
        from openeval.scorers.llm_judge import LLMJudgeScorer

        result = Eval(
            name="test",
            data=[
                {"input": "a", "expected_output": "a"},
                {"input": "b", "expected_output": "b"},
            ],
            task=lambda input: input,
            scorers=[
                LLMJudgeScorer(
                    name="test", criteria="test", client=mock_openai_client
                )
            ],
        )
        assert result.total_cost_usd is not None
        assert result.total_cost_usd >= 0.0

    def test_tracks_pass_fail_rate(self):
        """Should report pass/fail percentage."""
        from openeval import Eval
        from openeval.scorers.exact_match import ExactMatchScorer

        data = [
            {"input": "a", "expected_output": "a"},
            {"input": "b", "expected_output": "WRONG"},
        ]

        result = Eval(
            name="test",
            data=data,
            task=lambda input: input,
            scorers=[ExactMatchScorer(threshold=0.5)],
        )

        assert result.summary["ExactMatch"]["pass_rate"] == 0.5


class TestExperimentComparison:
    """Compare two experiment runs to detect regression."""

    def test_compare_two_experiments(self):
        """Should detect improvement or regression."""
        from openeval import Eval
        from openeval.experiment import compare_experiments
        from openeval.scorers.exact_match import ExactMatchScorer

        data = [{"input": "a", "expected_output": "a"}]

        # Version 1: bad
        result_v1 = Eval(
            name="v1",
            data=data,
            task=lambda input: "wrong",
            scorers=[ExactMatchScorer()],
        )

        # Version 2: good
        result_v2 = Eval(
            name="v2",
            data=data,
            task=lambda input: input,
            scorers=[ExactMatchScorer()],
        )

        comparison = compare_experiments(result_v1, result_v2)
        assert comparison["ExactMatch"]["delta"] > 0  # v2 is better
        assert comparison["ExactMatch"]["improved"] is True
