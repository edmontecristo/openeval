"""
test_report.py — Tests for the HTML report generator.

Reports are self-contained HTML files (like LangExtract)
that show evaluation results with interactive elements.
"""

import pytest


class TestHTMLReport:
    """Self-contained HTML report generation."""

    def test_generates_valid_html(self):
        """Report should be valid HTML."""
        from openeval.report import generate_html_report
        from openeval.types import ExperimentResult

        # Create minimal experiment result
        result = ExperimentResult(
            name="test-experiment",
            results=[],
            summary={},
            duration_ms=100,
        )
        html = generate_html_report(result)
        assert "<html" in html.lower()
        assert "</html>" in html.lower()

    def test_contains_experiment_name(self):
        """Report should display the experiment name."""
        from openeval.report import generate_html_report
        from openeval.types import ExperimentResult

        result = ExperimentResult(
            name="my-chatbot-v2-eval",
            results=[],
            summary={},
            duration_ms=100,
        )
        html = generate_html_report(result)
        assert "my-chatbot-v2-eval" in html

    def test_contains_score_data(self):
        """Report should show individual scores."""
        from openeval.report import generate_html_report
        from openeval.types import ExperimentResult, EvalResult, ScoreResult

        result = ExperimentResult(
            name="test",
            results=[
                EvalResult(
                    test_case_id="tc1",
                    input="hello",
                    actual_output="world",
                    expected_output="world",
                    scores={
                        "ExactMatch": ScoreResult(
                            name="ExactMatch", value=1.0, reason="Perfect match"
                        )
                    },
                )
            ],
            summary={"ExactMatch": {"mean": 1.0}},
            duration_ms=50,
        )
        html = generate_html_report(result)
        assert "ExactMatch" in html
        assert "1.0" in html or "100" in html

    def test_is_self_contained(self):
        """Report should work offline — no external CSS/JS links."""
        from openeval.report import generate_html_report
        from openeval.types import ExperimentResult

        result = ExperimentResult(
            name="test", results=[], summary={}, duration_ms=10
        )
        html = generate_html_report(result)
        # Should NOT have external links
        assert "https://" not in html or "<style" in html

    def test_saves_to_file(self, tmp_report_dir):
        """Should save HTML to disk."""
        from openeval.report import generate_html_report, save_report
        from openeval.types import ExperimentResult

        result = ExperimentResult(
            name="test", results=[], summary={}, duration_ms=10
        )
        path = tmp_report_dir / "report.html"
        save_report(result, path)
        assert path.exists()
        assert path.stat().st_size > 100


class TestReportComparison:
    """Report should support comparing two experiments."""

    def test_comparison_report_shows_delta(self):
        """Should highlight improvements and regressions."""
        from openeval.report import generate_comparison_report
        from openeval.types import ExperimentResult

        v1 = ExperimentResult(
            name="v1",
            results=[],
            summary={"ExactMatch": {"mean": 0.6}},
            duration_ms=100,
        )
        v2 = ExperimentResult(
            name="v2",
            results=[],
            summary={"ExactMatch": {"mean": 0.85}},
            duration_ms=90,
        )

        html = generate_comparison_report(v1, v2)
        assert "v1" in html
        assert "v2" in html
        # Should show improvement
        assert "+" in html or "improved" in html.lower() or "↑" in html
