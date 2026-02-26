"""
test_cli.py — Tests for the openeval CLI.

Users run: openeval run my_eval.py
Should show rich terminal output + optional HTML export.
"""

import pytest
import subprocess
import sys
import os


class TestCLIRun:
    """The `openeval run` command."""

    def test_cli_help(self):
        """openeval --help should not crash."""
        result = subprocess.run(
            [sys.executable, "-m", "openeval.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        assert result.returncode == 0
        assert "openeval" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_cli_run_eval_file(self, tmp_path):
        """openeval run <file> should execute evaluation."""
        # Create a minimal eval file
        eval_file = tmp_path / "my_eval.py"
        eval_file.write_text(
            '''
from openeval import Eval
from openeval.scorers.exact_match import ExactMatchScorer

result = Eval(
    name="cli-test",
    data=[{"input": "hello", "expected_output": "hello"}],
    task=lambda input: input,
    scorers=[ExactMatchScorer()],
)
'''
        )

        result = subprocess.run(
            [sys.executable, "-m", "openeval.cli", "run", str(eval_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_cli_outputs_table(self, tmp_path):
        """CLI should print a rich formatted results table."""
        eval_file = tmp_path / "my_eval.py"
        eval_file.write_text(
            '''
from openeval import Eval
from openeval.scorers.exact_match import ExactMatchScorer

result = Eval(
    name="table-test",
    data=[
        {"input": "a", "expected_output": "a"},
        {"input": "b", "expected_output": "wrong"},
    ],
    task=lambda input: input,
    scorers=[ExactMatchScorer()],
)
'''
        )

        result = subprocess.run(
            [sys.executable, "-m", "openeval.cli", "run", str(eval_file)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should print score summary
        assert "ExactMatch" in result.stdout or "exact" in result.stdout.lower()

    def test_cli_exit_code_on_failure(self, tmp_path):
        """CLI should exit with code 1 when evals fail threshold."""
        eval_file = tmp_path / "my_eval.py"
        eval_file.write_text(
            '''
from openeval import Eval
from openeval.scorers.exact_match import ExactMatchScorer

result = Eval(
    name="fail-test",
    data=[{"input": "a", "expected_output": "WRONG"}],
    task=lambda input: input,
    scorers=[ExactMatchScorer(threshold=0.5)],
)
'''
        )

        result = subprocess.run(
            [sys.executable, "-m", "openeval.cli", "run", str(eval_file), "--fail-under", "0.9"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should fail because score = 0.0 < 0.9
        assert result.returncode != 0


class TestCLIReport:
    """HTML report generation via CLI."""

    def test_generates_html_report(self, tmp_path):
        """--report flag should produce an HTML file."""
        eval_file = tmp_path / "my_eval.py"
        eval_file.write_text(
            '''
from openeval import Eval
from openeval.scorers.exact_match import ExactMatchScorer

result = Eval(
    name="report-test",
    data=[{"input": "a", "expected_output": "a"}],
    task=lambda input: input,
    scorers=[ExactMatchScorer()],
)
'''
        )

        report_path = tmp_path / "report.html"
        result = subprocess.run(
            [
                sys.executable, "-m", "openeval.cli", "run",
                str(eval_file),
                "--report", str(report_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert report_path.exists()
        content = report_path.read_text()
        assert "<html" in content.lower()
        assert "report-test" in content
