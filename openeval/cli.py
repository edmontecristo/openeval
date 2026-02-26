"""
openeval/cli.py — Command-line interface for OpenEval.

Agent-First Design:
- AI agents discover APIs via --ai-docs (machine-readable markdown)
- Clean, localized error messages (no framework trace spam)
- Auto sys.path handling (works from current directory)
- Graceful telemetry failures (no LangSmith stack traces)
"""

import os
import traceback
import inspect
from pathlib import Path
from typing import get_type_hints

import click
import sys
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax


# Auto-prepend CWD to sys.path for agent convenience
_cwd = str(Path.cwd())
if _cwd not in sys.path:
    sys.path.insert(0, _cwd)


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """OpenEval — AI Agent Evaluation Framework (Agent-First UX)."""
    if ctx.invoked_subcommand is None:
        console = Console()
        logo = """
 ██████╗ ██████╗ ███████╗███╗   ██╗███████╗██╗   ██╗ █████╗ ██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║   ██║██╔══██╗██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║█████╗  ██║   ██║███████║██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══╝  ╚██╗ ██╔╝██╔══██║██║
╚██████╔╝██║     ███████╗██║ ╚████║███████╗ ╚████╔╝ ██║  ██║███████╗
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
          Agent-First LLM Evaluation Framework
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        console.print(f"[bold blue]{logo}[/bold blue]", highlight=False)
        console.print("Welcome to OpenEval! \n")
        console.print("Get started by creating an evaluation script, ex: `eval.py`. Then run:")
        console.print("  [bold cyan]openeval run eval.py[/bold cyan]\n")
        console.print("For AI-friendly API docs:")
        console.print("  [bold cyan]openeval ai-docs[/bold cyan]\n")
        console.print("For more information and options, run:")
        console.print("  [bold cyan]openeval --help[/bold cyan]")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--fail-under",
    type=float,
    default=None,
    help="Exit with code 1 if any scorer mean is below this threshold",
)
@click.option(
    "--report",
    type=click.Path(),
    default=None,
    help="Path to save HTML report",
)
def run(file, fail_under, report):
    """
    Run an evaluation file.

    The FILE should be a Python script that creates an Eval() instance.
    The last ExperimentResult in the file will be displayed.

    Agent-First: CWD is automatically in sys.path. Errors are localized.
    """
    # Disable noisy telemetry (LangSmith, etc.) for clean agent output
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "")

    # Execute the eval file with isolated error handling
    namespace = {}
    try:
        with open(file, "r") as f:
            code = f.read()
            exec(code, namespace)
    except Exception as e:
        # Agent-friendly error: show ONLY the user's error, not framework internals
        _print_agent_error(e, file)
        sys.exit(1)

    # Find ExperimentResult in namespace
    from openeval.types import ExperimentResult

    result = None
    for val in namespace.values():
        if isinstance(val, ExperimentResult):
            result = val
            break

    if result is None:
        click.echo(
            f"Error: No ExperimentResult found in {file}. "
            "Ensure your script creates an Eval() instance.",
            err=True
        )
        sys.exit(1)

    # Print rich table
    _print_results(result)

    # Save HTML report if requested
    if report:
        try:
            from openeval.report import save_report
            save_report(result, report)
            click.echo(f"Report saved to {report}")
        except Exception as e:
            click.echo(f"Warning: Could not save report: {e}", err=True)

    # Check fail-under threshold
    if fail_under is not None:
        for scorer_name, stats in result.summary.items():
            if stats["mean"] < fail_under:
                click.echo(
                    f"FAIL: {scorer_name} mean {stats['mean']:.4f} < {fail_under}"
                )
                sys.exit(1)


def _print_results(result):
    """Print results as a rich table."""
    console = Console()

    table = Table(title=f"Experiment: {result.name}")
    table.add_column("Scorer", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Pass Rate", style="yellow")

    for name, stats in result.summary.items():
        pass_rate = stats.get("pass_rate", 0)
        table.add_row(
            name,
            f"{stats['mean']:.4f}",
            f"{pass_rate:.0%}",
        )

    console.print(table)

    if result.duration_ms:
        console.print(f"Duration: {result.duration_ms:.0f}ms")
    if result.total_cost_usd:
        console.print(f"Cost: ${result.total_cost_usd:.6f}")


def _print_agent_error(error, filepath):
    """Print agent-friendly localized error without framework trace spam."""
    console = Console()

    # Extract just the user's error line
    tb_lines = traceback.format_exception(type(error), error, error.__traceback__)

    # Filter out framework internals (click.py, openeval/cli.py, etc.)
    user_tb_lines = []
    for line in tb_lines:
        # Skip framework trace lines
        if any(x in line for x in ["click/", "openeval/cli.py", "site-packages/"]):
            continue
        user_tb_lines.append(line)

    user_tb = "".join(user_tb_lines)

    console.print(f"[bold red]Error in {filepath}[/bold red]\n")

    # Show clean error type and message
    console.print(f"[red]{type(error).__name__}: {error}[/red]\n")

    # Show filtered traceback if available
    if user_tb.strip():
        console.print("[dim]Traceback (most recent call last):[/dim]")
        console.print(Syntax(user_tb, "python", theme="monokai", line_numbers=False))

    # Agent hint
    console.print("\n[yellow]Hint:[/yellow] AI agents can fix this by addressing the specific error above.")


@main.command()
@click.option(
    "--format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    help="Output format for AI consumption",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Save to file instead of stdout",
)
def ai_docs(format, output):
    """
    Dump machine-readable API documentation for AI agents.

    Agents can parse this to understand the API without browsing docs.
    Includes exact signatures, type hints, and minimal examples.
    """
    docs = _generate_ai_docs()

    if output:
        Path(output).write_text(docs)
        click.echo(f"AI docs saved to {output}")
    else:
        click.echo(docs)


def _generate_ai_docs() -> str:
    """Generate markdown API documentation for AI consumption."""
    from openeval.scorers.exact_match import ExactMatchScorer
    from openeval.scorers.contains import ContainsAnyScorer, ContainsAllScorer
    from openeval.scorers.similarity import SimilarityScorer
    from openeval.scorers.llm_judge import LLMJudgeScorer
    from openeval.scorers.faithfulness import FaithfulnessScorer
    from openeval.scorers.tool_correctness import ToolCorrectnessScorer
    from openeval.scorers.base import FunctionScorer

    all_scorers = [
        ExactMatchScorer,
        ContainsAnyScorer,
        ContainsAllScorer,
        SimilarityScorer,
        LLMJudgeScorer,
        FaithfulnessScorer,
        ToolCorrectnessScorer,
        FunctionScorer,
    ]

    scorer_docs = []
    for scorer_cls in all_scorers:
        try:
            sig = inspect.signature(scorer_cls)
            scorer_docs.append(f"### `{scorer_cls.__name__}`\n\n")
            scorer_docs.append(f"**Signature:** `{scorer_cls.__name__}{sig}`\n\n")

            # Add docstring
            if scorer_cls.__doc__:
                doc_lines = scorer_cls.__doc__.strip().split("\n")[:3]
                scorer_docs.append(f"**Purpose:** {' '.join(doc_lines)}\n\n")
        except Exception:
            scorer_docs.append(f"### `{scorer_cls.__name__}`\n\n")

    return f"""# OpenEval API Documentation (AI-First)

> This document is machine-readable for AI agents to parse and use OpenEval correctly.

## Quick Start

```python
from openeval import Eval
from openeval.scorers import ExactMatchScorer

result = Eval(
    name="my-eval",
    data=[
        {{"input": "What is 2+2?", "expected_output": "4"}},
        {{"input": "Capital of France?", "expected_output": "Paris"}},
    ],
    task=lambda x: f"Answer: {{x}}",  # Your agent function
    scorers=[ExactMatchScorer()],
)

print(result.summary)
```

## Core API

### `Eval(name, data, task, scorers)`

**Parameters:**
- `name` (str): Experiment identifier
- `data` (list): List of test cases. Each can be:
  - A dict with keys: `input`, `expected_output`, `context`, `expected_tools`
  - A `TestCase` object
  - A `Dataset` object
- `task` (callable): Function that takes `input` (str) and returns `actual_output` (str)
- `scorers` (list): List of scorer instances (see below)

**Returns:** `ExperimentResult` object with:
- `results`: List of individual test results
- `summary`: Dict with mean, min, max, pass_rate per scorer
- `duration_ms`: Total time in milliseconds
- `total_cost_usd`: Total API cost in USD

### `TestCase` Fields

- `input` (str): The input prompt
- `actual_output` (str, optional): The agent's output
- `expected_output` (str, optional): Expected ground truth
- `context` (list[str], optional): Retrieved context for RAG
- `expected_tools` (list[str], optional): Expected tool calls in order
- `tools_called` (list[str], optional): Actual tools called
- `id` (str, optional): Test case identifier

## Scorers

{"".join(scorer_docs)}

### Custom Scorer with FunctionScorer

```python
from openeval.scorers.base import FunctionScorer

# FunctionScorer receives a TestCase object with all fields
def my_scorer(tc):
    # tc has: input, actual_output, expected_output, context, tools_called, etc.
    return 1.0 if "keyword" in tc.actual_output else 0.0

scorer = FunctionScorer(name="HasKeyword", fn=my_scorer)

# Or with lambda:
scorer = FunctionScorer(
    name="OutputLength",
    fn=lambda tc: min(len(tc.actual_output) / 100, 1.0),
)
```

## CLI Usage

```bash
# Run evaluation
openeval run eval.py

# Fail CI if scores below threshold
openeval run eval.py --fail-under 0.8

# Generate HTML report
openeval run eval.py --report results.html

# Get API docs (this command)
openeval ai-docs
```

## Test Data File Formats

**YAML (eval.yaml):**
```yaml
- input: "What is 2+2?"
  expected_output: "4"

- input: "Fix the bug"
  expected_tools: [read_file, edit_file, run_tests]
  context: ["Bug in line 42"]
```

**JSON (eval.json):**
```json
[
  {{"input": "What is 2+2?", "expected_output": "4"}},
  {{"input": "Capital of France?", "expected_output": "Paris"}}
]
```

**CSV (eval.csv):**
```csv
input,expected_output
"What is 2+2?","4"
"Capital of France?","Paris"
```

## Common Errors (Agent-Readable)

- `ValueError: data cannot be empty` → Provide at least one test case
- `ValueError: scorers cannot be empty` → Provide at least one scorer
- `ModuleNotFoundError` → Ensure CWD is in sys.path (auto-handled by CLI)
- `AttributeError: 'TestCase' object has no attribute 'foo'` → Check field name

---
Generated by OpenEval {os.environ.get("OPENEVAL_VERSION", "0.1.0")}
"""


if __name__ == "__main__":
    main()
