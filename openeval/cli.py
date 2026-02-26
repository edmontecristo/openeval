"""
openeval/cli.py — Command-line interface for OpenEval.

Provides the `openeval run` command to execute evaluation scripts
and generate HTML reports with optional failure thresholds for CI/CD.
"""

import click
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """OpenEval — Open-source LLM evaluation framework."""
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
              CLI-First LLM Evaluation Framework
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        console.print(f"[bold blue]{logo}[/bold blue]", highlight=False)
        console.print("Welcome to OpenEval! \n")
        console.print("Get started by creating an evaluation script, ex: `eval.py`. Then run:")
        console.print("  [bold cyan]openeval run eval.py[/bold cyan]\n")
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
    """
    # Execute the eval file
    namespace = {}
    with open(file, "r") as f:
        exec(f.read(), namespace)

    # Find ExperimentResult in namespace
    from openeval.types import ExperimentResult

    result = None
    for val in namespace.values():
        if isinstance(val, ExperimentResult):
            result = val
            break

    if result is None:
        click.echo("Error: No ExperimentResult found in file", err=True)
        sys.exit(1)

    # Print rich table
    _print_results(result)

    # Save HTML report if requested
    if report:
        from openeval.report import save_report

        save_report(result, report)
        click.echo(f"Report saved to {report}")

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


if __name__ == "__main__":
    main()
