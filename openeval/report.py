"""
openeval/report.py — Self-contained HTML report generation.

Generates single-file HTML reports with inline CSS/JS that work offline,
inspired by LangExtract. Reports include experiment summaries, detailed
results, and comparison views.
"""

from jinja2 import Template
from pathlib import Path
from openeval.types import ExperimentResult


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenEval Report — {{ name }}</title>
    <style>
        /* Inline all styles — must work offline */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 2rem;
            line-height: 1.6;
        }
        h1 { color: #38bdf8; margin-bottom: 0.5rem; font-size: 2rem; }
        .subtitle { color: #94a3b8; margin-bottom: 2rem; }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        .card {
            background: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #334155;
        }
        .card-label { color: #94a3b8; font-size: 0.875rem; margin-bottom: 0.5rem; }
        .score { font-size: 2rem; font-weight: bold; }
        .pass { color: #4ade80; }
        .fail { color: #f87171; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 2rem 0;
            background: #1e293b;
            border-radius: 8px;
            overflow: hidden;
        }
        th {
            background: #334155;
            padding: 1rem;
            text-align: left;
            font-weight: 600;
            color: #f1f5f9;
        }
        td {
            padding: 1rem;
            border-bottom: 1px solid #334155;
        }
        tr:last-child td { border-bottom: none; }
        tr:hover { background: #1e293b; }
        .reason {
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 0.25rem;
        }
        .output-cell {
            max-width: 300px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .value-cell {
            font-weight: 600;
        }
    </style>
</head>
<body>
    <h1>{{ name }}</h1>
    <p class="subtitle">
        Duration: {{ "%.0f" | format(duration_ms) }}ms
        {% if total_cost %}| Cost: ${{ "%.6f" | format(total_cost) }}{% endif %}
    </p>

    <div class="summary">
    {% for scorer_name, stats in summary.items() %}
        <div class="card">
            <div class="card-label">{{ scorer_name }}</div>
            <div class="score {{ 'pass' if stats.mean >= 0.7 else 'fail' }}">
                {{ "%.1f" | format(stats.mean * 100) }}%
            </div>
            <div class="card-label">Pass Rate: {{ "%.0f" | format(stats.get('pass_rate', 0) * 100) }}%</div>
        </div>
    {% endfor %}
    </div>

    <h2>Detailed Results</h2>
    <table>
        <thead>
            <tr>
                <th>#</th>
                <th>Input</th>
                <th>Output</th>
                {% for scorer in scorer_names %}
                <th>{{ scorer }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
        {% for r in results %}
            <tr>
                <td>{{ loop.index }}</td>
                <td class="output-cell" title="{{ r.input }}">{{ r.input[:80] }}</td>
                <td class="output-cell" title="{{ r.actual_output }}">{{ r.actual_output[:80] }}</td>
                {% for scorer in scorer_names %}
                <td class="value-cell">
                    {% if scorer in r.scores %}
                    <div class="{{ 'pass' if r.scores[scorer].value >= 0.7 else 'fail' }}">
                        {{ "%.2f" | format(r.scores[scorer].value) }}
                    </div>
                    {% if r.scores[scorer].reason %}
                    <div class="reason">{{ r.scores[scorer].reason[:80] }}</div>
                    {% endif %}
                    {% else %}
                    <span style="color: #64748b;">N/A</span>
                    {% endif %}
                </td>
                {% endfor %}
            </tr>
        {% endfor %}
        </tbody>
    </table>
</body>
</html>
"""


def generate_html_report(result: ExperimentResult) -> str:
    """
    Generate self-contained HTML report.

    Args:
        result: ExperimentResult with evaluation results

    Returns:
        HTML string with inline styles (no external dependencies)
    """
    template = Template(REPORT_TEMPLATE)
    scorer_names = list(result.summary.keys()) if result.summary else []
    return template.render(
        name=result.name,
        duration_ms=result.duration_ms or 0,
        total_cost=result.total_cost_usd,
        summary=result.summary,
        results=result.results,
        scorer_names=scorer_names,
    )


def save_report(result: ExperimentResult, path):
    """
    Save HTML report to file.

    Args:
        result: ExperimentResult with evaluation results
        path: Path (str or Path) where to save the HTML file
    """
    html = generate_html_report(result)
    Path(path).write_text(html, encoding="utf-8")


def generate_comparison_report(
    baseline: ExperimentResult, candidate: ExperimentResult
) -> str:
    """
    Generate comparison report for two experiments.

    Shows side-by-side comparison with delta indicators for improvements
    and regressions.

    Args:
        baseline: Baseline experiment result
        candidate: Candidate experiment result to compare

    Returns:
        HTML string with comparison table
    """
    from openeval.experiment import compare_experiments

    comparison = compare_experiments(baseline, candidate)

    # Build comparison HTML
    rows = []
    for scorer_name, comp in comparison.items():
        delta = comp["delta"]
        delta_symbol = "↑" if comp["improved"] else ("↓" if delta < 0 else "=")
        delta_color = "#4ade80" if comp["improved"] else ("#f87171" if delta < 0 else "#94a3b8")

        rows.append(
            f"""
            <tr>
                <td>{scorer_name}</td>
                <td>{comp['baseline']:.4f}</td>
                <td>{comp['candidate']:.4f}</td>
                <td style="color: {delta_color}">{delta_symbol} {delta:+.4f}</td>
            </tr>
        """
        )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Experiment Comparison — {baseline.name} vs {candidate.name}</title>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #0f172a;
                color: #e2e8f0;
                padding: 2rem;
            }}
            h1 {{ color: #38bdf8; margin-bottom: 2rem; }}
            h2 {{ color: #94a3b8; margin-bottom: 1rem; }}
            table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; }}
            th {{ background: #334155; padding: 1rem; text-align: left; }}
            td {{ padding: 1rem; border-bottom: 1px solid #334155; }}
            tr:last-child td {{ border-bottom: none; }}
        </style>
    </head>
    <body>
        <h1>Experiment Comparison</h1>
        <h2>{baseline.name} → {candidate.name}</h2>
        <table>
            <thead>
                <tr>
                    <th>Scorer</th>
                    <th>Baseline</th>
                    <th>Candidate</th>
                    <th>Delta</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </body>
    </html>
    """
    return html
