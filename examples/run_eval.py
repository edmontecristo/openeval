"""
Example: Run evaluation on the demo chatbot.

This is what users will copy-paste to get started.
Run: openeval run examples/run_eval.py
"""

from openeval import Eval
from openeval.scorers.exact_match import ExactMatchScorer
from openeval.scorers.contains import ContainsAnyScorer
from examples.demo_agent import chatbot, DATASET

# ─── Run evaluation ──────────────────────────────────────────────

result = Eval(
    name="customer-support-chatbot-v1",
    data=DATASET,
    task=chatbot,
    scorers=[
        ExactMatchScorer(),
        ContainsAnyScorer(keywords=["refund", "shipping", "support", "cancel"]),
    ],
)

# Result is printed automatically by CLI
# Or access programmatically:
print(f"\nExperiment: {result.name}")
print(f"Duration: {result.duration_ms:.0f}ms")
for scorer, stats in result.summary.items():
    print(f"  {scorer}: mean={stats['mean']:.2f}, pass_rate={stats['pass_rate']:.0%}")
