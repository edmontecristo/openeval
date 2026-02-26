"""
OpenEval + Ollama Demo — Real Agent Evaluation

This evaluates a local LLM agent using Ollama (no API keys needed!).
Tests a code assistant on Python questions.
"""

import json
import time
from openai import OpenAI

# Connect to Ollama (runs locally on port 11434)
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Ollama doesn't require a real key
)

# ── 1. The Agent: Code Assistant ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Python code assistant. Answer concisely with code examples when helpful.
Keep responses under 100 words unless code is needed."""

def code_agent(question: str) -> str:
    """Local AI agent using Ollama tinyllama."""
    try:
        response = client.chat.completions.create(
            model="tinyllama",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"Error: {e}"


# ── 2. Test Cases ────────────────────────────────────────────────────────────

TEST_DATA = [
    {
        "input": "Write a function to reverse a string in Python",
        "expected_output": "def reverse",
    },
    {
        "input": "How do I read a file in Python?",
        "expected_output": "open(",
    },
    {
        "input": "What is the sum of 5 and 3?",
        "expected_output": "8",
    },
    {
        "input": "Create a list comprehension for squares",
        "expected_output": "[x**2",
    },
]


# ── 3. Run Evaluation ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🦙 OpenEval + Ollama Demo")
    print("=" * 50)
    print("Model: tinyllama (local, no API key needed)")
    print("Test cases: 4")
    print()

    # Import after print for clean output
    from openeval import Eval
    from openeval.scorers import ContainsAnyScorer, ExactMatchScorer, LLMJudgeScorer

    start = time.time()

    result = Eval(
        name="ollama-tinyllama-demo",
        data=TEST_DATA,
        task=code_agent,
        scorers=[
            ContainsAnyScorer(keywords=["def", "open(", "8", "[x**2"]),
            ExactMatchScorer(case_sensitive=False),
            # LLM judge also works with Ollama!
            LLMJudgeScorer(
                name="Helpfulness",
                criteria="Is the answer helpful for a Python question?",
                client=client,
                model="tinyllama",
                threshold=0.5,
            ),
        ],
    )

    duration = time.time() - start

    # Print results
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)

    for scorer_name, stats in result.summary.items():
        icon = "✅" if stats["mean"] >= 0.5 else "❌"
        print(f"{icon} {scorer_name:20s}: {stats['mean']:.4f} (pass_rate={stats['pass_rate']:.0%})")

    print(f"\n⏱️  Duration: {duration:.1f}s")
    print(f"💰 Cost: $0.00 (local Ollama!)")

    # Show sample outputs
    print("\n" + "=" * 50)
    print("📝 SAMPLE OUTPUTS")
    print("=" * 50)
    for i, r in enumerate(result.results[:2]):
        print(f"\n--- Test {i+1} ---")
        print(f"Input: {r.input}")
        print(f"Output: {r.actual_output[:100]}...")
        print(f"Scores: {r.scores}")
