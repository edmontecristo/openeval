"""
OpenEval Quickstart — Real LLM Agent Evaluation

This example evaluates a real GPT-4o-mini customer support agent using:
1. ContainsAnyScorer (Keyword matching)
2. FaithfulnessScorer (Hallucination detection via LLM-as-a-judge)
3. LLMJudgeScorer (Helpfulness detection via LLM-as-a-judge)

Requirements:
    pip install openai python-dotenv
    Make sure you have an OPENAI_API_KEY in your .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load API key from .env
load_dotenv(Path(__file__).parent.parent / ".env")

from openai import OpenAI
from openeval import Eval
from openeval.scorers import ContainsAnyScorer, LLMJudgeScorer, FaithfulnessScorer

client = OpenAI()

# ── 1. The LLM Agent to Evaluate ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a customer support agent. Answer based ONLY on this knowledge base:
- Return policy: 30-day full refund.
- Shipping: Free on orders over $50.
- Contact: support@example.com

If the question is outside this knowledge base, say "I don't have information about that."
"""

CONTEXT = [
    "Return policy: 30-day full refund.",
    "Shipping: Free on orders over $50.",
    "Contact: support@example.com",
]

def support_agent(question: str) -> str:
    """Real AI agent using GPT-4o-mini."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ── 2. Test Cases ────────────────────────────────────────────────────────────

TEST_DATA = [
    {
        "input": "What is your return policy?",
        "expected_output": "30-day full refund.",
        "context": CONTEXT,
    },
    {
        "input": "How much does shipping cost?",
        "expected_output": "Free shipping on orders over $50.",
        "context": CONTEXT,
    },
    {
        "input": "Tell me a joke about dogs.",
        "expected_output": "I don't have information about that.",
        "context": CONTEXT,
    },
]


# ── 3. Run the Evaluation ────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required.")
        exit(1)
        
    print("Running evaluation with real GPT-4o-mini... (this takes ~10 seconds)\n")

    result = Eval(
        name="gpt4o-mini-support-agent",
        data=TEST_DATA,
        task=support_agent,
        scorers=[
            ContainsAnyScorer(keywords=["refund", "shipping", "don't have"]),
            FaithfulnessScorer(client=client, threshold=0.7),
            LLMJudgeScorer(
                name="Helpfulness",
                criteria="Is this response helpful and accurate?",
                client=client,
                threshold=0.7,
            ),
        ],
    )

    # Automatically print comprehensive results table via rich (if run via CLI)
    # But since we're running as a script, we'll print a simple summary:
    for scorer_name, stats in result.summary.items():
        icon = "✅" if stats["mean"] >= 0.7 else "❌"
        print(f"{icon} {scorer_name:15s}: pass_rate={stats['pass_rate']:.0%} (mean={stats['mean']:.2f})")
    
    print(f"\nTotal Cost: ${result.total_cost_usd:.6f}")
