"""
OpenEval Quickstart — Ollama Demo for CLI

Run with: openeval run examples/quickstart_ollama.py
"""

from openai import OpenAI

# Connect to Ollama (runs locally)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def code_agent(question: str) -> str:
    """Local AI agent using Ollama."""
    response = client.chat.completions.create(
        model="tinyllama",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content or ""

# Test data
TEST_DATA = [
    {"input": "What is 2+2?", "expected_output": "4"},
    {"input": "Write a Python function to add two numbers", "expected_output": "def"},
]

# Import and run
from openeval import Eval
from openeval.scorers import ContainsAnyScorer, ExactMatchScorer

result = Eval(
    name="ollama-quickstart",
    data=TEST_DATA,
    task=code_agent,
    scorers=[
        ContainsAnyScorer(keywords=["4", "four", "def"]),
        ExactMatchScorer(case_sensitive=False),
    ],
)
