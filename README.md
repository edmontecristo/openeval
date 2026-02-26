# 🔬 OpenEval

**Open-source LLM evaluation framework. Evaluate, compare, and ship better AI apps.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Why OpenEval?

LLM outputs are non-deterministic. You can't just `assertEqual`. You need specialized scorers that understand semantics, faithfulness, and tool usage.

OpenEval gives you:

- **7 built-in scorers** — from exact match to LLM-as-a-Judge
- **CLI-first** — run evals from terminal, get rich tables
- **CI/CD native** — `--fail-under 0.8` breaks your build on quality drops
- **Self-contained HTML reports** — share results without a server
- **Cost tracking** — know exactly how much each eval costs in tokens and USD
- **100% self-hosted** — no SaaS, no data leaves your machine

## Quick Start

### Install

```bash
pip install openeval
```

### Write your first eval

```python
from openeval import Eval
from openeval.scorers.exact_match import ExactMatchScorer
from openeval.scorers.contains import ContainsAnyScorer

result = Eval(
    name="my-chatbot-v2",
    data=[
        {"input": "What are your hours?", "expected_output": "9am to 5pm"},
        {"input": "Return policy?", "expected_output": "30-day refund"},
    ],
    task=lambda input: my_chatbot(input),  # your LLM app
    scorers=[
        ExactMatchScorer(),
        ContainsAnyScorer(keywords=["hours", "refund", "policy"]),
    ],
)

print(f"Score: {result.summary}")
```

### Run from CLI

```bash
openeval run my_eval.py
openeval run my_eval.py --report report.html
openeval run my_eval.py --fail-under 0.8  # exit code 1 if below threshold
```

## Scorers

| Scorer | Type | What it checks |
| --- | --- | --- |
| `ExactMatchScorer` | Deterministic | Output matches expected exactly |
| `ContainsAnyScorer` | Deterministic | Output contains at least one keyword |
| `ContainsAllScorer` | Deterministic | Output contains all keywords |
| `SimilarityScorer` | Embedding | Cosine similarity via OpenAI embeddings |
| `LLMJudgeScorer` | LLM-as-a-Judge | Custom criteria evaluated by GPT |
| `FaithfulnessScorer` | LLM-as-a-Judge | Is output grounded in context? (hallucination detection) |
| `ToolCorrectnessScorer` | Deterministic | Did the agent call the right tools? |

### Custom scorers

```python
from openeval.scorers.base import FunctionScorer

length_scorer = FunctionScorer(
    name="OutputLength",
    fn=lambda tc: min(len(tc.actual_output) / 100, 1.0),
)
```

## Features

### Datasets

```python
from openeval.dataset import Dataset

ds = Dataset.from_csv("test_cases.csv")
ds = Dataset.from_json("test_cases.json")
ds_easy = ds.filter(tags=["easy"])
ds_sample = ds.sample(50)
```

### Tracing

```python
from openeval.tracing import trace

@trace
def my_llm_call(query: str) -> str:
    return openai.chat(query)

# Captures input, output, duration, and errors automatically
```

### Cost Tracking

```python
# Costs tracked automatically for LLM scorers
print(f"Total cost: ${result.total_cost_usd:.6f}")
print(f"Total tokens: {result.summary['total_tokens']}")
```

### HTML Reports

```bash
openeval run eval.py --report results.html
```

Generates a self-contained HTML file with scores, reasons, and comparisons. No server needed — just open in a browser.

### CI/CD Integration

```yaml
# GitHub Actions
- name: Run LLM eval
  run: openeval run tests/eval_chatbot.py --fail-under 0.8
```

Exit code 1 when quality drops below threshold. Your PR won't merge if the chatbot gets worse.

## Project Structure

```
openeval/
├── eval.py              # Eval() orchestrator
├── test_case.py         # TestCase data model
├── types.py             # ScoreResult, ExperimentResult
├── dataset.py           # Dataset loading and filtering
├── tracing.py           # @trace decorator
├── cost.py              # Token and cost tracking
├── report.py            # HTML report generator
├── cli.py               # CLI interface
└── scorers/
    ├── base.py           # BaseScorer interface
    ├── exact_match.py
    ├── contains.py
    ├── similarity.py
    ├── llm_judge.py
    ├── faithfulness.py
    └── tool_correctness.py
```

## Development

```bash
git clone https://github.com/yourusername/openeval.git
cd openeval
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
