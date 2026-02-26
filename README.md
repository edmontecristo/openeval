<pre align="center">
 ██████╗ ██████╗ ███████╗███╗   ██╗███████╗██╗   ██╗ █████╗ ██╗     
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║   ██║██╔══██╗██║     
██║   ██║██████╔╝█████╗  ██╔██╗ ██║█████╗  ██║   ██║███████║██║     
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══╝  ╚██╗ ██╔╝██╔══██║██║     
╚██████╔╝██║     ███████╗██║ ╚████║███████╗ ╚████╔╝ ██║  ██║███████╗
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
</pre>

<p align="center">
  <em>CLI-first LLM evaluation — like Pytest for AI agents</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/openeval/">
    <img src="https://img.shields.io/badge/pip-openeval-0C55D6?logo=pypi&logoColor=white" alt="PyPI">
  </a>
  <a href="https://github.com/YOUR/openeval/actions">
    <img src="https://img.shields.io/badge/tests-117%20passing-brightgreen" alt="Tests">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  </a>
  <a href="https://python.org">
    <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  </a>
</p>

<p align="center">
  <sub>
    <a href="https://github.com/confident-ai/deepeval">DeepEval</a> ×
    <a href="https://www.braintrust.dev">Braintrust</a> —
    but CLI-first, self-hosted, and free forever
  </sub>
</p>

---

## Why OpenEval?

**LLM outputs are non-deterministic.** You can't just `assertEqual`. You need specialized scorers that understand semantics, faithfulness, and tool usage.

OpenEval gives you:

- **7 built-in scorers** — from exact match to LLM-as-a-Judge
- **CLI-first** — `openeval run eval.py` with beautiful terminal output
- **CI/CD native** — `--fail-under 0.8` breaks your build on quality drops
- **Self-contained HTML reports** — share results without a server
- **Cost tracking** — know exactly how much each eval costs
- **100% self-hosted** — works with Ollama for $0 local evals
- **Zero vendor lock-in** — your data stays on your machine

---

## Quick Start

```bash
pip install openeval
```

Create `eval.py`:

```python
from openai import OpenAI
from openeval import Eval
from openeval.scorers import ContainsAnyScorer, FaithfulnessScorer

client = OpenAI()

def my_agent(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )
    return response.choices[0].message.content

result = Eval(
    name="my-eval",
    data=[
        {"input": "What is 2+2?", "expected_output": "4"},
        {"input": "Return policy?", "expected_output": "30 days", "context": ["30-day refund policy"]},
    ],
    task=my_agent,
    scorers=[
        ContainsAnyScorer(keywords=["4", "four"]),
        FaithfulnessScorer(client=client),
    ],
)
```

Run it:

```bash
openeval run eval.py
```

**Output:**

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  Experiment: my-eval                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Scorer       │ Mean    │ Pass Rate   │
├──────────────┼─────────┼─────────────┤
│ ContainsAny  │ 1.0000  │ 100%        │
│ Faithfulness │ 0.9500  │ 100%        │
├──────────────┴─────────┴─────────────┤
│ Duration: 2.3s                         │
│ Cost: $0.00045                         │
└────────────────────────────────────────┘
```

---

## Why NOT DeepEval / AgentOps / Braintrust?

| | OpenEval | DeepEval | AgentOps | Braintrust |
|---|---|---|---|---|
| **Price** | ✅ Free forever | Freemium | Freemium | $249/mo |
| **CLI-first** | ✅ Native | ❌ Library-only | ❌ Dashboard-first | ❌ Web-only |
| **Self-contained HTML** | ✅ No server needed | ❌ Requires platform | ❌ Requires app | ❌ Web-only |
| **CI/CD native** | ✅ Exit codes | ⚠️ Manual | ⚠️ Manual | ❌ No |
| **Local LLM support** | ✅ Ollama | ❌ OpenAI only | ⚠️ Partial | ❌ No |
| **Philosophy** | Tool you own | Framework | Platform | SaaS |
| **Best for** | CI/CD quality gates | Research evals | Production monitoring | Teams |

**OpenEval is a tool, not a platform.** You own your data, you run it where you want.

---

## CLI Usage

```bash
# Basic run
openeval run eval.py

# Generate HTML report
openeval run eval.py --report results.html

# Fail CI if scores below threshold
openeval run eval.py --fail-under 0.8

# Run with Ollama (free, local)
# Just set OPENAI_BASE_URL=http://localhost:11434/v1
```

---

## Scorers

| Scorer | Type | What it checks |
|---|---|---|
| `ExactMatchScorer` | Deterministic | Output matches expected exactly |
| `ContainsAnyScorer` | Deterministic | Output contains at least one keyword |
| `ContainsAllScorer` | Deterministic | Output contains all keywords |
| `SimilarityScorer` | Embedding | Cosine similarity via embeddings |
| `LLMJudgeScorer` | LLM-as-a-Judge | Custom criteria evaluated by LLM |
| `FaithfulnessScorer` | LLM-as-a-Judge | Is output grounded in context? (hallucination detection) |
| `ToolCorrectnessScorer` | Deterministic | Did the agent call the right tools? |

**Custom scorers:**

```python
from openeval.scorers.base import FunctionScorer

length_scorer = FunctionScorer(
    name="OutputLength",
    fn=lambda tc: min(len(tc.actual_output) / 100, 1.0),
)
```

---

## Datasets

```python
from openeval.dataset import Dataset

# Load from file
ds = Dataset.from_csv("test_cases.csv")
ds = Dataset.from_json("test_cases.json")

# Filter and sample
ds_easy = ds.filter(tags=["easy"])
ds_sample = ds.sample(50)
```

---

## CI/CD Integration

```yaml
# .github/workflows/llm-eval.yml
name: LLM Quality Gate
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install openeval
      - run: openeval run tests/eval_chatbot.py --fail-under 0.8
```

Exit code 1 when quality drops → PR blocked.

---

## Cost Tracking

```python
# Costs tracked automatically
print(f"Total cost: ${result.total_cost_usd:.6f}")
print(f"Total tokens: {result.summary['total_tokens']}")

# Breakdown by scorer
for scorer_name, stats in result.summary.items():
    print(f"{scorer_name}: ${stats.get('cost_usd', 0):.6f}")
```

---

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
    ├── base.py          # BaseScorer interface
    ├── exact_match.py
    ├── contains.py
    ├── similarity.py    # Embedding-based
    ├── llm_judge.py     # LLM-as-a-Judge
    ├── faithfulness.py  # Hallucination detection
    └── tool_correctness.py
```

---

## Development

```bash
git clone https://github.com/YOUR/openeval.git
cd openeval
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT © OpenEval Contributors

---

**Built for developers who ship AI products.**
