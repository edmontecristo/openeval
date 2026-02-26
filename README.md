<pre align="center">
 ██████╗ ██████╗ ███████╗███╗   ██╗███████╗██╗   ██╗ █████╗ ██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██╔════╝██║   ██║██╔══██╗██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║█████╗  ██║   ██║███████║██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══╝  ╚██╗ ██╔╝██╔══██║██║
╚██████╔╝██║     ███████╗██║ ╚████║███████╗ ╚████╔╝ ██║  ██║███████╗
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
</pre>

<p align="center">
  <em>AI Agent Evaluation Framework — test agents like you test software</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/openeval-cli/">
    <img src="https://img.shields.io/badge/pip-openeval--cli-0C55D6?logo=pypi&logoColor=white" alt="PyPI">
  </a>
  <a href="https://github.com/edmontecristo/openeval/actions">
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
    For coding agents, RAG systems, and multi-step tool users
  </sub>
</p>

---

## Why OpenEval?

**AI agents are non-deterministic software.** They can call tools in the wrong order, hallucinate facts, or lose context across multi-step tasks. You can't just `assertEqual` — you need specialized evaluation.

OpenEval gives you:

- **🤖 Agent-first** — Built for tool-coding, multi-step reasoning, and stateful agents
- **🔧 7 built-in scorers** — Tool correctness, hallucination detection, semantic similarity, LLM-as-a-Judge
- **💻 CLI-first** — `openeval run eval.py` with beautiful terminal output
- **🚦 CI/CD native** — `--fail-under 0.8` breaks your build on quality drops
- **📊 Cost tracking** — Know exactly what each eval run costs
- **🏠 100% self-hosted** — Works with Ollama for $0 local evals
- **🔓 Zero vendor lock-in** — Your data stays on your machine

---

## Quick Start

```bash
pip install openeval-cli
```

Create `eval.py` to test a **coding agent**:

```python
from openai import OpenAI
from openeval import Eval
from openeval.scorers import ToolCorrectnessScorer, ContainsAnyScorer

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def coding_agent(task: str) -> str:
    """Agent that reads files, edits code, and runs tests"""
    # Your agent implementation here
    # Returns: JSON with tools_called and final_output
    return '{"tools_called": ["read_file", "edit_file", "run_tests"], "output": "Tests passed"}'

result = Eval(
    name="coding-agent-eval",
    data=[
        {
            "input": "Fix the failing test in test_calculator.py",
            "expected_tools": ["read_file", "edit_file", "run_tests"],
            "actual_output": '{"tools_called": ["read_file", "edit_file", "run_tests"], "output": "Tests passed"}',
        },
        {
            "input": "Add a division function to calculator.py",
            "expected_tools": ["read_file", "edit_file"],
            "actual_output": '{"tools_called": ["read_file", "edit_file"], "output": "Added divide()"}',
        },
    ],
    task=coding_agent,
    scorers=[
        ToolCorrectnessScorer(),  # Did agent call the right tools in order?
        ContainsAnyScorer(keywords=["Tests passed", "divide"]),  # Did it solve the task?
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
┃  Experiment: coding-agent-eval         ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Scorer            │ Mean    │ Pass Rate   │
├───────────────────┼─────────┼─────────────┤
│ ToolCorrectness   │ 1.0000  │ 100%        │
│ ContainsAny       │ 1.0000  │ 100%        │
├───────────────────┴─────────┴─────────────┤
│ Duration: 2.3s                             │
│ Cost: $0.00000 (Ollama, free)             │
└────────────────────────────────────────────┘
```

---

## Agent Evaluation

OpenEval specializes in testing **agentic behavior** that traditional unit tests miss:

### 🛠️ **Tool Calling**
```python
from openeval.scorers import ToolCorrectnessScorer

ToolCorrectnessScorer()
# Checks: Did the agent call the right tools, in the right order?
# Use case: Coding agents, data analysis agents, research assistants
```

### 🧠 **Hallucination Detection**
```python
from openeval.scorers import FaithfulnessScorer

FaithfulnessScorer(client=client)
# Checks: Is the agent's output grounded in the provided context?
# Use case: RAG systems, knowledge bases, documentation agents
```

### 🎯 **Custom Criteria**
```python
from openeval.scorers import LLMJudgeScorer

LLMJudgeScorer(
    criteria="Did the agent follow the user's instructions exactly?",
    client=client,
)
# Use case: Instruction following, format compliance, tone checks
```

### 📏 **Semantic Similarity**
```python
from openeval.scorers import SimilarityScorer

SimilarityScorer(client=client)
# Checks: Is the output semantically similar to the expected answer?
# Use case: Open-ended tasks, multiple valid solutions
```

---

## CLI Usage

```bash
# Basic run
openeval run eval.py

# Generate HTML report
openeval run eval.py --report results.html

# Fail CI if scores below threshold (blocks PR on quality drop)
openeval run eval.py --fail-under 0.8

# Run with Ollama (free, local)
# Just set OPENAI_BASE_URL=http://localhost:11434/v1
openeval run eval.py

# Compare two experiments
openeval compare experiment_a.json experiment_b.json
```

---

## All Scorers

| Scorer | Type | What it checks | Best for |
|---|---|---|---|
| `ToolCorrectnessScorer` | Deterministic | Did agent call right tools in order? | **Coding agents, multi-step agents** |
| `FaithfulnessScorer` | LLM-as-a-Judge | Is output grounded in context? | **RAG systems, hallucination detection** |
| `LLMJudgeScorer` | LLM-as-a-Judge | Custom criteria evaluated by LLM | **Instruction following, quality checks** |
| `SimilarityScorer` | Embedding | Cosine similarity via embeddings | Open-ended tasks, semantic match |
| `ExactMatchScorer` | Deterministic | Output matches expected exactly | Structured outputs, IDs, codes |
| `ContainsAnyScorer` | Deterministic | Output contains at least one keyword | **Keyword presence checks** |
| `ContainsAllScorer` | Deterministic | Output contains all keywords | Must-have requirements |

**Custom scorers:**

```python
from openeval.scorers.base import FunctionScorer

# Define your own evaluation logic
code_quality_scorer = FunctionScorer(
    name="CodeQuality",
    fn=lambda tc: 1.0 if "def " in tc.actual_output and "import " in tc.actual_output else 0.5,
)
```

---

## CI/CD Integration

Block PRs that degrade agent performance:

```yaml
# .github/workflows/agent-quality.yml
name: Agent Quality Gate
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install openeval-cli
      - run: openeval run tests/eval_coding_agent.py --fail-under 0.8
```

Exit code 1 when quality drops → PR blocked. Ship better agents with confidence.

---

## Why NOT DeepEval / AgentOps / Braintrust?

| | OpenEval | DeepEval | AgentOps | Braintrust |
|---|---|---|---|---|
| **Agent-first** | ✅ Built for tool-calling agents | ❌ General LLM testing | ⚠️ Monitoring only | ❌ General LLM testing |
| **Price** | ✅ Free forever | Freemium | Freemium | $249/mo |
| **CLI-first** | ✅ Native | ❌ Library-only | ❌ Dashboard-first | ❌ Web-only |
| **Self-contained HTML** | ✅ No server needed | ❌ Requires platform | ❌ Requires app | ❌ Web-only |
| **CI/CD native** | ✅ Exit codes | ⚠️ Manual | ⚠️ Manual | ❌ No |
| **Local LLM support** | ✅ Ollama | ❌ OpenAI only | ⚠️ Partial | ❌ No |
| **Philosophy** | Tool you own | Framework | Platform | SaaS |
| **Best for** | **Agent dev & CI** | Research evals | Production monitoring | Teams |

**OpenEval is a tool, not a platform.** You own your data, you run it where you want.

---

## Datasets

```python
from openeval.dataset import Dataset

# Load from file
ds = Dataset.from_csv("test_cases.csv")
ds = Dataset.from_json("test_cases.json")

# Filter and sample
ds_hard = ds.filter(tags=["hard", "edge-case"])
ds_sample = ds.sample(50)

# Version control your test cases
ds.save("test_cases_v2.json")
```

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
├── test_case.py         # TestCase data model with tools_called support
├── types.py             # ScoreResult, ExperimentResult
├── dataset.py           # Dataset loading and filtering
├── tracing.py           # @trace decorator for agent debugging
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
    └── tool_correctness.py  # ✨ Agent tool calling validation
```

---

## Development

```bash
git clone https://github.com/edmontecristo/openeval.git
cd openeval
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT © OpenEval Contributors

---

**Built for developers who ship AI agents.**
