"""
test_integration_ollama.py — Ollama-specific integration tests.

Tests that specifically verify Ollama works as an OpenEval backend.
Ollama is the recommended free, local backend for development and CI.

Run with: pytest tests/test_integration_ollama.py -m integration -v
Requires: Ollama running at localhost:11434
"""

import pytest
import httpx

pytestmark = pytest.mark.integration


def _get_any_ollama_model() -> str:
    """Return the full model tag of the first available Ollama CHAT model."""
    r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
    models = r.json().get("models", [])
    if not models:
        pytest.skip("No Ollama models pulled (run: ollama pull tinyllama)")
    # Skip embedding-only models — they don't support chat completions
    chat_models = [m["name"] for m in models if "embed" not in m["name"].lower()]
    if not chat_models:
        pytest.skip("No Ollama chat models available (only embedding models found)")
    return chat_models[0]


class TestOllamaConnectivity:
    """Verify Ollama server is reachable and has models."""

    def test_ollama_server_responds(self, ollama_available):
        """Can we reach the Ollama server at all?"""
        if not ollama_available:
            pytest.skip("Ollama not running")
        assert ollama_available is True

    def test_ollama_has_models(self, ollama_available):
        """Does Ollama have at least one model pulled?"""
        if not ollama_available:
            pytest.skip("Ollama not running")

        r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        models = r.json().get("models", [])
        assert len(models) > 0, (
            "Ollama is running but has NO models. Run: ollama pull tinyllama"
        )
        print(f"\nAvailable Ollama models: {[m['name'] for m in models]}")

    def test_ollama_chat_completions_work(self, real_ollama_client):
        """Test raw chat completion through Ollama's OpenAI-compatible API."""
        model = _get_any_ollama_model()
        response = real_ollama_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            temperature=0,
        )
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        print(f"\nModel {model} responded: {response.choices[0].message.content[:80]}")


class TestOllamaLLMJudge:
    """LLMJudge using Ollama as the backend."""

    def test_ollama_judge_basic(self, real_ollama_client):
        """Ollama should work as an LLM judge."""
        from openeval.scorers.llm_judge import LLMJudgeScorer
        from openeval.test_case import TestCase

        model = _get_any_ollama_model()

        scorer = LLMJudgeScorer(
            name="OllamaJudge",
            criteria="Is the answer correct?",
            client=real_ollama_client,
            model=model,
        )
        tc = TestCase(
            input="What color is the sky on a clear day?",
            actual_output="The sky is blue.",
            expected_output="Blue",
        )
        result = scorer.score(tc)

        print(f"\nModel: {model} | Score: {result.value} | Reason: {result.reason[:80]}")
        assert result is not None
        assert 0.0 <= result.value <= 1.0
        assert result.reason is not None
        assert result.value > 0.3, f"'Sky is blue' scored only {result.value}"


class TestOllamaEvalEndToEnd:
    """Full Eval() using Ollama — the whole pipeline with a free backend."""

    def test_full_eval_ollama(self, real_ollama_client):
        """End-to-end eval using Ollama as the LLM backend."""
        from openeval import Eval
        from openeval.scorers.llm_judge import LLMJudgeScorer
        from openeval.scorers.exact_match import ExactMatchScorer

        model = _get_any_ollama_model()

        result = Eval(
            name="ollama-integration-test",
            data=[
                {"input": "2+2?", "expected_output": "4"},
                {"input": "Capital of Germany?", "expected_output": "Berlin"},
            ],
            task=lambda input: {
                "2+2?": "4",
                "Capital of Germany?": "Berlin is the capital of Germany.",
            }.get(input, "unknown"),
            scorers=[
                ExactMatchScorer(),
                LLMJudgeScorer(
                    name="Correctness",
                    criteria="Is the answer factually correct?",
                    client=real_ollama_client,
                    model=model,
                ),
            ],
        )

        assert result is not None
        assert len(result.results) == 2

        for r in result.results:
            assert "Correctness" in r.scores
            score = r.scores["Correctness"].value
            print(f"\n  Input: {r.input!r} → Correctness: {score:.2f}")
            assert 0.0 <= score <= 1.0
