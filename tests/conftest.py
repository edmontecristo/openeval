"""
conftest.py — Shared fixtures for OpenEval test suite.

These fixtures simulate real-world LLM evaluation scenarios:
- Customer support chatbot
- RAG pipeline responses
- Multi-turn agent interactions
- Tool-calling agents
"""

import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch


# ─── Fixtures: Test Cases ─────────────────────────────────────────


@pytest.fixture
def simple_test_case():
    """A basic QA test case — customer asking about returns."""
    from openeval.test_case import TestCase

    return TestCase(
        input="What is your return policy?",
        actual_output="You can return items within 30 days for a full refund.",
        expected_output="We offer a 30-day full refund policy at no extra cost.",
    )


@pytest.fixture
def rag_test_case():
    """RAG test case — answer grounded in retrieved context."""
    from openeval.test_case import TestCase

    return TestCase(
        input="How much does the Pro plan cost?",
        actual_output="The Pro plan costs $49 per month with unlimited features.",
        expected_output="The Pro plan is $49/month and includes unlimited projects.",
        context=[
            "Pro Plan: $49/month. Includes unlimited projects, priority support.",
            "Free Plan: $0/month. Limited to 3 projects.",
            "Enterprise: Custom pricing. Includes SSO and dedicated support.",
        ],
    )


@pytest.fixture
def hallucination_test_case():
    """Test case where LLM hallucinates — answer NOT in context."""
    from openeval.test_case import TestCase

    return TestCase(
        input="Does the Pro plan include phone support?",
        actual_output="Yes, the Pro plan includes 24/7 phone support and a personal account manager.",
        expected_output="The Pro plan includes priority support but does not mention phone support.",
        context=[
            "Pro Plan: $49/month. Includes unlimited projects, priority support.",
            "Enterprise: Custom pricing. Includes SSO and dedicated support.",
        ],
    )


@pytest.fixture
def agent_test_case():
    """Agent test case — agent should use specific tools in order."""
    from openeval.test_case import TestCase

    return TestCase(
        input="Book a flight from NYC to London for next Friday",
        actual_output="I've booked flight BA178 departing JFK at 7pm on March 7th.",
        expected_output="Flight booked successfully.",
        tools_called=["search_flights", "check_availability", "book_flight"],
        expected_tools=["search_flights", "check_availability", "book_flight"],
    )


@pytest.fixture
def conversation_test_cases():
    """Multi-turn conversation — tests coherence across turns."""
    from openeval.test_case import TestCase

    return [
        TestCase(
            input="Hi, I need help with my order #12345",
            actual_output="I'd be happy to help with order #12345. What's the issue?",
            expected_output="Acknowledge order and ask for details.",
        ),
        TestCase(
            input="The package arrived damaged",
            actual_output="I'm sorry to hear that. We can send a replacement or issue a refund. Which would you prefer?",
            expected_output="Offer replacement or refund.",
        ),
        TestCase(
            input="I'd like a refund please",
            actual_output="I've initiated a refund of $89.99 to your original payment method. It should appear in 3-5 business days.",
            expected_output="Process refund and give timeline.",
        ),
    ]


@pytest.fixture
def dataset_dicts():
    """Raw dataset in dict format — like CSV/JSON import."""
    return [
        {
            "input": "What are your hours?",
            "expected_output": "We are open Monday-Friday 9am-5pm.",
        },
        {
            "input": "Where are you located?",
            "expected_output": "We are located at 123 Main St, San Francisco.",
        },
        {
            "input": "Do you offer free shipping?",
            "expected_output": "Free shipping on orders over $50.",
        },
        {
            "input": "How do I cancel my subscription?",
            "expected_output": "Go to Settings > Subscription > Cancel.",
        },
        {
            "input": "What payment methods do you accept?",
            "expected_output": "We accept Visa, Mastercard, and PayPal.",
        },
    ]


# ─── Fixtures: Mock LLM ──────────────────────────────────────────


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client that returns predictable responses."""
    client = MagicMock()

    # Mock chat completion
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"score": 0.85, "reason": "Output is mostly correct with minor differences in wording."}'
    mock_response.usage.prompt_tokens = 150
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 200
    mock_response.model = "gpt-4o-mini"

    client.chat.completions.create.return_value = mock_response

    # Mock embeddings
    mock_embedding = MagicMock()
    mock_embedding.data = [MagicMock()]
    mock_embedding.data[0].embedding = [0.1] * 1536  # OpenAI ada-002 dimension
    mock_embedding.usage.total_tokens = 10

    client.embeddings.create.return_value = mock_embedding

    return client


@pytest.fixture
def mock_openai_client_low_score():
    """Mock OpenAI client returning LOW scores (for failure tests)."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"score": 0.15, "reason": "Output contains fabricated information not present in the context."}'
    mock_response.usage.prompt_tokens = 150
    mock_response.usage.completion_tokens = 50
    mock_response.usage.total_tokens = 200
    mock_response.model = "gpt-4o-mini"
    client.chat.completions.create.return_value = mock_response
    return client


# ─── Fixtures: Temp dirs ─────────────────────────────────────────


@pytest.fixture
def tmp_report_dir(tmp_path):
    """Temporary directory for HTML report output."""
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    return report_dir


@pytest.fixture
def tmp_dataset_dir(tmp_path):
    """Temporary directory with sample dataset files."""
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()

    # Create CSV dataset
    csv_file = dataset_dir / "test_data.csv"
    csv_file.write_text(
        "input,expected_output\n"
        '"What is AI?","Artificial Intelligence is the simulation of human intelligence by machines."\n'
        '"What is ML?","Machine Learning is a subset of AI that learns from data."\n'
    )

    # Create JSON dataset
    import json

    json_file = dataset_dir / "test_data.json"
    json_file.write_text(
        json.dumps(
            [
                {
                    "input": "What is Python?",
                    "expected_output": "Python is a programming language.",
                },
                {
                    "input": "What is JavaScript?",
                    "expected_output": "JavaScript is a web programming language.",
                },
            ]
        )
    )

    return dataset_dir
