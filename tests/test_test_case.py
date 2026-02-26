"""
test_test_case.py — Tests for TestCase data model.

A TestCase is the fundamental unit. If this breaks, everything breaks.
Real-world QA: validate data integrity, serialization, edge cases.
"""

import pytest


class TestTestCaseCreation:
    """TestCase should be easy to create and validate data."""

    def test_basic_creation(self):
        """Minimal test case — just input and output."""
        from openeval.test_case import TestCase

        tc = TestCase(
            input="Hello",
            actual_output="Hi there!",
        )
        assert tc.input == "Hello"
        assert tc.actual_output == "Hi there!"
        assert tc.expected_output is None
        assert tc.context is None

    def test_full_creation(self, simple_test_case):
        """Test case with all fields populated."""
        assert simple_test_case.input is not None
        assert simple_test_case.actual_output is not None
        assert simple_test_case.expected_output is not None

    def test_rag_creation_with_context(self, rag_test_case):
        """RAG test case must carry retrieval context."""
        assert rag_test_case.context is not None
        assert len(rag_test_case.context) == 3
        assert "Pro Plan" in rag_test_case.context[0]

    def test_agent_creation_with_tools(self, agent_test_case):
        """Agent test case carries tool call history."""
        assert agent_test_case.tools_called == [
            "search_flights",
            "check_availability",
            "book_flight",
        ]
        assert agent_test_case.expected_tools is not None


class TestTestCaseValidation:
    """TestCase should reject invalid data."""

    def test_empty_input_raises(self):
        """Cannot create test case without input."""
        from openeval.test_case import TestCase

        with pytest.raises((ValueError, Exception)):
            TestCase(input="", actual_output="something")

    def test_none_input_raises(self):
        """Cannot create test case with None input."""
        from openeval.test_case import TestCase

        with pytest.raises((ValueError, TypeError, Exception)):
            TestCase(input=None, actual_output="something")

    def test_whitespace_only_input_raises(self):
        """Whitespace-only input should be rejected."""
        from openeval.test_case import TestCase

        with pytest.raises((ValueError, Exception)):
            TestCase(input="   ", actual_output="something")


class TestTestCaseSerialization:
    """TestCase should serialize to/from dict and JSON."""

    def test_to_dict(self, simple_test_case):
        """Convert test case to dictionary."""
        d = simple_test_case.to_dict()
        assert isinstance(d, dict)
        assert d["input"] == simple_test_case.input
        assert d["actual_output"] == simple_test_case.actual_output
        assert d["expected_output"] == simple_test_case.expected_output

    def test_from_dict(self):
        """Create test case from dictionary."""
        from openeval.test_case import TestCase

        data = {
            "input": "test input",
            "actual_output": "test output",
            "expected_output": "expected",
        }
        tc = TestCase.from_dict(data)
        assert tc.input == "test input"
        assert tc.actual_output == "test output"

    def test_roundtrip_serialization(self, rag_test_case):
        """dict → TestCase → dict should preserve all data."""
        from openeval.test_case import TestCase

        d1 = rag_test_case.to_dict()
        tc = TestCase.from_dict(d1)
        d2 = tc.to_dict()
        assert d1 == d2

    def test_to_json(self, simple_test_case):
        """Should serialize to valid JSON string."""
        import json

        json_str = simple_test_case.to_json()
        parsed = json.loads(json_str)
        assert parsed["input"] == simple_test_case.input


class TestTestCaseMetadata:
    """TestCase should track metadata (timestamps, IDs, tags)."""

    def test_auto_generates_id(self, simple_test_case):
        """Each test case gets a unique ID."""
        assert hasattr(simple_test_case, "id")
        assert simple_test_case.id is not None
        assert len(simple_test_case.id) > 0

    def test_unique_ids(self):
        """Two test cases should never have the same ID."""
        from openeval.test_case import TestCase

        tc1 = TestCase(input="a", actual_output="b")
        tc2 = TestCase(input="a", actual_output="b")
        assert tc1.id != tc2.id

    def test_supports_tags(self):
        """Test case can carry custom tags for filtering."""
        from openeval.test_case import TestCase

        tc = TestCase(
            input="test",
            actual_output="output",
            tags=["production", "v2", "critical"],
        )
        assert "production" in tc.tags
        assert len(tc.tags) == 3

    def test_supports_metadata(self):
        """Test case accepts arbitrary metadata dict."""
        from openeval.test_case import TestCase

        tc = TestCase(
            input="test",
            actual_output="output",
            metadata={"model": "gpt-4o", "temperature": 0.7, "latency_ms": 450},
        )
        assert tc.metadata["model"] == "gpt-4o"
        assert tc.metadata["latency_ms"] == 450
