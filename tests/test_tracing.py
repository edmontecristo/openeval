"""
test_tracing.py — Tests for the @trace decorator and span collection.

Tracing records what happens inside LLM calls for observability.
The @trace decorator should wrap functions transparently.
"""

import pytest
import time


class TestTraceDecorator:
    """The @trace decorator wraps functions to capture execution data."""

    def test_decorated_function_still_works(self):
        """@trace should not change function behavior."""
        from openeval.tracing import trace

        @trace
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        result = greet("World")
        assert result == "Hello, World!"

    def test_captures_input_output(self):
        """Should record what went in and what came out."""
        from openeval.tracing import trace, get_current_traces

        @trace
        def add(a: int, b: int) -> int:
            return a + b

        add(2, 3)

        traces = get_current_traces()
        assert len(traces) >= 1
        last_trace = traces[-1]
        assert last_trace.input == {"a": 2, "b": 3}
        assert last_trace.output == 5

    def test_captures_duration(self):
        """Should measure how long the function took."""
        from openeval.tracing import trace, get_current_traces

        @trace
        def slow_fn():
            time.sleep(0.1)
            return "done"

        slow_fn()

        traces = get_current_traces()
        assert traces[-1].duration_ms >= 90  # ~100ms

    def test_captures_exceptions(self):
        """If function throws, trace should capture the error."""
        from openeval.tracing import trace, get_current_traces

        @trace
        def buggy():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            buggy()

        traces = get_current_traces()
        assert traces[-1].error is not None
        assert "oops" in traces[-1].error

    def test_nested_traces(self):
        """Nested @trace calls should create parent-child spans."""
        from openeval.tracing import trace, get_current_traces

        @trace
        def inner():
            return "inner result"

        @trace
        def outer():
            return inner()

        outer()

        traces = get_current_traces()
        # Should have 2 spans — outer is parent, inner is child
        assert len(traces) >= 2


class TestSpanModel:
    """Span is a single unit of traced execution."""

    def test_span_has_required_fields(self):
        from openeval.tracing import Span

        span = Span(
            name="test_fn",
            input={"query": "hello"},
            output="world",
            duration_ms=150,
        )
        assert span.name == "test_fn"
        assert span.span_id is not None
        assert span.timestamp is not None

    def test_span_serializes_to_dict(self):
        from openeval.tracing import Span

        span = Span(name="test", input={}, output="x", duration_ms=10)
        d = span.to_dict()
        assert "name" in d
        assert "span_id" in d
        assert "timestamp" in d
        assert "duration_ms" in d


class TestTraceSession:
    """A session groups multiple traces from one evaluation run."""

    def test_create_session(self):
        from openeval.tracing import TraceSession

        session = TraceSession(name="eval-run-1")
        assert session.name == "eval-run-1"
        assert session.session_id is not None

    def test_session_collects_spans(self):
        from openeval.tracing import TraceSession, Span

        session = TraceSession(name="test")
        session.add_span(
            Span(name="step1", input={}, output="a", duration_ms=10)
        )
        session.add_span(
            Span(name="step2", input={}, output="b", duration_ms=20)
        )
        assert len(session.spans) == 2

    def test_session_computes_total_duration(self):
        from openeval.tracing import TraceSession, Span

        session = TraceSession(name="test")
        session.add_span(
            Span(name="s1", input={}, output="a", duration_ms=100)
        )
        session.add_span(
            Span(name="s2", input={}, output="b", duration_ms=200)
        )
        assert session.total_duration_ms == 300

    def test_session_exports_to_dict(self):
        from openeval.tracing import TraceSession

        session = TraceSession(name="test")
        d = session.to_dict()
        assert "session_id" in d
        assert "name" in d
        assert "spans" in d
