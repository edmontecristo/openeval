"""
tracing.py — Observability through automatic span recording.

The @trace decorator captures function inputs, outputs, duration,
and errors for production monitoring and debugging.
"""

import time
import uuid
import functools
import threading
import inspect
from datetime import datetime, timezone
from typing import Optional, Any, List, Dict
from pydantic import BaseModel, Field


# Thread-local storage for collecting traces
_trace_storage = threading.local()


def get_current_traces() -> List["Span"]:
    """Get all traces collected in the current thread."""
    if not hasattr(_trace_storage, "traces"):
        _trace_storage.traces = []
    return _trace_storage.traces


def _clear_traces():
    """Clear all traces in the current thread (for testing)."""
    _trace_storage.traces = []


class Span(BaseModel):
    """A single span representing a function execution."""
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    input: Any
    output: Any
    duration_ms: float
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    parent_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert span to dictionary representation."""
        return self.model_dump()


def trace(fn):
    """Decorator that captures function input/output/duration as a Span.

    Usage:
        @trace
        def my_llm_call(query: str) -> str:
            return openai.chat(query)

    Key behaviors:
    - Function still works normally (returns same value)
    - Captures args as input dict, return value as output
    - Measures duration in ms
    - If function raises, captures error string but re-raises
    - Nested @trace creates multiple spans in the list
    - All spans stored in thread-local storage via get_current_traces()
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Build input dict from kwargs and positional args
        input_data: Dict[str, Any] = kwargs.copy()

        # Add positional args by parameter name
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        for i, arg in enumerate(args):
            if i < len(params):
                input_data[params[i]] = arg

        start = time.time()
        error = None
        output = None

        try:
            output = fn(*args, **kwargs)
            return output
        except Exception as e:
            error = str(e)
            raise
        finally:
            duration = (time.time() - start) * 1000
            span = Span(
                name=fn.__name__,
                input=input_data,
                output=output,
                duration_ms=duration,
                error=error,
            )
            get_current_traces().append(span)

    return wrapper


class TraceSession(BaseModel):
    """A collection of spans representing a trace session."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    spans: List[Span] = Field(default_factory=list)

    def add_span(self, span: Span):
        """Add a span to this session."""
        self.spans.append(span)

    @property
    def total_duration_ms(self) -> float:
        """Total duration of all spans in this session."""
        return sum(s.duration_ms for s in self.spans)

    def to_dict(self) -> dict:
        """Convert session to dictionary representation."""
        return self.model_dump()
