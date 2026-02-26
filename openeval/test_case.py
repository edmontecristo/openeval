"""
openeval/test_case.py — TestCase data model.

The TestCase is the fundamental unit of evaluation in OpenEval.
It represents a single evaluation point with input, output, and optional context.
"""

import uuid
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator


class TestCase(BaseModel):
    """
    A single test case for LLM/agent evaluation.

    Test cases can represent simple QA pairs, RAG scenarios with context,
    or agent interactions with tool call histories.

    Attributes:
        id: Unique identifier (auto-generated UUID)
        input: The input prompt/question (required, cannot be empty)
        actual_output: The actual output from the system being evaluated
        expected_output: The ground truth expected output
        context: Optional list of context strings (e.g., RAG retrieval results)
        tools_called: Optional list of tools called by an agent
        expected_tools: Optional list of expected tools for validation
        tags: Optional list of tags for filtering/categorization
        metadata: Optional dictionary for custom metadata
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: str
    actual_output: Optional[str] = None
    expected_output: Optional[str] = None
    context: Optional[List[str]] = None
    tools_called: Optional[List[str]] = None
    expected_tools: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("input")
    @classmethod
    def input_not_empty(cls, v: str) -> str:
        """
        Validate that input is not empty or whitespace-only.

        Args:
            v: The input value to validate

        Returns:
            The validated input value

        Raises:
            ValueError: If input is empty or contains only whitespace
        """
        if not v or not v.strip():
            raise ValueError("input cannot be empty")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the test case to a dictionary.

        Returns:
            Dictionary representation of the test case
        """
        return self.model_dump()

    def to_json(self) -> str:
        """
        Convert the test case to a JSON string.

        Returns:
            JSON string representation of the test case
        """
        return self.model_dump_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """
        Create a TestCase from a dictionary.

        Args:
            data: Dictionary containing test case data

        Returns:
            A new TestCase instance
        """
        return cls(**data)
