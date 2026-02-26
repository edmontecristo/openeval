"""
dataset.py — Dataset management for repeatable evaluations.

Provides Dataset class for loading, managing, and filtering test case data.
Supports CSV/JSON import, deduplication, tagging, and sampling.
"""

import csv
import json
import random
from pathlib import Path
from typing import Optional


class Dataset:
    """
    A collection of test cases for LLM evaluation.

    Features:
    - Load from CSV or JSON files
    - Automatic deduplication by input+expected_output
    - Filter by tags
    - Random sampling
    - Iteration support for use in Eval()

    Example:
        >>> ds = Dataset.from_csv("test_cases.csv")
        >>> easy_ds = ds.filter(tags=["easy"])
        >>> sample = ds.sample(10)
        >>> for item in ds:
        ...     print(item["input"])
    """

    def __init__(self, name: str):
        """
        Initialize an empty dataset.

        Args:
            name: Dataset identifier for logging/export
        """
        self.name = name
        self._items: list[dict] = []

    def add(self, item: dict):
        """
        Add an item to the dataset with automatic deduplication.

        Deduplication logic: If an item with identical input AND expected_output
        already exists, skip adding the new item.

        Args:
            item: Dictionary representing a test case (at minimum: input, expected_output)
        """
        # Check for duplicate by comparing input + expected_output
        for existing in self._items:
            if (
                existing.get("input") == item.get("input")
                and existing.get("expected_output") == item.get("expected_output")
            ):
                return
        self._items.append(item)

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self._items)

    def __iter__(self):
        """Make dataset iterable for use in Eval()."""
        return iter(self._items)

    def __getitem__(self, idx):
        """Support indexing: ds[0], ds[1], etc."""
        return self._items[idx]

    @classmethod
    def from_csv(cls, path: str | Path) -> "Dataset":
        """
        Load dataset from CSV file.

        CSV must have headers (e.g., input, expected_output, tags).
        First row is treated as header, all subsequent rows as data.

        Args:
            path: Path to CSV file

        Returns:
            Dataset populated with CSV rows as dicts

        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        dataset = cls(name=path.stem)
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dataset.add(dict(row))
        return dataset

    @classmethod
    def from_json(cls, path: str | Path) -> "Dataset":
        """
        Load dataset from JSON file.

        JSON should be a list of objects, where each object is a test case.

        Args:
            path: Path to JSON file

        Returns:
            Dataset populated with JSON objects

        Raises:
            FileNotFoundError: If JSON file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)

        dataset = cls(name=path.stem)
        for item in items:
            dataset.add(item)
        return dataset

    def save_json(self, path: str | Path):
        """
        Save dataset to JSON file.

        Args:
            path: Output path for JSON file
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._items, f, indent=2)

    def filter(self, tags: Optional[list[str]] = None) -> "Dataset":
        """
        Filter dataset by tags.

        Returns items where ALL specified tags are present in the item's tag list.
        If no tags specified, returns empty dataset.

        Args:
            tags: List of tags to filter by (all must be present)

        Returns:
            New Dataset containing only filtered items

        Example:
            >>> ds = Dataset(name="test")
            >>> ds.add({"input": "q1", "tags": ["easy", "math"]})
            >>> ds.add({"input": "q2", "tags": ["hard", "math"]})
            >>> easy_math = ds.filter(tags=["easy", "math"])
            >>> len(easy_math)
            1
        """
        result = Dataset(name=f"{self.name}_filtered")

        if not tags:
            return result

        for item in self._items:
            item_tags = item.get("tags", [])
            if isinstance(item_tags, list):
                # All specified tags must be present
                if all(tag in item_tags for tag in tags):
                    result.add(item)

        return result

    def sample(self, n: int) -> "Dataset":
        """
        Return a random sample of N items from the dataset.

        If n > dataset size, returns all items (no error).

        Args:
            n: Number of items to sample

        Returns:
            New Dataset containing sampled items
        """
        result = Dataset(name=f"{self.name}_sample")
        sampled = random.sample(self._items, min(n, len(self._items)))
        for item in sampled:
            result.add(item)
        return result
