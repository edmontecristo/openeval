"""
test_dataset.py — Tests for Dataset management.

Datasets are collections of test cases used for repeatable evaluations.
Must support: create, load from CSV/JSON, versioning, filtering.
"""

import pytest
import json


class TestDatasetCreation:
    """Creating and populating datasets."""

    def test_create_empty_dataset(self):
        from openeval.dataset import Dataset

        ds = Dataset(name="my-dataset")
        assert ds.name == "my-dataset"
        assert len(ds) == 0

    def test_add_items(self, dataset_dicts):
        from openeval.dataset import Dataset

        ds = Dataset(name="test")
        for item in dataset_dicts:
            ds.add(item)
        assert len(ds) == 5

    def test_iterate_over_dataset(self, dataset_dicts):
        """Should be iterable for use in Eval()."""
        from openeval.dataset import Dataset

        ds = Dataset(name="test")
        for item in dataset_dicts:
            ds.add(item)

        items = list(ds)
        assert len(items) == 5
        assert items[0]["input"] == "What are your hours?"

    def test_dataset_deduplication(self):
        """Adding same item twice should not duplicate."""
        from openeval.dataset import Dataset

        ds = Dataset(name="test")
        item = {"input": "hello", "expected_output": "world"}
        ds.add(item)
        ds.add(item)
        assert len(ds) == 1


class TestDatasetIO:
    """Loading and saving datasets from/to files."""

    def test_load_from_csv(self, tmp_dataset_dir):
        from openeval.dataset import Dataset

        ds = Dataset.from_csv(tmp_dataset_dir / "test_data.csv")
        assert len(ds) == 2
        assert ds[0]["input"] == "What is AI?"

    def test_load_from_json(self, tmp_dataset_dir):
        from openeval.dataset import Dataset

        ds = Dataset.from_json(tmp_dataset_dir / "test_data.json")
        assert len(ds) == 2
        assert ds[0]["input"] == "What is Python?"

    def test_save_to_json(self, tmp_path, dataset_dicts):
        from openeval.dataset import Dataset

        ds = Dataset(name="test")
        for item in dataset_dicts:
            ds.add(item)

        output_path = tmp_path / "output.json"
        ds.save_json(output_path)

        # Verify saved file
        with open(output_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 5

    def test_load_nonexistent_file_raises(self):
        from openeval.dataset import Dataset

        with pytest.raises((FileNotFoundError, Exception)):
            Dataset.from_csv("/nonexistent/path.csv")


class TestDatasetFiltering:
    """Filter datasets by tags, metadata, etc."""

    def test_filter_by_tag(self):
        from openeval.dataset import Dataset

        ds = Dataset(name="test")
        ds.add({"input": "q1", "expected_output": "a1", "tags": ["easy"]})
        ds.add({"input": "q2", "expected_output": "a2", "tags": ["hard"]})
        ds.add({"input": "q3", "expected_output": "a3", "tags": ["easy"]})

        easy = ds.filter(tags=["easy"])
        assert len(easy) == 2

    def test_sample_n_items(self):
        """Sample N random items from dataset."""
        from openeval.dataset import Dataset

        ds = Dataset(name="test")
        for i in range(100):
            ds.add({"input": f"q{i}", "expected_output": f"a{i}"})

        sample = ds.sample(10)
        assert len(sample) == 10
