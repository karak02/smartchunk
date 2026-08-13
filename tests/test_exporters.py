"""Tests for exporters."""

from __future__ import annotations

import json
from pathlib import Path

from smartchunk.exporters.json_export import JsonExporter
from smartchunk.models import SmartChunk


class TestJsonExporter:
    """Tests for JSON/JSONL export."""

    def test_to_json(self, tmp_path: Path, sample_smart_chunks: list[SmartChunk]):
        filepath = tmp_path / "output.json"
        JsonExporter.to_json(sample_smart_chunks, filepath)

        assert filepath.exists()
        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert len(data) == 3
        assert data[0]["id"] == "chunk_test_001"
        assert data[0]["summary"] == "Board approves major capital expenditure"
        assert data[0]["entities"] == ["Board of Directors", "$50M"]

    def test_to_jsonl(self, tmp_path: Path, sample_smart_chunks: list[SmartChunk]):
        filepath = tmp_path / "output.jsonl"
        JsonExporter.to_jsonl(sample_smart_chunks, filepath)

        assert filepath.exists()
        lines = filepath.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

        first = json.loads(lines[0])
        assert first["id"] == "chunk_test_001"

    def test_to_dict(self, sample_smart_chunks: list[SmartChunk]):
        dicts = JsonExporter.to_dict(sample_smart_chunks)

        assert len(dicts) == 3
        assert isinstance(dicts[0], dict)
        assert dicts[0]["text"] == "The board approved a $50M expansion plan."
        assert dicts[0]["metadata"]["source"] == "test.pdf"

    def test_to_json_creates_parent_dirs(
        self, tmp_path: Path, sample_smart_chunks: list[SmartChunk]
    ):
        filepath = tmp_path / "nested" / "deep" / "output.json"
        JsonExporter.to_json(sample_smart_chunks, filepath)
        assert filepath.exists()

    def test_to_json_empty_list(self, tmp_path: Path):
        filepath = tmp_path / "empty.json"
        JsonExporter.to_json([], filepath)

        data = json.loads(filepath.read_text(encoding="utf-8"))
        assert data == []

    def test_json_roundtrip(self, tmp_path: Path, sample_smart_chunks: list[SmartChunk]):
        """Test that JSON export can be loaded back and validated."""
        filepath = tmp_path / "roundtrip.json"
        JsonExporter.to_json(sample_smart_chunks, filepath)

        data = json.loads(filepath.read_text(encoding="utf-8"))
        loaded_chunks = [SmartChunk(**item) for item in data]

        assert len(loaded_chunks) == len(sample_smart_chunks)
        for original, loaded in zip(sample_smart_chunks, loaded_chunks):
            assert original.text == loaded.text
            assert original.summary == loaded.summary
            assert original.entities == loaded.entities
            assert original.keywords == loaded.keywords
