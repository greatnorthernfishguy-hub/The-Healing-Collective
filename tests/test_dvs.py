"""
Tests for core/dvs.py — Diagnostic Vector Store.

# ---- Changelog ----
# [2026-02-26] Claude (Opus 4.6) — Initial creation.
# -------------------
"""

import os
import tempfile

import numpy as np
import pytest

from core.dvs import DiagnosticVectorStore, DVSEntry, DVSEntryType


def _make_embedding(seed: int = 42, dims: int = 384) -> np.ndarray:
    """Create a deterministic test embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dims).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _make_entry(
    entry_type: DVSEntryType = DVSEntryType.FAILURE_SIGNATURE,
    seed: int = 42,
    **kwargs,
) -> DVSEntry:
    """Create a test DVS entry."""
    return DVSEntry.create(
        entry_type=entry_type,
        source_module="test_module",
        embedding=_make_embedding(seed),
        content={"test": True},
        **kwargs,
    )


class TestDVSEntry:
    def test_create(self):
        entry = _make_entry()
        assert entry.entry_id
        assert entry.entry_type == DVSEntryType.FAILURE_SIGNATURE.value
        assert entry.source_module == "test_module"
        assert len(entry.embedding) == 384
        assert entry.ttl_accesses == 0

    def test_roundtrip(self):
        entry = _make_entry(confidence=0.85, repair_outcome="success")
        d = entry.to_dict()
        restored = DVSEntry.from_dict(d)
        assert restored.entry_id == entry.entry_id
        assert restored.confidence == 0.85
        assert restored.repair_outcome == "success"

    def test_embedding_array(self):
        entry = _make_entry()
        arr = entry.embedding_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (384,)


class TestDVSCRUD:
    def test_add_and_get(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        entry = _make_entry()
        eid = dvs.add(entry)
        assert eid == entry.entry_id
        assert dvs.size == 1

        retrieved = dvs.get(eid)
        assert retrieved is not None
        assert retrieved.entry_id == eid
        assert retrieved.ttl_accesses == 1  # get() increments

    def test_update(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        entry = _make_entry()
        dvs.add(entry)

        assert dvs.update(entry.entry_id, confidence=0.99)
        updated = dvs.get(entry.entry_id)
        assert updated.confidence == 0.99

    def test_delete(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        entry = _make_entry()
        dvs.add(entry)
        assert dvs.size == 1
        assert dvs.delete(entry.entry_id)
        assert dvs.size == 0

    def test_get_by_type(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        dvs.add(_make_entry(DVSEntryType.FAILURE_SIGNATURE, seed=1))
        dvs.add(_make_entry(DVSEntryType.FAILURE_SIGNATURE, seed=2))
        dvs.add(_make_entry(DVSEntryType.REPAIR_RECORD, seed=3))

        failures = dvs.get_by_type(DVSEntryType.FAILURE_SIGNATURE)
        assert len(failures) == 2
        repairs = dvs.get_by_type(DVSEntryType.REPAIR_RECORD)
        assert len(repairs) == 1


class TestDVSEviction:
    def test_lru_eviction(self):
        dvs = DiagnosticVectorStore(max_entries=3)
        e1 = _make_entry(seed=1)
        e2 = _make_entry(seed=2)
        e3 = _make_entry(seed=3)
        dvs.add(e1)
        dvs.add(e2)
        dvs.add(e3)
        assert dvs.size == 3

        # Access e1 and e3 to boost their ttl_accesses
        dvs.get(e1.entry_id)
        dvs.get(e3.entry_id)

        # Add a 4th entry — should evict e2 (lowest access count)
        e4 = _make_entry(seed=4)
        dvs.add(e4)
        assert dvs.size == 3
        assert dvs.get(e2.entry_id) is None  # Evicted
        assert dvs.get(e1.entry_id) is not None


class TestDVSSearch:
    def test_cosine_search(self):
        dvs = DiagnosticVectorStore(max_entries=100)

        # Add entries with known embeddings
        similar_emb = _make_embedding(seed=42)
        different_emb = _make_embedding(seed=99)

        e1 = DVSEntry.create(
            entry_type=DVSEntryType.FAILURE_SIGNATURE,
            source_module="test",
            embedding=similar_emb,
            content={"label": "similar"},
        )
        e2 = DVSEntry.create(
            entry_type=DVSEntryType.FAILURE_SIGNATURE,
            source_module="test",
            embedding=different_emb,
            content={"label": "different"},
        )
        dvs.add(e1)
        dvs.add(e2)

        # Search with similar embedding — should rank e1 higher
        results = dvs.search(similar_emb, top_k=2)
        assert len(results) == 2
        assert results[0][0].entry_id == e1.entry_id
        assert results[0][1] > results[1][1]  # Higher similarity

    def test_search_by_type(self):
        dvs = DiagnosticVectorStore(max_entries=100)

        emb = _make_embedding(seed=42)
        dvs.add(DVSEntry.create(DVSEntryType.FAILURE_SIGNATURE, "test", emb, {"a": 1}))
        dvs.add(DVSEntry.create(DVSEntryType.REPAIR_RECORD, "test", emb, {"b": 2}))

        results = dvs.search(emb, entry_type=DVSEntryType.REPAIR_RECORD)
        assert len(results) == 1
        assert results[0][0].content == {"b": 2}

    def test_search_empty_dvs(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        results = dvs.search(_make_embedding(), top_k=5)
        assert results == []


class TestDVSPersistence:
    def test_msgpack_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
            path = f.name

        try:
            # Create and save
            dvs1 = DiagnosticVectorStore(max_entries=100, persistence_path=path)
            e1 = _make_entry(seed=1, confidence=0.5)
            e2 = _make_entry(seed=2, confidence=0.9, repair_outcome="success")
            dvs1.add(e1)
            dvs1.add(e2)
            dvs1.save()

            # Load in fresh instance
            dvs2 = DiagnosticVectorStore(max_entries=100, persistence_path=path)
            assert dvs2.size == 2

            # Verify content
            r1 = dvs2.get(e1.entry_id)
            assert r1 is not None
            assert r1.confidence == 0.5
            r2 = dvs2.get(e2.entry_id)
            assert r2 is not None
            assert r2.repair_outcome == "success"

        finally:
            os.unlink(path)

    def test_json_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dvs = DiagnosticVectorStore(
                max_entries=100,
                persistence_path=os.path.join(tmpdir, "dvs.msgpack"),
            )
            dvs.add(_make_entry(seed=1))
            dvs.export_json(os.path.join(tmpdir, "dvs.json"))
            assert os.path.exists(os.path.join(tmpdir, "dvs.json"))

    def test_corrupt_file_starts_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
            f.write(b"this is not valid msgpack")
            f.flush()

            dvs = DiagnosticVectorStore(max_entries=100, persistence_path=f.name)
            assert dvs.size == 0

        os.unlink(f.name)


class TestDVSStats:
    def test_stats(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        dvs.add(_make_entry(DVSEntryType.FAILURE_SIGNATURE, seed=1))
        dvs.add(_make_entry(DVSEntryType.REPAIR_RECORD, seed=2))

        stats = dvs.stats()
        assert stats["total_entries"] == 2
        assert stats["max_entries"] == 100
        assert stats["type_counts"]["FAILURE_SIGNATURE"] == 1
        assert stats["type_counts"]["REPAIR_RECORD"] == 1
        assert stats["substrate_augmented"] is False
