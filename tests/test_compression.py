"""
Tests for core/compression.py — DVS pattern compression.

# ---- Changelog ----
# [2026-02-27] Claude (Opus 4.6) — Initial creation.
# -------------------
"""

import time

import numpy as np
import pytest

from core.compression import CompressionResult, PatternCompressor
from core.config import CompressionConfig
from core.dvs import DiagnosticVectorStore, DVSEntry, DVSEntryType


def _make_embedding(seed: int = 42, dims: int = 384) -> np.ndarray:
    """Create a deterministic test embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dims).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _make_similar_embeddings(base_seed: int, count: int, noise: float = 0.01) -> list:
    """Create a set of similar embeddings (close in cosine space)."""
    rng = np.random.RandomState(base_seed)
    base = rng.randn(384).astype(np.float32)
    base = base / np.linalg.norm(base)

    embeddings = []
    for i in range(count):
        noisy = base + rng.randn(384).astype(np.float32) * noise
        noisy = noisy / np.linalg.norm(noisy)
        embeddings.append(noisy)
    return embeddings


class TestShouldCompress:
    def test_below_threshold(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        # Add a few entries (below 80%)
        for i in range(10):
            dvs.add(DVSEntry.create(
                DVSEntryType.FAILURE_SIGNATURE, "test",
                _make_embedding(seed=i), {"test": True},
            ))

        config = CompressionConfig(trigger_pct=0.80)
        compressor = PatternCompressor(config=config, dvs=dvs)
        assert compressor.should_compress() is False

    def test_above_threshold(self):
        dvs = DiagnosticVectorStore(max_entries=10)
        # Fill to 90%
        for i in range(9):
            dvs.add(DVSEntry.create(
                DVSEntryType.FAILURE_SIGNATURE, "test",
                _make_embedding(seed=i), {"test": True},
            ))

        config = CompressionConfig(trigger_pct=0.80)
        compressor = PatternCompressor(config=config, dvs=dvs)
        assert compressor.should_compress() is True


class TestClustering:
    def test_similar_entries_cluster(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        config = CompressionConfig(similarity_threshold=0.85, min_cluster=3)
        compressor = PatternCompressor(config=config, dvs=dvs)

        # Create 5 similar embeddings
        embeddings = _make_similar_embeddings(42, 5, noise=0.01)
        entries = []
        for emb in embeddings:
            entry = DVSEntry.create(
                DVSEntryType.FAILURE_SIGNATURE, "test", emb, {"test": True},
            )
            entries.append(entry)

        clusters = compressor._find_clusters(entries)
        # With very similar embeddings, should form 1 cluster
        assert len(clusters) == 1
        assert len(clusters[0]) == 5

    def test_dissimilar_entries_separate(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        config = CompressionConfig(similarity_threshold=0.85, min_cluster=3)
        compressor = PatternCompressor(config=config, dvs=dvs)

        # Create 5 very different embeddings
        entries = []
        for i in range(5):
            entry = DVSEntry.create(
                DVSEntryType.FAILURE_SIGNATURE, "test",
                _make_embedding(seed=i * 1000), {"test": True},
            )
            entries.append(entry)

        clusters = compressor._find_clusters(entries)
        # With very different embeddings, should form 5 clusters of size 1
        assert len(clusters) == 5
        assert all(len(c) == 1 for c in clusters)


class TestSynthesizePattern:
    def test_basic_synthesis(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        config = CompressionConfig()
        compressor = PatternCompressor(config=config, dvs=dvs)

        embeddings = _make_similar_embeddings(42, 4, noise=0.01)
        entries = []
        for i, emb in enumerate(embeddings):
            entry = DVSEntry.create(
                DVSEntryType.FAILURE_SIGNATURE, "test", emb,
                {"proposed_primitive": "process_restart"},
                confidence=0.6 + i * 0.05,
                repair_outcome="success" if i % 2 == 0 else "failed",
            )
            entries.append(entry)

        compressed = compressor._synthesize_pattern(entries)
        assert compressed is not None
        assert compressed.entry_type == DVSEntryType.COMPRESSED_PATTERN.value
        assert len(compressed.compression_sources) == 4
        assert compressed.confidence == max(e.confidence for e in entries)
        assert compressed.content["cluster_size"] == 4

    def test_empty_cluster_returns_none(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        config = CompressionConfig()
        compressor = PatternCompressor(config=config, dvs=dvs)
        assert compressor._synthesize_pattern([]) is None


class TestCompress:
    def test_compress_replaces_originals(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        config = CompressionConfig(
            similarity_threshold=0.85, min_cluster=3, trigger_pct=0.80,
        )
        compressor = PatternCompressor(config=config, dvs=dvs)

        # Add 5 similar entries
        embeddings = _make_similar_embeddings(42, 5, noise=0.01)
        original_ids = []
        for emb in embeddings:
            entry = DVSEntry.create(
                DVSEntryType.FAILURE_SIGNATURE, "test", emb,
                {"proposed_primitive": "cache_clear"},
                confidence=0.7,
                repair_outcome="success",
            )
            dvs.add(entry)
            original_ids.append(entry.entry_id)

        size_before = dvs.size
        assert size_before == 5

        result = compressor.compress()

        assert isinstance(result, CompressionResult)
        assert result.entries_compressed == 5
        assert result.entries_created == 1
        assert result.dvs_size_after < result.dvs_size_before

        # Original entries should be gone
        for eid in original_ids:
            assert dvs.get(eid) is None

        # One compressed entry should exist
        compressed = dvs.get_by_type(DVSEntryType.COMPRESSED_PATTERN)
        assert len(compressed) == 1
        assert compressed[0].content["cluster_size"] == 5

    def test_no_compression_below_min_cluster(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        config = CompressionConfig(
            similarity_threshold=0.85, min_cluster=3,
        )
        compressor = PatternCompressor(config=config, dvs=dvs)

        # Add only 2 similar entries (below min_cluster=3)
        embeddings = _make_similar_embeddings(42, 2, noise=0.01)
        for emb in embeddings:
            dvs.add(DVSEntry.create(
                DVSEntryType.FAILURE_SIGNATURE, "test", emb, {},
            ))

        result = compressor.compress()
        assert result.entries_compressed == 0
        assert result.entries_created == 0

    def test_compressed_patterns_not_recompressed(self):
        """COMPRESSED_PATTERN entries should not be compressed again."""
        dvs = DiagnosticVectorStore(max_entries=100)
        config = CompressionConfig(
            similarity_threshold=0.85, min_cluster=3,
        )
        compressor = PatternCompressor(config=config, dvs=dvs)

        # Add one compressed pattern
        emb = _make_embedding(42)
        compressed_entry = DVSEntry.create(
            DVSEntryType.COMPRESSED_PATTERN, "test", emb,
            {"cluster_size": 5},
        )
        dvs.add(compressed_entry)

        compressible = compressor._get_compressible_entries()
        assert all(
            e.entry_type != DVSEntryType.COMPRESSED_PATTERN.value
            for e in compressible
        )


class TestCompressionStats:
    def test_initial_stats(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        config = CompressionConfig()
        compressor = PatternCompressor(config=config, dvs=dvs)
        stats = compressor.stats()
        assert stats["compression_count"] == 0
        assert stats["total_compressed"] == 0
        assert stats["trigger_pct"] == 0.80
