"""
Tests for core/tier3_upgrade.py — Cluster-wide repair knowledge sharing.

# ---- Changelog ----
# [2026-06-29] Claude Code (Sonnet 4.6) — Update for #335 JSONL retirement
#   What: broadcast_repair() and sync_cluster_knowledge() are now no-ops (JSONL retired).
#         Tests updated: file-I/O assertions removed, counter-tracking + return-type tests kept,
#         cluster-confidence with-peer-data test uses manual DVS injection instead of file sync.
# [2026-02-27] Claude (Opus 4.6) — Initial creation.
# -------------------
"""

import json
import os
import tempfile

import numpy as np
import pytest

from core.dvs import DiagnosticVectorStore, DVSEntry, DVSEntryType
from core.tier3_upgrade import SyncResult, Tier3Coordinator


def _make_embedding(seed: int = 42, dims: int = 384) -> np.ndarray:
    """Create a deterministic test embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dims).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestBroadcast:
    def test_broadcast_is_noop_no_file_written(self):
        """broadcast_repair() no longer writes JSONL — Commons deposit handles it (#335)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dvs = DiagnosticVectorStore(max_entries=100)
            coord = Tier3Coordinator(
                module_id="healing_collective",
                ng_ecosystem=None,
                dvs=dvs,
                embed_fn=None,
                shared_dir=tmpdir,
            )

            emb = _make_embedding(42)
            coord.broadcast_repair(
                failure_description="Test failure",
                embedding=emb,
                proposed_primitive="process_restart",
                confidence=0.75,
                outcome="success",
                tracking_id="test-123",
            )

            broadcast_file = os.path.join(tmpdir, "healing_collective_repairs.jsonl")
            assert not os.path.exists(broadcast_file), (
                "broadcast_repair() must not write JSONL — Commons deposit handles broadcast (#335)"
            )

    def test_broadcast_increments_counter(self):
        """Counter still increments for stats() even though method is a no-op."""
        dvs = DiagnosticVectorStore(max_entries=100)
        coord = Tier3Coordinator(
            module_id="healing_collective",
            ng_ecosystem=None,
            dvs=dvs,
            embed_fn=None,
        )

        for i in range(3):
            coord.broadcast_repair(
                failure_description=f"Failure {i}",
                embedding=_make_embedding(i),
                proposed_primitive="cache_clear",
                confidence=0.8,
                outcome="success",
            )

        assert coord.stats()["broadcasts"] == 3

    def test_broadcast_does_not_raise(self):
        """broadcast_repair() must not raise regardless of arguments."""
        dvs = DiagnosticVectorStore(max_entries=100)
        coord = Tier3Coordinator(
            module_id="healing_collective",
            ng_ecosystem=None,
            dvs=dvs,
            embed_fn=None,
        )
        coord.broadcast_repair(
            failure_description="",
            embedding=_make_embedding(99),
            proposed_primitive="unknown",
            confidence=0.0,
            outcome="failed",
        )


class TestSync:
    def test_sync_returns_empty_result(self):
        """sync_cluster_knowledge() returns empty SyncResult — JSONL reads retired (#335)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dvs = DiagnosticVectorStore(max_entries=100)
            coord = Tier3Coordinator(
                module_id="healing_collective",
                ng_ecosystem=None,
                dvs=dvs,
                embed_fn=None,
                shared_dir=tmpdir,
            )

            # Even if peer JSONL files exist, sync ignores them (retired)
            peer_file = os.path.join(tmpdir, "trollguard_repairs.jsonl")
            with open(peer_file, "w") as f:
                emb = _make_embedding(42)
                for i in range(3):
                    record = {
                        "module_id": "trollguard",
                        "failure_description": f"TG failure {i}",
                        "embedding": emb.tolist(),
                        "proposed_primitive": "process_restart",
                        "confidence": 0.8,
                        "outcome": "success",
                    }
                    f.write(json.dumps(record) + "\n")

            result = coord.sync_cluster_knowledge()
            assert isinstance(result, SyncResult)
            assert result.peers_read == 0
            assert result.records_imported == 0
            assert dvs.size == 0, "sync_cluster_knowledge() must not import JSONL records (#335)"

    def test_sync_increments_counter(self):
        """Counter still increments for stats() even though method is a no-op."""
        dvs = DiagnosticVectorStore(max_entries=100)
        coord = Tier3Coordinator(
            module_id="healing_collective",
            ng_ecosystem=None,
            dvs=dvs,
            embed_fn=None,
        )
        coord.sync_cluster_knowledge()
        coord.sync_cluster_knowledge()
        assert coord.stats()["syncs"] == 2

    def test_sync_does_not_raise(self):
        """sync_cluster_knowledge() must not raise even with no shared_dir."""
        dvs = DiagnosticVectorStore(max_entries=100)
        coord = Tier3Coordinator(
            module_id="healing_collective",
            ng_ecosystem=None,
            dvs=dvs,
            embed_fn=None,
        )
        result = coord.sync_cluster_knowledge()
        assert isinstance(result, SyncResult)


class TestClusterConfidence:
    def test_no_peer_data(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        coord = Tier3Coordinator(
            module_id="healing_collective",
            ng_ecosystem=None,
            dvs=dvs,
            embed_fn=None,
        )
        result = coord.get_cluster_confidence(
            embedding=_make_embedding(42),
            primitive="process_restart",
            local_confidence=0.60,
        )
        assert result == 0.60

    def test_with_peer_dvs_entries(self):
        """Peer data injected directly into DVS (simulating _bucket_commons_repair()) influences
        cluster confidence — no JSONL sync needed (#335)."""
        dvs = DiagnosticVectorStore(max_entries=100)
        coord = Tier3Coordinator(
            module_id="healing_collective",
            ng_ecosystem=None,
            dvs=dvs,
            embed_fn=None,
        )

        emb = _make_embedding(42)

        # Manually inject peer repair records as _bucket_commons_repair() would
        for i in range(5):
            entry = DVSEntry.create(
                entry_type=DVSEntryType.REPAIR_RECORD,
                source_module="trollguard",
                embedding=emb,
                content={
                    "proposed_primitive": "process_restart",
                    "failure_description": f"peer failure {i}",
                    "tracking_id": f"peer-{i}",
                    "source": "tier3_sync",
                    "original_module": "trollguard",
                },
                confidence=0.9,
                repair_outcome="success",
            )
            dvs.add(entry)

        result = coord.get_cluster_confidence(
            embedding=emb,
            primitive="process_restart",
            local_confidence=0.60,
        )
        assert result > 0.60, "Peer DVS entries should raise cluster confidence above local"


class TestTier3Stats:
    def test_initial_stats(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        coord = Tier3Coordinator(
            module_id="healing_collective",
            ng_ecosystem=None,
            dvs=dvs,
            embed_fn=None,
        )
        stats = coord.stats()
        assert stats["broadcasts"] == 0
        assert stats["syncs"] == 0
        assert stats["total_imported"] == 0

    def test_stats_after_operations(self):
        dvs = DiagnosticVectorStore(max_entries=100)
        coord = Tier3Coordinator(
            module_id="healing_collective",
            ng_ecosystem=None,
            dvs=dvs,
            embed_fn=None,
        )

        coord.broadcast_repair(
            failure_description="Test",
            embedding=_make_embedding(42),
            proposed_primitive="test",
            confidence=0.8,
            outcome="success",
        )
        coord.sync_cluster_knowledge()

        stats = coord.stats()
        assert stats["broadcasts"] == 1
        assert stats["syncs"] == 1
