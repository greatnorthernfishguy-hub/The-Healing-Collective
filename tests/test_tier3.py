"""
Tests for core/tier3_upgrade.py — Cluster-wide repair knowledge sharing.

# ---- Changelog ----
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
    def test_broadcast_creates_file(self):
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
            assert os.path.exists(broadcast_file)

            with open(broadcast_file, "r") as f:
                lines = f.readlines()
            assert len(lines) == 1

            record = json.loads(lines[0])
            assert record["module_id"] == "healing_collective"
            assert record["proposed_primitive"] == "process_restart"
            assert record["outcome"] == "success"

    def test_multiple_broadcasts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dvs = DiagnosticVectorStore(max_entries=100)
            coord = Tier3Coordinator(
                module_id="healing_collective",
                ng_ecosystem=None,
                dvs=dvs,
                embed_fn=None,
                shared_dir=tmpdir,
            )

            for i in range(5):
                coord.broadcast_repair(
                    failure_description=f"Failure {i}",
                    embedding=_make_embedding(i),
                    proposed_primitive="cache_clear",
                    confidence=0.8,
                    outcome="success",
                )

            broadcast_file = os.path.join(tmpdir, "healing_collective_repairs.jsonl")
            with open(broadcast_file, "r") as f:
                lines = f.readlines()
            assert len(lines) == 5


class TestSync:
    def test_sync_reads_peer_repairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dvs = DiagnosticVectorStore(max_entries=100)
            coord = Tier3Coordinator(
                module_id="healing_collective",
                ng_ecosystem=None,
                dvs=dvs,
                embed_fn=None,
                shared_dir=tmpdir,
            )

            # Write peer repair broadcasts
            emb = _make_embedding(42)
            peer_file = os.path.join(tmpdir, "trollguard_repairs.jsonl")
            with open(peer_file, "w") as f:
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
            assert result.peers_read == 1
            assert result.records_imported == 3

            # Verify records are in DVS
            assert dvs.size == 3
            repairs = dvs.get_by_type(DVSEntryType.REPAIR_RECORD)
            assert len(repairs) == 3

    def test_sync_skips_own_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dvs = DiagnosticVectorStore(max_entries=100)
            coord = Tier3Coordinator(
                module_id="healing_collective",
                ng_ecosystem=None,
                dvs=dvs,
                embed_fn=None,
                shared_dir=tmpdir,
            )

            # Write own repair broadcasts
            own_file = os.path.join(tmpdir, "healing_collective_repairs.jsonl")
            with open(own_file, "w") as f:
                record = {
                    "module_id": "healing_collective",
                    "embedding": _make_embedding(42).tolist(),
                    "proposed_primitive": "test",
                    "confidence": 0.8,
                    "outcome": "success",
                }
                f.write(json.dumps(record) + "\n")

            result = coord.sync_cluster_knowledge()
            assert result.records_imported == 0

    def test_incremental_sync(self):
        """Second sync should only read new records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dvs = DiagnosticVectorStore(max_entries=100)
            coord = Tier3Coordinator(
                module_id="healing_collective",
                ng_ecosystem=None,
                dvs=dvs,
                embed_fn=None,
                shared_dir=tmpdir,
            )

            peer_file = os.path.join(tmpdir, "trollguard_repairs.jsonl")

            # Write 2 records
            emb = _make_embedding(42)
            with open(peer_file, "w") as f:
                for i in range(2):
                    record = {
                        "module_id": "trollguard",
                        "embedding": emb.tolist(),
                        "proposed_primitive": "test",
                        "confidence": 0.8,
                        "outcome": "success",
                    }
                    f.write(json.dumps(record) + "\n")

            r1 = coord.sync_cluster_knowledge()
            assert r1.records_imported == 2

            # Write 1 more record
            with open(peer_file, "a") as f:
                record = {
                    "module_id": "trollguard",
                    "embedding": emb.tolist(),
                    "proposed_primitive": "test",
                    "confidence": 0.8,
                    "outcome": "success",
                }
                f.write(json.dumps(record) + "\n")

            r2 = coord.sync_cluster_knowledge()
            assert r2.records_imported == 1  # Only the new record


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

    def test_with_peer_data(self):
        """Peer data should influence cluster confidence."""
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

            # Write peer repairs
            peer_file = os.path.join(tmpdir, "trollguard_repairs.jsonl")
            with open(peer_file, "w") as f:
                for i in range(5):
                    record = {
                        "module_id": "trollguard",
                        "embedding": emb.tolist(),
                        "proposed_primitive": "process_restart",
                        "confidence": 0.9,
                        "outcome": "success",
                    }
                    f.write(json.dumps(record) + "\n")

            # Sync first
            coord.sync_cluster_knowledge()

            # Get cluster confidence
            result = coord.get_cluster_confidence(
                embedding=emb,
                primitive="process_restart",
                local_confidence=0.60,
            )
            # Should be higher than 0.60 due to peer successes
            assert result > 0.60


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
        with tempfile.TemporaryDirectory() as tmpdir:
            dvs = DiagnosticVectorStore(max_entries=100)
            coord = Tier3Coordinator(
                module_id="healing_collective",
                ng_ecosystem=None,
                dvs=dvs,
                embed_fn=None,
                shared_dir=tmpdir,
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
