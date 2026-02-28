"""
Tests for core/congregation.py — Peer deliberation.

# ---- Changelog ----
# [2026-02-27] Claude (Opus 4.6) — Initial creation.
# -------------------
"""

import json
import os
import tempfile

import numpy as np
import pytest

from core.config import CongregationConfig
from core.congregation import Congregation, CongregationResult, PeerVote


def _make_embedding(seed: int = 42, dims: int = 384) -> np.ndarray:
    """Create a deterministic test embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dims).astype(np.float32)
    return vec / np.linalg.norm(vec)


class MockPeerBridge:
    def __init__(self, shared_dir):
        self._shared_dir = shared_dir


class MockEcosystem:
    def __init__(self, shared_dir=None):
        self.module_id = "healing_collective"
        self._peer_bridge = MockPeerBridge(shared_dir) if shared_dir else None


class TestShouldDeliberate:
    def test_too_low_confidence(self):
        config = CongregationConfig()
        cong = Congregation(config=config, ng_ecosystem=None, dvs=None, embed_fn=None)
        assert cong.should_deliberate(
            confidence=0.3, source="host",
            auto_execute_threshold=0.85, recommend_threshold=0.40,
        ) is False

    def test_already_confident(self):
        config = CongregationConfig()
        cong = Congregation(config=config, ng_ecosystem=None, dvs=None, embed_fn=None)
        assert cong.should_deliberate(
            confidence=0.90, source="host",
            auto_execute_threshold=0.85, recommend_threshold=0.40,
        ) is False

    def test_in_recommend_zone(self):
        config = CongregationConfig()
        cong = Congregation(config=config, ng_ecosystem=None, dvs=None, embed_fn=None)
        assert cong.should_deliberate(
            confidence=0.60, source="host",
            auto_execute_threshold=0.85, recommend_threshold=0.40,
        ) is True

    def test_host_require_for_host_repairs(self):
        config = CongregationConfig(require_for_host_repairs=True)
        cong = Congregation(config=config, ng_ecosystem=None, dvs=None, embed_fn=None)
        assert cong.should_deliberate(
            confidence=0.60, source="host",
            auto_execute_threshold=0.85, recommend_threshold=0.40,
        ) is True


class TestVoteAggregation:
    def test_no_votes_returns_original(self):
        config = CongregationConfig()
        cong = Congregation(config=config, ng_ecosystem=None, dvs=None, embed_fn=None)
        adjusted = cong._aggregate_votes([], 0.50)
        assert adjusted == 0.50

    def test_positive_votes_increase_confidence(self):
        config = CongregationConfig(confidence_spread=0.15, max_candidates=4)
        cong = Congregation(config=config, ng_ecosystem=None, dvs=None, embed_fn=None)
        votes = [
            PeerVote(peer_module="trollguard", confidence=0.9, supporting_repairs=5, similarity=0.8),
            PeerVote(peer_module="tid", confidence=0.85, supporting_repairs=3, similarity=0.7),
        ]
        adjusted = cong._aggregate_votes(votes, 0.50)
        assert adjusted > 0.50

    def test_negative_votes_decrease_confidence(self):
        config = CongregationConfig(confidence_spread=0.15, max_candidates=4)
        cong = Congregation(config=config, ng_ecosystem=None, dvs=None, embed_fn=None)
        votes = [
            PeerVote(peer_module="trollguard", confidence=0.1, supporting_repairs=0, similarity=0.8),
            PeerVote(peer_module="tid", confidence=0.0, supporting_repairs=0, similarity=0.7),
        ]
        adjusted = cong._aggregate_votes(votes, 0.50)
        assert adjusted < 0.50

    def test_confidence_bounded(self):
        config = CongregationConfig(confidence_spread=0.15, max_candidates=4)
        cong = Congregation(config=config, ng_ecosystem=None, dvs=None, embed_fn=None)
        votes = [
            PeerVote(peer_module="p1", confidence=1.0, supporting_repairs=100, similarity=1.0),
        ]
        adjusted = cong._aggregate_votes(votes, 0.99)
        assert 0.0 <= adjusted <= 1.0


class TestDeliberate:
    def test_deliberate_no_peers(self):
        """Deliberation with no peer bridge returns original confidence."""
        config = CongregationConfig()
        eco = MockEcosystem(shared_dir=None)
        cong = Congregation(config=config, ng_ecosystem=eco, dvs=None, embed_fn=None)
        result = cong.deliberate(
            failure_embedding=_make_embedding(),
            proposed_primitive="process_restart",
            local_confidence=0.55,
        )
        assert isinstance(result, CongregationResult)
        assert result.peers_consulted == 0
        assert result.adjusted_confidence == 0.55

    def test_deliberate_with_peer_events(self):
        """Deliberation reads peer event files and adjusts confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            shared_dir = Path(tmpdir)

            # Write peer events for trollguard
            emb = _make_embedding(seed=42)
            events = []
            for i in range(5):
                events.append({
                    "module_id": "trollguard",
                    "target_id": "repair:process_restart",
                    "success": True,
                    "embedding": emb.tolist(),
                    "timestamp": 1000 + i,
                })
            with open(shared_dir / "trollguard.jsonl", "w") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")

            eco = MockEcosystem(shared_dir=shared_dir)
            config = CongregationConfig()
            cong = Congregation(config=config, ng_ecosystem=eco, dvs=None, embed_fn=None)

            result = cong.deliberate(
                failure_embedding=emb,
                proposed_primitive="process_restart",
                local_confidence=0.55,
            )
            assert result.peers_consulted >= 1


class TestCongregationStats:
    def test_initial_stats(self):
        config = CongregationConfig()
        cong = Congregation(config=config, ng_ecosystem=None, dvs=None, embed_fn=None)
        stats = cong.stats()
        assert stats["deliberations"] == 0
        assert stats["consensus_reached"] == 0
        assert stats["max_candidates"] == 4

    def test_stats_after_deliberation(self):
        config = CongregationConfig()
        eco = MockEcosystem(shared_dir=None)
        cong = Congregation(config=config, ng_ecosystem=eco, dvs=None, embed_fn=None)
        cong.deliberate(_make_embedding(), "test", 0.5)
        stats = cong.stats()
        assert stats["deliberations"] == 1
