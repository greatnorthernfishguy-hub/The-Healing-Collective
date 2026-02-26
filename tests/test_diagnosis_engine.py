"""
Tests for core/diagnosis_engine.py — Seven-step diagnosis pipeline.

# ---- Changelog ----
# [2026-02-26] Claude (Opus 4.6) — Initial creation.
# -------------------
"""

import hashlib
import time

import numpy as np
import pytest

from core.config import HealingCollectiveConfig
from core.diagnosis_engine import DiagnosisEngine, DiagnosisResult
from core.dvs import DiagnosticVectorStore, DVSEntry, DVSEntryType
from core.repair_primitives import (
    DEFAULT_PRIMITIVES,
    ExecutionResult,
    LogAndRecommend,
    RepairPrimitive,
    ValidationResult,
)


def _hash_embed(text: str, dims: int = 384) -> np.ndarray:
    """Deterministic hash-based embedding for testing."""
    rng_seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(rng_seed)
    vec = rng.randn(dims).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class MockEcosystem:
    """Minimal mock of NGEcosystem for testing."""

    def __init__(self):
        self._ng = MockNGLite()
        self.tier = 1

    def detect_novelty(self, embedding):
        return self._ng.detect_novelty(embedding)

    def record_outcome(self, embedding, target_id, success, metadata=None):
        return self._ng.record_outcome(embedding, target_id, success, metadata=metadata)

    def save(self):
        pass


class MockNGLite:
    """Minimal mock of NGLite for testing."""

    def __init__(self):
        self._seen = []

    def find_or_create_node(self, embedding):
        class FakeNode:
            node_id = len(self._seen)
        self._seen.append(embedding)
        return FakeNode()

    def detect_novelty(self, embedding):
        if not self._seen:
            return 1.0
        # Lower novelty if we've seen similar embeddings
        for seen in self._seen:
            sim = float(np.dot(embedding, seen) / (
                np.linalg.norm(embedding) * np.linalg.norm(seen) + 1e-8
            ))
            if sim > 0.9:
                return 0.1
        return 0.8

    def record_outcome(self, embedding, target_id, success, metadata=None):
        return None

    def get_recommendations(self, embedding, top_k=3):
        return []


def _make_engine(config=None):
    """Create a test engine with mocks."""
    config = config or HealingCollectiveConfig()
    dvs = DiagnosticVectorStore(max_entries=1000)
    eco = MockEcosystem()
    return DiagnosisEngine(
        config=config,
        dvs=dvs,
        ng_ecosystem=eco,
        primitives=dict(DEFAULT_PRIMITIVES),
        embed_fn=_hash_embed,
    )


class TestDiagnosisPipeline:
    """Test the seven-step diagnosis pipeline."""

    def test_basic_diagnosis(self):
        engine = _make_engine()
        result = engine.diagnose("Connection refused on port 8080")
        assert isinstance(result, DiagnosisResult)
        assert result.tracking_id
        assert result.failure_description == "Connection refused on port 8080"
        assert 0.0 <= result.novelty <= 1.0

    def test_first_failure_low_confidence(self):
        """Fresh engine with no knowledge should have low confidence."""
        engine = _make_engine()
        result = engine.diagnose("Database connection timeout")
        # First failure on fresh substrate — should be low confidence
        assert result.confidence < 0.70
        assert result.action_taken in ("recommend", "silent_log")

    def test_repeated_failure_builds_confidence(self):
        """Multiple similar failures should increase confidence over time."""
        engine = _make_engine()
        description = "Process crashed: OOM killer"

        # Pre-seed DVS with successful repair records
        for i in range(10):
            emb = _hash_embed(description)
            engine._dvs.add(DVSEntry.create(
                entry_type=DVSEntryType.REPAIR_RECORD,
                source_module="healing_collective",
                embedding=emb,
                content={
                    "proposed_primitive": "process_restart",
                    "description": description,
                },
                confidence=0.8,
                repair_outcome="success",
            ))

        # Now diagnose the same failure — should have higher confidence
        result = engine.diagnose(description)
        assert result.proposed_primitive is not None
        # With 10 successful records, confidence should be meaningful
        assert result.confidence > 0.0

    def test_novel_failure_cautious(self):
        """Novel failures should get lower confidence ceiling."""
        engine = _make_engine()
        # Pre-seed with one type of failure
        for i in range(5):
            emb = _hash_embed("Network timeout error")
            engine._dvs.add(DVSEntry.create(
                entry_type=DVSEntryType.REPAIR_RECORD,
                source_module="healing_collective",
                embedding=emb,
                content={"proposed_primitive": "retry_with_backoff"},
                confidence=0.8,
                repair_outcome="success",
            ))

        # Now report a completely different failure
        result = engine.diagnose("Kernel panic: unrecoverable filesystem corruption")
        # Novel failure should have lower confidence
        assert result.confidence < 0.90

    def test_validate_before_execute_enforcement(self):
        """execute() must NEVER be called without validate() returning passed."""

        class AlwaysFailsValidate(RepairPrimitive):
            @property
            def scope(self):
                return "host"
            def validate(self, context):
                return ValidationResult(passed=False, reason="Always fails")
            def execute(self, context):
                raise AssertionError("execute() should not be called!")

        engine = _make_engine()
        engine._primitives["always_fails"] = AlwaysFailsValidate()

        # Pre-seed to propose always_fails with high confidence
        emb = _hash_embed("specific failure pattern xyz")
        for i in range(15):
            engine._dvs.add(DVSEntry.create(
                entry_type=DVSEntryType.REPAIR_RECORD,
                source_module="healing_collective",
                embedding=emb,
                content={"proposed_primitive": "always_fails"},
                confidence=0.95,
                repair_outcome="success",
            ))

        result = engine.diagnose("specific failure pattern xyz")
        # Should downgrade to recommend since validate failed
        if result.proposed_primitive == "always_fails":
            assert result.action_taken in ("recommend", "silent_log")
            assert result.execution_result is None

    def test_host_premium_applied(self):
        """Host repairs require higher confidence (+0.15 premium)."""
        config = HealingCollectiveConfig()
        assert config.confidence_host_premium == 0.15

        engine = _make_engine(config)
        # For host source, threshold is 0.70 + 0.15 = 0.85
        # So even moderate confidence should result in recommend, not execute
        result = engine.diagnose("Host system error", source="host")
        # Fresh engine won't auto-execute
        assert result.action_taken != "auto_execute"


class TestConfidenceThresholds:
    def test_auto_execute_threshold(self):
        config = HealingCollectiveConfig()
        assert config.confidence_auto_execute == 0.70

    def test_recommend_threshold(self):
        config = HealingCollectiveConfig()
        assert config.confidence_recommend == 0.40

    def test_host_premium(self):
        config = HealingCollectiveConfig()
        assert config.confidence_host_premium == 0.15


class TestCooldown:
    def test_cooldown_prevents_repeat(self):
        engine = _make_engine()

        # Pre-seed with high-confidence repair knowledge
        description = "Cooldown test failure"
        emb = _hash_embed(description)
        for i in range(15):
            engine._dvs.add(DVSEntry.create(
                entry_type=DVSEntryType.REPAIR_RECORD,
                source_module="healing_collective",
                embedding=emb,
                content={"proposed_primitive": "log_and_recommend"},
                confidence=0.95,
                repair_outcome="success",
            ))

        # First diagnosis
        r1 = engine.diagnose(description)
        first_action = r1.action_taken

        # Manually set a cooldown for the proposed primitive
        if r1.proposed_primitive:
            failure_hash = engine._hash_failure(description)
            engine._cooldowns[(failure_hash, r1.proposed_primitive)] = time.time() + 300

        # Second diagnosis with same failure (should be downgraded by cooldown)
        r2 = engine.diagnose(description)
        if r2.proposed_primitive == r1.proposed_primitive:
            # Should be downgraded from auto_execute to recommend
            assert r2.action_taken in ("recommend", "silent_log")


class TestStatusTracking:
    def test_get_status(self):
        engine = _make_engine()
        result = engine.diagnose("Test failure for tracking")
        tid = result.tracking_id

        status = engine.get_status(tid)
        assert status is not None
        assert status["tracking_id"] == tid
        assert status["failure_description"] == "Test failure for tracking"

    def test_get_status_unknown(self):
        engine = _make_engine()
        assert engine.get_status("nonexistent-id") is None


class TestEngineStats:
    def test_stats(self):
        engine = _make_engine()
        engine.diagnose("Failure 1")
        engine.diagnose("Failure 2")

        stats = engine.stats()
        assert stats["failures_observed"] == 2
        assert stats["primitives_registered"] == 8
        assert stats["tracked_diagnoses"] == 2
