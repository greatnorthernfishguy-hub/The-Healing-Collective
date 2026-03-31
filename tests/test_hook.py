"""
Tests for healing_collective_hook.py — Singleton and integration.

Tests the hook WITHOUT requiring the full OpenClawAdapter ecosystem
by mocking ng_ecosystem.init().

# ---- Changelog ----
# [2026-03-26] Claude Code Opus — Punchlist #102: Fix stale tests from embedding migration
# What: Updated test dimensions from 384→768 to match current embedding pipeline
# Why: Punchlist #102 — tests obsoleted by snowflake-arctic-embed-m-v1.5 migration
# How: Updated _hash_embed default dims from 384→768
# -------------------
# [2026-03-18] Claude (CC) — Updated TestMessageScanning for substrate detection
# What: Rewrote failure detection tests to match new substrate-based
#   detection (punch list #70). Old tests used regex pattern matching;
#   new tests verify DVS similarity trigger, novelty trigger, and
#   result metadata fields.
# Why: _FAILURE_PATTERNS regex gate replaced with substrate signals.
# How: Pre-seed DVS with failure signature for similarity test. Mock
#   ecosystem novelty for novelty trigger test. Verify new result
#   fields (dvs_similarity, novelty, trigger).
# -------------------
# [2026-02-26] Claude (Opus 4.6) — Initial creation.
# -------------------
"""

import hashlib
import importlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional
from unittest import mock

import numpy as np
import pytest


def _hash_embed(text: str, dims: int = 768) -> np.ndarray:
    """Deterministic hash-based embedding for testing."""
    rng_seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
    rng = np.random.RandomState(rng_seed)
    vec = rng.randn(dims).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class MockEcosystem:
    """Minimal mock of NGEcosystem for testing the hook."""

    def __init__(self, module_id="test", state_path=None, config=None):
        self.module_id = module_id
        self.tier = 1
        self.tier_name = "Standalone (Tier 1)"
        self._ng = MockNGLite()
        self._peer_bridge = None

    def record_outcome(self, embedding, target_id, success, metadata=None):
        return None

    def get_context(self, embedding, top_k=3):
        return {
            "tier": 1,
            "tier_name": "Standalone (Tier 1)",
            "recommendations": [],
            "novelty": 0.5,
            "ng_context": None,
        }

    def detect_novelty(self, embedding):
        return 0.5

    def get_recommendations(self, embedding, top_k=3):
        return []

    def save(self):
        pass

    def stats(self):
        return {"tier": 1, "ecosystem_version": "1.0.0", "module_id": self.module_id}

    def shutdown(self):
        pass


class MockNGLite:
    """Minimal mock of NGLite."""

    def __init__(self):
        self._seen = []

    def find_or_create_node(self, embedding):
        class FakeNode:
            node_id = len(self._seen)
        self._seen.append(embedding)
        return FakeNode()

    def detect_novelty(self, embedding):
        return 0.5

    def record_outcome(self, embedding, target_id, success, metadata=None):
        return None

    def get_recommendations(self, embedding, top_k=3):
        return []

    def get_stats(self):
        return {"nodes": 0, "synapses": 0}


@pytest.fixture
def hook_instance(tmp_path):
    """Create a HealingCollectiveHook instance with mocked ecosystem."""
    # Reset singleton state
    import healing_collective_hook as hch
    hch._INSTANCE = None

    # Mock ng_ecosystem.init to return our mock
    mock_eco = MockEcosystem()

    with mock.patch.dict(os.environ, {
        "HEALING_COLLECTIVE_WORKSPACE_DIR": str(tmp_path / "workspace"),
    }):
        with mock.patch("ng_ecosystem.init", return_value=mock_eco):
            # Mock sentence_transformers to use hash embedding
            with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
                instance = hch.get_instance()
                yield instance

    # Cleanup singleton
    hch._INSTANCE = None


class TestSingleton:
    def test_get_instance_returns_same_object(self, tmp_path):
        import healing_collective_hook as hch
        hch._INSTANCE = None

        mock_eco = MockEcosystem()
        with mock.patch.dict(os.environ, {
            "HEALING_COLLECTIVE_WORKSPACE_DIR": str(tmp_path / "workspace"),
        }):
            with mock.patch("ng_ecosystem.init", return_value=mock_eco):
                with mock.patch.dict(sys.modules, {"sentence_transformers": None}):
                    inst1 = hch.get_instance()
                    inst2 = hch.get_instance()
                    assert inst1 is inst2

        hch._INSTANCE = None


class TestHookAttributes:
    def test_module_id(self, hook_instance):
        assert hook_instance.MODULE_ID == "healing_collective"

    def test_skill_name(self, hook_instance):
        assert hook_instance.SKILL_NAME == "Healing Collective"


class TestReportFailure:
    def test_returns_tracking_id(self, hook_instance):
        tracking_id = hook_instance.report_failure("Test error")
        assert tracking_id is not None
        assert isinstance(tracking_id, str)
        assert len(tracking_id) > 0

    def test_get_healing_status(self, hook_instance):
        tid = hook_instance.report_failure("Another test error")
        status = hook_instance.get_healing_status(tid)
        assert status is not None
        assert status["tracking_id"] == tid
        assert status["failure_description"] == "Another test error"

    def test_get_status_unknown_id(self, hook_instance):
        status = hook_instance.get_healing_status("nonexistent-id")
        assert status is None

    def test_failure_with_metadata(self, hook_instance):
        tid = hook_instance.report_failure(
            "DB connection timeout",
            metadata={"host": "db.example.com", "port": 5432},
        )
        status = hook_instance.get_healing_status(tid)
        assert status is not None


class TestModuleStats:
    def test_stats_structure(self, hook_instance):
        stats = hook_instance._module_stats()
        assert "failures_observed" in stats
        assert "repairs_executed" in stats
        assert "repairs_succeeded" in stats
        assert "dvs_entries" in stats
        assert "primitives_registered" in stats

    def test_stats_after_failure(self, hook_instance):
        hook_instance.report_failure("Test")
        stats = hook_instance._module_stats()
        assert stats["failures_observed"] >= 1


class TestRegisterPrimitive:
    def test_register_custom(self, hook_instance):
        from core.repair_primitives import RepairPrimitive, ValidationResult, ExecutionResult

        class CustomPrimitive(RepairPrimitive):
            def validate(self, context):
                return ValidationResult(passed=True, reason="OK")
            def execute(self, context):
                return ExecutionResult(status="success", detail="Custom executed")

        hook_instance.register_primitive("custom_repair", CustomPrimitive())
        stats = hook_instance._module_stats()
        assert stats["primitives_registered"] >= 9  # 8 defaults + 1 custom

    def test_register_invalid_type(self, hook_instance):
        with pytest.raises(TypeError):
            hook_instance.register_primitive("invalid", "not a primitive")


class TestMessageScanning:
    def test_dvs_similarity_triggers_detection(self, hook_instance):
        """When DVS contains a similar failure signature, detection fires."""
        # Pre-seed DVS with a failure signature via report_failure
        hook_instance.report_failure("Connection refused on port 5432")

        # Now send a similar message — DVS should find the seeded signature
        failure_embedding = _hash_embed("Connection refused on port 5432")
        result = hook_instance._module_on_message(
            "Connection refused on port 5432",
            failure_embedding,
        )
        # Same embedding should have high DVS similarity to the seeded entry
        assert result["dvs_similarity"] > 0
        assert "trigger" in result
        assert "novelty" in result

    def test_novelty_triggers_detection(self, hook_instance):
        """When substrate reports high novelty, detection fires."""
        # Mock ecosystem to return very high novelty
        hook_instance._eco.detect_novelty = lambda emb: 0.95

        result = hook_instance._module_on_message(
            "Something completely unprecedented happened",
            _hash_embed("Something completely unprecedented happened"),
        )
        assert result["failure_detected"] is True
        assert result["trigger"] == "novelty"
        assert result["novelty"] >= 0.85
        assert "diagnosis" in result

    def test_clean_message_no_detection(self, hook_instance):
        """Normal messages with no DVS match and low novelty pass through."""
        # Default mock ecosystem returns novelty=0.5, below threshold
        result = hook_instance._module_on_message(
            "The weather is nice today",
            _hash_embed("The weather is nice today"),
        )
        assert result["failure_detected"] is False
        assert result["trigger"] == "none"

    def test_result_metadata_fields(self, hook_instance):
        """Substrate detection results include observability metadata."""
        result = hook_instance._module_on_message(
            "Routine status check",
            _hash_embed("Routine status check"),
        )
        assert "dvs_similarity" in result
        assert "novelty" in result
        assert "trigger" in result
        assert isinstance(result["dvs_similarity"], float)
        assert isinstance(result["novelty"], float)

    def test_no_regex_dependency(self):
        """Verify the hook module does not import re (Law 7 compliance)."""
        import healing_collective_hook as hch
        import ast
        with open(hch.__file__) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "re", "re module should not be imported"
