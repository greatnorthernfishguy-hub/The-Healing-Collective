"""
Tests for healing_collective_hook.py — Singleton and integration.

Tests the hook WITHOUT requiring the full OpenClawAdapter ecosystem
by mocking ng_ecosystem.init().

# ---- Changelog ----
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


def _hash_embed(text: str, dims: int = 384) -> np.ndarray:
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
    def test_failure_detection_in_message(self, hook_instance):
        result = hook_instance._module_on_message(
            "Error: connection refused on port 5432",
            _hash_embed("Error: connection refused on port 5432"),
        )
        assert result["failure_detected"] is True
        assert "diagnosis" in result

    def test_clean_message_no_detection(self, hook_instance):
        result = hook_instance._module_on_message(
            "The weather is nice today",
            _hash_embed("The weather is nice today"),
        )
        assert result["failure_detected"] is False
