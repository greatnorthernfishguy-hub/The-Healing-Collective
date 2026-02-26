"""
Tests for core/repair_primitives.py — Validate/execute contract.

# ---- Changelog ----
# [2026-02-26] Claude (Opus 4.6) — Initial creation.
# -------------------
"""

import os
import tempfile

import pytest

from core.repair_primitives import (
    CacheClear,
    CheckpointRestore,
    ConfigAdjust,
    ConnectionPoolReset,
    ExecutionResult,
    LogAndRecommend,
    NGLiteRebalance,
    ProcessRestart,
    RepairPrimitive,
    RetryWithBackoff,
    ValidationResult,
    DEFAULT_PRIMITIVES,
)


class TestValidationResult:
    def test_passed(self):
        r = ValidationResult(passed=True, reason="OK")
        assert r.passed is True
        assert r.reason == "OK"
        assert r.preconditions == {}

    def test_failed(self):
        r = ValidationResult(passed=False, reason="Missing pid")
        assert r.passed is False


class TestExecutionResult:
    def test_success(self):
        r = ExecutionResult(status="success", detail="Done")
        assert r.status == "success"
        assert r.rollback_info is None

    def test_with_rollback(self):
        r = ExecutionResult(status="success", detail="Done", rollback_info={"key": "val"})
        assert r.rollback_info == {"key": "val"}


class TestLogAndRecommend:
    """LogAndRecommend always passes validate() — it's the safe fallback."""

    def test_always_validates(self):
        p = LogAndRecommend()
        result = p.validate({})
        assert result.passed is True

    def test_executes_successfully(self):
        p = LogAndRecommend()
        result = p.execute({"description": "Test failure", "recommendation": "Restart"})
        assert result.status == "success"

    def test_scope(self):
        p = LogAndRecommend()
        assert p.scope == "any"


class TestCacheClear:
    def test_validate_missing_path(self):
        p = CacheClear()
        result = p.validate({})
        assert result.passed is False

    def test_validate_nonexistent_path(self):
        p = CacheClear()
        result = p.validate({"cache_path": "/nonexistent/path/xyz"})
        assert result.passed is False

    def test_validate_and_execute_dir(self):
        p = CacheClear()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some cache files
            for i in range(3):
                with open(os.path.join(tmpdir, f"cache_{i}.tmp"), "w") as f:
                    f.write("cached data")

            result = p.validate({"cache_path": tmpdir})
            assert result.passed is True

            exec_result = p.execute({"cache_path": tmpdir})
            assert exec_result.status == "success"
            # Directory should still exist but be empty
            assert os.path.isdir(tmpdir)
            assert len(os.listdir(tmpdir)) == 0

    def test_validate_and_execute_file(self):
        p = CacheClear()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"cached")
            f.flush()

            result = p.validate({"cache_path": f.name})
            assert result.passed is True

            exec_result = p.execute({"cache_path": f.name})
            assert exec_result.status == "success"
            assert not os.path.exists(f.name)


class TestProcessRestart:
    def test_validate_missing_pid(self):
        p = ProcessRestart()
        result = p.validate({})
        assert result.passed is False

    def test_validate_nonexistent_pid(self):
        p = ProcessRestart()
        result = p.validate({"pid": 999999999})
        assert result.passed is False

    def test_validate_own_pid(self):
        p = ProcessRestart()
        result = p.validate({"pid": os.getpid()})
        assert result.passed is True

    def test_scope(self):
        p = ProcessRestart()
        assert p.scope == "host"


class TestConfigAdjust:
    def test_validate_missing_key(self):
        p = ConfigAdjust()
        result = p.validate({})
        assert result.passed is False

    def test_validate_within_bounds(self):
        p = ConfigAdjust()
        result = p.validate({
            "config_key": "timeout",
            "config_value": 30,
            "safe_bounds": {"min": 1, "max": 60},
        })
        assert result.passed is True

    def test_validate_out_of_bounds(self):
        p = ConfigAdjust()
        result = p.validate({
            "config_key": "timeout",
            "config_value": 100,
            "safe_bounds": {"min": 1, "max": 60},
        })
        assert result.passed is False

    def test_execute_env_var(self):
        p = ConfigAdjust()
        result = p.execute({
            "config_key": "HC_TEST_VAR",
            "config_value": "42",
        })
        assert result.status == "success"
        assert os.environ.get("HC_TEST_VAR") == "42"
        assert result.rollback_info is not None
        # Cleanup
        os.environ.pop("HC_TEST_VAR", None)


class TestNGLiteRebalance:
    def test_validate_missing_path(self):
        p = NGLiteRebalance()
        result = p.validate({})
        assert result.passed is False

    def test_validate_nonexistent_path(self):
        p = NGLiteRebalance()
        result = p.validate({"peer_state_path": "/nonexistent/path.json"})
        assert result.passed is False

    def test_scope(self):
        p = NGLiteRebalance()
        assert p.scope == "peer"


class TestCheckpointRestore:
    def test_validate_missing_checkpoint(self):
        p = CheckpointRestore()
        result = p.validate({"peer_state_path": "/some/path"})
        assert result.passed is False

    def test_validate_and_execute(self):
        p = CheckpointRestore()
        import json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as cp:
            json.dump({"nodes": [], "synapses": []}, cp)
            cp.flush()

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as target:
                json.dump({"nodes": [{"broken": True}]}, target)
                target.flush()

                result = p.validate({
                    "checkpoint_path": cp.name,
                    "peer_state_path": target.name,
                })
                assert result.passed is True

                exec_result = p.execute({
                    "checkpoint_path": cp.name,
                    "peer_state_path": target.name,
                })
                assert exec_result.status == "success"

                # Verify target now matches checkpoint
                with open(target.name) as f:
                    restored = json.load(f)
                assert restored == {"nodes": [], "synapses": []}

        os.unlink(cp.name)
        os.unlink(target.name)


class TestDefaultPrimitives:
    """Verify all 8 PRD primitives are registered."""

    def test_all_registered(self):
        assert "process_restart" in DEFAULT_PRIMITIVES
        assert "cache_clear" in DEFAULT_PRIMITIVES
        assert "log_and_recommend" in DEFAULT_PRIMITIVES
        assert "config_adjust" in DEFAULT_PRIMITIVES
        assert "retry_with_backoff" in DEFAULT_PRIMITIVES
        assert "ng_lite_rebalance" in DEFAULT_PRIMITIVES
        assert "checkpoint_restore" in DEFAULT_PRIMITIVES
        assert "connection_pool_reset" in DEFAULT_PRIMITIVES
        assert len(DEFAULT_PRIMITIVES) == 8

    def test_all_are_repair_primitives(self):
        for name, prim in DEFAULT_PRIMITIVES.items():
            assert isinstance(prim, RepairPrimitive), f"{name} is not a RepairPrimitive"
            assert hasattr(prim, "validate")
            assert hasattr(prim, "execute")
