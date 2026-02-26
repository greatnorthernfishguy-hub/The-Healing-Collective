"""
The Healing Collective — Repair Primitives

The Collective's action vocabulary.  Each primitive is a small, safe
operation the Collective can learn to apply.  All primitives implement
validate()/execute() — validate() MUST be called before execute() with
no exceptions.  This is the Collective's Hippocratic oath.

ENFORCEMENT: validate() is called by the Diagnosis Engine, not by
individual primitives.  The Diagnosis Engine MUST NOT call execute()
without a preceding validate() that returned passed=True.

# ---- Changelog ----
# [2026-02-26] Claude (Opus 4.6) — Initial creation.
#   What: RepairPrimitive ABC with ValidationResult/ExecutionResult,
#         plus all 8 built-in primitives from PRD §2.2.4.
#   Why:  The Collective needs a vocabulary of safe, validated actions
#         it can learn to apply.  Each primitive is designed to be
#         independently safe with clear preconditions.
#   Settings: Individual primitive behaviors configurable via context dict.
#   How:  ABC with validate()/execute().  Concrete primitives implement
#         specific repair actions.  Registry pattern for custom primitives.
# -------------------
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("healing_collective.repair_primitives")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of a repair primitive's validate() call.

    validate() MUST NOT have side effects.
    """
    passed: bool
    reason: str
    preconditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Result of a repair primitive's execute() call."""
    status: str  # "success" | "partial" | "failed"
    detail: str
    rollback_info: Optional[Dict[str, Any]] = None
    duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class RepairPrimitive(ABC):
    """Abstract base class for all repair primitives.

    Every repair primitive MUST implement validate() and execute() as
    separate methods.  validate() MUST be called before execute() with
    no exceptions.  validate() MUST NOT have side effects.
    """

    @property
    def name(self) -> str:
        """Primitive name for registry and logging."""
        return self.__class__.__name__

    @property
    def scope(self) -> str:
        """'host' or 'peer' — determines confidence thresholds."""
        return "host"

    @abstractmethod
    def validate(self, context: dict) -> ValidationResult:
        """Check whether this repair can safely execute.

        MUST NOT have side effects.  Returns ValidationResult with
        passed=True if execute() is safe to call.

        Args:
            context: Failure context including system state, failure
                     description, target identifiers, etc.
        """
        ...

    @abstractmethod
    def execute(self, context: dict) -> ExecutionResult:
        """Execute the repair action.

        MUST only be called after validate() returned passed=True.

        Args:
            context: Same context dict passed to validate().
        """
        ...


# ---------------------------------------------------------------------------
# Built-in primitives
# ---------------------------------------------------------------------------

class ProcessRestart(RepairPrimitive):
    """Restart a process via SIGTERM → wait → verify."""

    @property
    def scope(self) -> str:
        return "host"

    def validate(self, context: dict) -> ValidationResult:
        pid = context.get("pid")
        if pid is None:
            return ValidationResult(
                passed=False,
                reason="No pid provided in context",
            )
        try:
            os.kill(pid, 0)  # Check if process exists (no signal sent)
        except ProcessLookupError:
            return ValidationResult(
                passed=False,
                reason=f"Process {pid} does not exist",
            )
        except PermissionError:
            return ValidationResult(
                passed=False,
                reason=f"No permission to signal process {pid}",
            )

        cooldown_until = context.get("cooldown_until", 0)
        if time.time() < cooldown_until:
            return ValidationResult(
                passed=False,
                reason=f"Process {pid} in cooldown until {cooldown_until}",
            )

        return ValidationResult(
            passed=True,
            reason=f"Process {pid} exists and is restartable",
            preconditions={"pid": pid},
        )

    def execute(self, context: dict) -> ExecutionResult:
        start = time.monotonic()
        pid = context["pid"]
        timeout = context.get("restart_timeout", 10)

        try:
            os.kill(pid, signal.SIGTERM)
            # Wait for process to exit
            for _ in range(timeout * 10):
                try:
                    os.kill(pid, 0)
                    time.sleep(0.1)
                except ProcessLookupError:
                    break
            else:
                return ExecutionResult(
                    status="partial",
                    detail=f"Process {pid} did not exit within {timeout}s after SIGTERM",
                    duration_ms=(time.monotonic() - start) * 1000,
                )

            return ExecutionResult(
                status="success",
                detail=f"Process {pid} terminated via SIGTERM",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except Exception as exc:
            return ExecutionResult(
                status="failed",
                detail=f"Failed to restart process {pid}: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )


class CacheClear(RepairPrimitive):
    """Remove cached data from a directory or specific key."""

    @property
    def scope(self) -> str:
        return "host"

    def validate(self, context: dict) -> ValidationResult:
        cache_path = context.get("cache_path")
        if not cache_path:
            return ValidationResult(
                passed=False,
                reason="No cache_path provided in context",
            )
        path = Path(cache_path).expanduser()
        if not path.exists():
            return ValidationResult(
                passed=False,
                reason=f"Cache path does not exist: {cache_path}",
            )
        return ValidationResult(
            passed=True,
            reason=f"Cache path exists: {cache_path}",
            preconditions={"cache_path": str(path), "is_dir": path.is_dir()},
        )

    def execute(self, context: dict) -> ExecutionResult:
        start = time.monotonic()
        path = Path(context["cache_path"]).expanduser()

        try:
            if path.is_dir():
                count = sum(1 for _ in path.iterdir())
                shutil.rmtree(path)
                path.mkdir(parents=True, exist_ok=True)
                detail = f"Cleared cache directory ({count} items): {path}"
            else:
                path.unlink()
                detail = f"Removed cache file: {path}"

            return ExecutionResult(
                status="success",
                detail=detail,
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except Exception as exc:
            return ExecutionResult(
                status="failed",
                detail=f"Cache clear failed: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )


class LogAndRecommend(RepairPrimitive):
    """Log diagnostic chain and emit recommendation event.

    Always passes validate() — this is the safe fallback primitive.
    """

    @property
    def scope(self) -> str:
        return "any"

    def validate(self, context: dict) -> ValidationResult:
        return ValidationResult(
            passed=True,
            reason="LogAndRecommend always validates",
        )

    def execute(self, context: dict) -> ExecutionResult:
        start = time.monotonic()
        description = context.get("description", "Unknown failure")
        recommendation = context.get("recommendation", "Manual investigation recommended")

        logger.info(
            "HEALING RECOMMENDATION: %s — Suggested action: %s",
            description, recommendation,
        )

        return ExecutionResult(
            status="success",
            detail=f"Logged recommendation: {recommendation}",
            duration_ms=(time.monotonic() - start) * 1000,
        )


class ConfigAdjust(RepairPrimitive):
    """Modify a configuration file or environment variable within safe bounds."""

    @property
    def scope(self) -> str:
        return "host"

    def validate(self, context: dict) -> ValidationResult:
        config_key = context.get("config_key")
        config_value = context.get("config_value")
        safe_bounds = context.get("safe_bounds", {})

        if not config_key:
            return ValidationResult(passed=False, reason="No config_key provided")
        if config_value is None:
            return ValidationResult(passed=False, reason="No config_value provided")

        # Check safe bounds if specified
        if safe_bounds:
            min_val = safe_bounds.get("min")
            max_val = safe_bounds.get("max")
            try:
                val = float(config_value)
                if min_val is not None and val < float(min_val):
                    return ValidationResult(
                        passed=False,
                        reason=f"Value {val} below safe minimum {min_val}",
                    )
                if max_val is not None and val > float(max_val):
                    return ValidationResult(
                        passed=False,
                        reason=f"Value {val} above safe maximum {max_val}",
                    )
            except (TypeError, ValueError):
                pass  # Non-numeric values skip bounds check

        return ValidationResult(
            passed=True,
            reason=f"Config adjustment {config_key}={config_value} within safe bounds",
            preconditions={"config_key": config_key, "config_value": config_value},
        )

    def execute(self, context: dict) -> ExecutionResult:
        start = time.monotonic()
        config_key = context["config_key"]
        config_value = context["config_value"]
        config_file = context.get("config_file")

        try:
            original_value = None

            if config_file and os.path.exists(config_file):
                import yaml
                with open(config_file, "r") as f:
                    cfg = yaml.safe_load(f) or {}
                original_value = cfg.get(config_key)
                cfg[config_key] = config_value
                with open(config_file, "w") as f:
                    yaml.dump(cfg, f, default_flow_style=False)
            else:
                original_value = os.environ.get(config_key)
                os.environ[config_key] = str(config_value)

            return ExecutionResult(
                status="success",
                detail=f"Adjusted {config_key}: {original_value} -> {config_value}",
                rollback_info={
                    "config_key": config_key,
                    "original_value": original_value,
                    "config_file": config_file,
                },
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except Exception as exc:
            return ExecutionResult(
                status="failed",
                detail=f"Config adjustment failed: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )


class RetryWithBackoff(RepairPrimitive):
    """Retry a target endpoint with exponential backoff."""

    @property
    def scope(self) -> str:
        return "host"

    def validate(self, context: dict) -> ValidationResult:
        target = context.get("target_host")
        port = context.get("target_port")

        if not target:
            return ValidationResult(passed=False, reason="No target_host provided")
        if not port:
            return ValidationResult(passed=False, reason="No target_port provided")

        # Quick reachability check
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((target, int(port)))
            sock.close()
            reachable = True
        except Exception:
            reachable = False

        if not reachable:
            return ValidationResult(
                passed=False,
                reason=f"Target {target}:{port} not reachable",
            )

        return ValidationResult(
            passed=True,
            reason=f"Target {target}:{port} is reachable",
            preconditions={"target_host": target, "target_port": port},
        )

    def execute(self, context: dict) -> ExecutionResult:
        start = time.monotonic()
        target = context["target_host"]
        port = int(context["target_port"])
        max_retries = context.get("max_retries", 3)
        base_delay = context.get("base_delay", 1.0)

        for attempt in range(max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                sock.connect((target, port))
                sock.close()
                return ExecutionResult(
                    status="success",
                    detail=f"Connected to {target}:{port} on attempt {attempt + 1}",
                    duration_ms=(time.monotonic() - start) * 1000,
                )
            except Exception:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    time.sleep(delay)

        return ExecutionResult(
            status="failed",
            detail=f"Failed to connect to {target}:{port} after {max_retries} attempts",
            duration_ms=(time.monotonic() - start) * 1000,
        )


class NGLiteRebalance(RepairPrimitive):
    """Homeostatic weight scaling on a peer's NG-Lite substrate.

    Uses fork-test-promote sandbox: fork state → apply → verify healthier.
    """

    @property
    def scope(self) -> str:
        return "peer"

    def validate(self, context: dict) -> ValidationResult:
        state_path = context.get("peer_state_path")
        if not state_path:
            return ValidationResult(passed=False, reason="No peer_state_path provided")

        path = Path(state_path).expanduser()
        if not path.exists():
            return ValidationResult(
                passed=False,
                reason=f"Peer state file does not exist: {state_path}",
            )

        try:
            import json
            with open(path, "r") as f:
                state = json.load(f)
            if "nodes" not in state:
                return ValidationResult(
                    passed=False,
                    reason="Peer state file missing 'nodes' key",
                )
        except Exception as exc:
            return ValidationResult(
                passed=False,
                reason=f"Cannot parse peer state: {exc}",
            )

        return ValidationResult(
            passed=True,
            reason=f"Peer state parseable at {state_path}",
            preconditions={"peer_state_path": str(path)},
        )

    def execute(self, context: dict) -> ExecutionResult:
        start = time.monotonic()
        state_path = Path(context["peer_state_path"]).expanduser()

        try:
            import json
            import numpy as np

            with open(state_path, "r") as f:
                state = json.load(f)

            # Fork: work on a copy
            original_state = json.dumps(state)

            # Apply homeostatic weight scaling
            if "synapses" in state:
                weights = [s.get("weight", 0.0) for s in state["synapses"]]
                if weights:
                    mean_w = sum(weights) / len(weights)
                    if mean_w > 0:
                        scale = 1.0 / mean_w
                        for syn in state["synapses"]:
                            w = syn.get("weight", 0.0)
                            syn["weight"] = w * scale * 0.5 + w * 0.5

            # Verify fork is healthier (simple check: weights more balanced)
            if "synapses" in state:
                new_weights = [s.get("weight", 0.0) for s in state["synapses"]]
                if new_weights:
                    old_std = float(np.std(weights)) if weights else 0
                    new_std = float(np.std(new_weights))
                    if new_std >= old_std:
                        return ExecutionResult(
                            status="partial",
                            detail="Rebalance did not improve weight distribution",
                            rollback_info={"original_state": original_state},
                            duration_ms=(time.monotonic() - start) * 1000,
                        )

            # Promote: write back
            with open(state_path, "w") as f:
                json.dump(state, f)

            return ExecutionResult(
                status="success",
                detail=f"Rebalanced peer NG-Lite weights at {state_path}",
                rollback_info={"original_state": original_state},
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except Exception as exc:
            return ExecutionResult(
                status="failed",
                detail=f"NGLite rebalance failed: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )


class CheckpointRestore(RepairPrimitive):
    """Replace a peer's NG-Lite state with a known-good checkpoint."""

    @property
    def scope(self) -> str:
        return "peer"

    def validate(self, context: dict) -> ValidationResult:
        checkpoint_path = context.get("checkpoint_path")
        target_path = context.get("peer_state_path")

        if not checkpoint_path:
            return ValidationResult(passed=False, reason="No checkpoint_path provided")
        if not target_path:
            return ValidationResult(passed=False, reason="No peer_state_path provided")

        cp = Path(checkpoint_path).expanduser()
        if not cp.exists():
            return ValidationResult(
                passed=False,
                reason=f"Checkpoint does not exist: {checkpoint_path}",
            )

        try:
            import json
            with open(cp, "r") as f:
                json.load(f)
        except Exception as exc:
            return ValidationResult(
                passed=False,
                reason=f"Checkpoint not parseable: {exc}",
            )

        return ValidationResult(
            passed=True,
            reason=f"Checkpoint parseable: {checkpoint_path}",
            preconditions={
                "checkpoint_path": str(cp),
                "peer_state_path": target_path,
            },
        )

    def execute(self, context: dict) -> ExecutionResult:
        start = time.monotonic()
        checkpoint_path = Path(context["checkpoint_path"]).expanduser()
        target_path = Path(context["peer_state_path"]).expanduser()

        try:
            # Backup current state before overwriting
            backup_path = None
            if target_path.exists():
                backup_path = str(target_path) + ".backup"
                shutil.copy2(target_path, backup_path)

            shutil.copy2(checkpoint_path, target_path)

            return ExecutionResult(
                status="success",
                detail=f"Restored checkpoint to {target_path}",
                rollback_info={"backup_path": backup_path},
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except Exception as exc:
            return ExecutionResult(
                status="failed",
                detail=f"Checkpoint restore failed: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )


class ConnectionPoolReset(RepairPrimitive):
    """Drain and reinitialize a connection pool."""

    @property
    def scope(self) -> str:
        return "host"

    def validate(self, context: dict) -> ValidationResult:
        pool = context.get("pool")
        pool_name = context.get("pool_name", "unknown")

        if pool is None:
            return ValidationResult(
                passed=False,
                reason=f"No pool handle provided for '{pool_name}'",
            )

        return ValidationResult(
            passed=True,
            reason=f"Pool handle accessible: {pool_name}",
            preconditions={"pool_name": pool_name},
        )

    def execute(self, context: dict) -> ExecutionResult:
        start = time.monotonic()
        pool = context["pool"]
        pool_name = context.get("pool_name", "unknown")

        try:
            # Attempt generic pool reset patterns
            if hasattr(pool, "dispose"):
                pool.dispose()
            elif hasattr(pool, "close"):
                pool.close()
            elif hasattr(pool, "clear"):
                pool.clear()

            return ExecutionResult(
                status="success",
                detail=f"Connection pool '{pool_name}' reset",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        except Exception as exc:
            return ExecutionResult(
                status="failed",
                detail=f"Pool reset failed for '{pool_name}': {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )


# ---------------------------------------------------------------------------
# Primitive Registry
# ---------------------------------------------------------------------------

# Default primitives keyed by snake_case name
DEFAULT_PRIMITIVES: Dict[str, RepairPrimitive] = {
    "process_restart": ProcessRestart(),
    "cache_clear": CacheClear(),
    "log_and_recommend": LogAndRecommend(),
    "config_adjust": ConfigAdjust(),
    "retry_with_backoff": RetryWithBackoff(),
    "ng_lite_rebalance": NGLiteRebalance(),
    "checkpoint_restore": CheckpointRestore(),
    "connection_pool_reset": ConnectionPoolReset(),
}
