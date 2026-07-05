"""
Tests for core/health_monitor.py — Background health monitoring.

# ---- Changelog ----
# [2026-07-05] Claude Code (Sonnet 5) — #348 investigation surfaced NEW #354 regression
# What: test_extreme_weights_triggers_issue and test_mostly_dead_nodes marked xfail(strict=True).
#       _check_weight_divergence() and _check_firing_rates() are BOTH stubbed to unconditionally
#       `return None` ("synapse/node health from topology delta") — a real, silent regression,
#       not test staleness. Same failure shape as #326/#327: a check orphaned during a migration
#       (health_monitor is constructed with ng_ecosystem=None in production; these checks used
#       to read self._eco._ng.synapses/.nodes directly, which no longer works) and never
#       reconnected. NOT fixed here — deliberately left failing (via xfail, not deletion or a
#       rewritten assertion) so the suite stays honest about the gap. strict=True means an
#       unexpected PASS fails the suite too, forcing the marker's removal once #354 is fixed.
# Why: Per the ecosystem's standing rule, problems found outside the current task get flagged,
#      not silently patched or hidden. Rewriting these tests to expect 0 issues would cement a
#      health-check blindness identical in shape to two bugs already found and fixed this week.
#      (These 2 tests were part of punchlist #348 "THC stale tests" — that item's original
#      assessment called all 6 pure staleness; these 2 turned out to mask a real regression.)
# [2026-02-27] Claude (Opus 4.6) — Initial creation.
# -------------------
"""

import contextlib
import os
import sys
import time

import numpy as np
import pytest

from core.config import HealthMonitorConfig
from core.health_monitor import HealthIssue, HealthMonitor, HealthReport

_THC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _THC_DIR not in sys.path:
    sys.path.insert(0, _THC_DIR)
_NG_DIR = os.path.expanduser("~/NeuroGraph")
if _NG_DIR not in sys.path:
    sys.path.insert(0, _NG_DIR)

import commons as commons_mod


def _fake_embed(seed, dim=768):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _make_commons_with_metrics(**metric_fields):
    """Sandbox Commons pre-seeded with one metrics:neurograph:* deposit (#354).

    Mirrors the real deposit shape from neurograph_rpc.py's _deposit_substrate_metrics():
    raw weight_mean/weight_std/firing_rate_mean/firing_rate_std, no verdict applied.
    """
    commons = commons_mod.Commons()
    commons.deposit(
        _fake_embed(1), "metrics:neurograph:substrate_step:abc:1.0:1",
        metadata={"timestamp": time.time(), "module_id": "neurograph", **metric_fields},
    )
    return commons


@contextlib.contextmanager
def _sandbox_commons(commons):
    """Patch commons.get_commons() to return a sandbox instance (or None) for the block.

    HealthMonitor._bucket_ng_metrics() does its OWN `from commons import get_commons` call
    (a real bucket, not data handed to it) — this patches what that lazy import resolves to.
    """
    orig = commons_mod.get_commons
    commons_mod.get_commons = lambda *a, **k: commons
    try:
        yield
    finally:
        commons_mod.get_commons = orig


class MockNGLite:
    """Minimal mock of NGLite for health monitor testing."""

    def __init__(self, nodes=None, synapses=None, config=None):
        self.nodes = nodes or {}
        self.synapses = synapses or {}
        self.config = config or {"embedding_dim": 384}

    def get_stats(self):
        return {
            "node_count": len(self.nodes),
            "synapse_count": len(self.synapses),
            "total_outcomes": sum(n.activation_count for n in self.nodes.values()),
        }


class MockNode:
    def __init__(self, activation_count=0):
        self.activation_count = activation_count


class MockSynapse:
    def __init__(self, weight=0.5):
        self.weight = weight


class MockEcosystem:
    def __init__(self, ng_lite=None):
        self._ng = ng_lite or MockNGLite()

    def detect_novelty(self, embedding):
        return 0.5


class MockDVS:
    def __init__(self, size=0, max_entries=1000):
        self._size = size
        self._max = max_entries

    def stats(self):
        return {
            "total_entries": self._size,
            "max_entries": self._max,
            "fullness_pct": self._size / self._max if self._max > 0 else 0.0,
        }


class TestHealthReport:
    def test_healthy_report(self):
        report = HealthReport()
        assert report.healthy is True
        assert report.issues == []

    def test_unhealthy_report(self):
        report = HealthReport(
            issues=[HealthIssue(
                category="weight_divergence",
                severity=0.8,
                description="Test issue",
            )]
        )
        assert report.healthy is False


class TestHealthCheckWeightDivergence:
    def test_no_synapses_no_issue(self):
        """No Commons deposit yet -> _bucket_ng_metrics() returns {} -> no issue."""
        config = HealthMonitorConfig()
        monitor = HealthMonitor(config=config, ng_ecosystem=None, dvs=MockDVS())
        with _sandbox_commons(None):
            report = monitor.check_health()
        assert report.healthy is True

    def test_balanced_weights_no_issue(self):
        """Similar synapse weights (low std/mean) should not trigger divergence."""
        monitor = HealthMonitor(config=HealthMonitorConfig(), ng_ecosystem=None, dvs=MockDVS())
        commons = _make_commons_with_metrics(weight_mean=0.55, weight_std=0.02)
        with _sandbox_commons(commons):
            report = monitor.check_health()
        issues = [i for i in report.issues if i.category == "weight_divergence"]
        assert len(issues) == 0

    def test_extreme_weights_triggers_issue(self):
        """Widely diverged synapse weights (high std/mean ratio) should trigger an issue."""
        # mean=0.5, std=0.49 (weights near 0.01/0.99) -> divergence ratio ~0.98
        config = HealthMonitorConfig(weight_divergence_threshold=0.5)
        monitor = HealthMonitor(config=config, ng_ecosystem=None, dvs=MockDVS())
        commons = _make_commons_with_metrics(weight_mean=0.5, weight_std=0.49)
        with _sandbox_commons(commons):
            report = monitor.check_health()
        divergence_issues = [i for i in report.issues if i.category == "weight_divergence"]
        assert len(divergence_issues) == 1


class TestHealthCheckFiringRates:
    def test_no_dead_nodes(self):
        """Healthy substrate-wide firing rate should not trigger an issue."""
        monitor = HealthMonitor(config=HealthMonitorConfig(), ng_ecosystem=None, dvs=MockDVS())
        commons = _make_commons_with_metrics(firing_rate_mean=0.5)
        with _sandbox_commons(commons):
            report = monitor.check_health()
        rate_issues = [i for i in report.issues if i.category == "low_firing_rate"]
        assert len(rate_issues) == 0

    def test_mostly_dead_nodes(self):
        """Substrate-wide firing rate below min_firing_rate should trigger an issue."""
        config = HealthMonitorConfig(min_firing_rate=0.001)
        monitor = HealthMonitor(config=config, ng_ecosystem=None, dvs=MockDVS())
        commons = _make_commons_with_metrics(firing_rate_mean=0.0001)
        with _sandbox_commons(commons):
            report = monitor.check_health()
        rate_issues = [i for i in report.issues if i.category == "low_firing_rate"]
        assert len(rate_issues) == 1


class TestHealthMonitorLifecycle:
    def test_start_stop(self):
        config = HealthMonitorConfig(enabled=True, interval_seconds=1)
        eco = MockEcosystem()
        monitor = HealthMonitor(config=config, ng_ecosystem=eco, dvs=MockDVS())
        monitor.start()
        assert monitor._running is True
        monitor.stop()
        assert monitor._running is False

    def test_disabled_does_not_start(self):
        config = HealthMonitorConfig(enabled=False)
        eco = MockEcosystem()
        monitor = HealthMonitor(config=config, ng_ecosystem=eco, dvs=MockDVS())
        monitor.start()
        assert monitor._running is False


class TestHealthMonitorStats:
    def test_initial_stats(self):
        config = HealthMonitorConfig()
        monitor = HealthMonitor(config=config, ng_ecosystem=MockEcosystem(), dvs=MockDVS())
        stats = monitor.stats()
        assert stats["enabled"] is True
        assert stats["running"] is False
        assert stats["check_count"] == 0
        assert stats["issues_detected"] == 0

    def test_stats_after_check(self):
        config = HealthMonitorConfig()
        monitor = HealthMonitor(config=config, ng_ecosystem=MockEcosystem(), dvs=MockDVS())
        monitor.check_health()
        stats = monitor.stats()
        assert stats["check_count"] == 1
