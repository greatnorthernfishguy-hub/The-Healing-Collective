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

import time

import numpy as np
import pytest

from core.config import HealthMonitorConfig
from core.health_monitor import HealthIssue, HealthMonitor, HealthReport


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
        config = HealthMonitorConfig()
        eco = MockEcosystem(MockNGLite())
        monitor = HealthMonitor(config=config, ng_ecosystem=eco, dvs=MockDVS())
        report = monitor.check_health()
        assert report.healthy is True

    def test_balanced_weights_no_issue(self):
        """Synapses with similar weights should not trigger divergence."""
        synapses = {
            ("a", "b"): MockSynapse(0.5),
            ("a", "c"): MockSynapse(0.6),
            ("b", "c"): MockSynapse(0.55),
        }
        ng = MockNGLite(synapses=synapses)
        eco = MockEcosystem(ng)
        monitor = HealthMonitor(
            config=HealthMonitorConfig(), ng_ecosystem=eco, dvs=MockDVS(),
        )
        report = monitor.check_health()
        # Low std_dev / mean should not exceed threshold
        issues = [i for i in report.issues if i.category == "weight_divergence"]
        assert len(issues) == 0

    @pytest.mark.xfail(
        reason="#354: _check_weight_divergence() is stubbed to `return None` unconditionally "
                "(orphaned during the ng_ecosystem=None migration, never reconnected to the "
                "Commons) — a real regression, not test staleness. See test file changelog.",
        strict=True,
    )
    def test_extreme_weights_triggers_issue(self):
        """Synapses with extreme divergence should trigger an issue."""
        synapses = {
            ("a", "b"): MockSynapse(0.01),
            ("a", "c"): MockSynapse(0.99),
            ("b", "c"): MockSynapse(0.01),
            ("c", "d"): MockSynapse(0.99),
        }
        ng = MockNGLite(synapses=synapses)
        eco = MockEcosystem(ng)
        # Use a very low threshold to ensure it triggers
        config = HealthMonitorConfig(weight_divergence_threshold=0.5)
        monitor = HealthMonitor(config=config, ng_ecosystem=eco, dvs=MockDVS())
        report = monitor.check_health()
        divergence_issues = [i for i in report.issues if i.category == "weight_divergence"]
        assert len(divergence_issues) == 1


class TestHealthCheckFiringRates:
    def test_no_dead_nodes(self):
        """All active nodes should not trigger firing rate issue."""
        nodes = {
            "a": MockNode(activation_count=100),
            "b": MockNode(activation_count=80),
        }
        ng = MockNGLite(nodes=nodes)
        eco = MockEcosystem(ng)
        monitor = HealthMonitor(
            config=HealthMonitorConfig(), ng_ecosystem=eco, dvs=MockDVS(),
        )
        report = monitor.check_health()
        rate_issues = [i for i in report.issues if i.category == "low_firing_rate"]
        assert len(rate_issues) == 0

    @pytest.mark.xfail(
        reason="#354: _check_firing_rates() is stubbed to `return None` unconditionally "
                "(orphaned during the ng_ecosystem=None migration, never reconnected to the "
                "Commons) — a real regression, not test staleness. See test file changelog.",
        strict=True,
    )
    def test_mostly_dead_nodes(self):
        """Majority dead nodes should trigger firing rate issue."""
        nodes = {
            f"n_{i}": MockNode(activation_count=0) for i in range(10)
        }
        # One active node to ensure total_outcomes > 0
        nodes["active"] = MockNode(activation_count=1000)
        ng = MockNGLite(nodes=nodes)
        eco = MockEcosystem(ng)
        config = HealthMonitorConfig(min_firing_rate=0.001)
        monitor = HealthMonitor(config=config, ng_ecosystem=eco, dvs=MockDVS())
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
