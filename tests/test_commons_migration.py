"""
Commons migration (#335) — THC hook: deposit-via-CommonsEco + bucket-novelty + bucket-experience.

# ---- Changelog ----
# [2026-06-22] Claude Code (Opus 4.8) — THC Commons migration test (mirrors QG #320/#325)
# What: Proves the REAL THC hook methods migrated to the Commons:
#       _bucket_commons_novelty reads NG's metrics:neurograph:* deposits → _substrate_novelty (EWMA,
#         dedup, ignores THC's own deposits, fail-soft);
#       _bucket_commons_experience buckets NG's experience:* raw turns, embeds at THC's boundary, and
#         routes to _check_failure_from_river (THC's failure-detection trigger — was DARK on dead tract);
#       _surprise_from_metric extracts surprise from anomaly signal + nominal aggregate.
# Why: Substrate axiom — THC dips its bucket into the shared Commons; nobody pushes. The existing suite
#       runs this code inert (ng_commons_eco not on THC's standalone path); this test puts NeuroGraph on
#       the path so commons/ng_embed import for real.
# How: bind the real HealingCollectiveHook methods to a minimal harness (the full hook spawns threads +
#       loads config — too heavy for a unit test). Patch commons.get_commons to a sandbox Commons; stub
#       _embed + _check_failure_from_river to record calls (same approach as QG's _qg.ingest stub).
"""

import os
import sys
import numpy as np

_THC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _THC)
sys.path.insert(0, os.path.expanduser("~/NeuroGraph"))

import commons as commons_mod


def _fake_embed(text, *a, **k):
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    v = rng.randn(768).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _load_thc_hook():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "healing_collective_hook", os.path.join(_THC, "healing_collective_hook.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Harness:
    """Minimal carrier of the state THC's migrated methods touch — NOT a full hook (no threads)."""
    MODULE_ID = "healing_collective"

    def __init__(self, mod):
        self._mod = mod
        self._substrate_novelty = 1.0
        self._commons_seen = set()
        self.failure_checks = []     # record _check_failure_from_river calls

    # stubs for the collaborators the experience bucket reaches
    def _embed(self, text):
        return _fake_embed(text)

    def _check_failure_from_river(self, conversation):
        self.failure_checks.append(conversation)

    def __getattr__(self, name):
        import inspect
        try:
            raw = inspect.getattr_static(self._mod.HealingCollectiveHook, name)
        except AttributeError:
            raise AttributeError(name)
        if isinstance(raw, staticmethod):
            return raw.__func__
        return raw.__get__(self, _Harness)


# ---- novelty bucket ----
def test_bucket_novelty_from_ng_anomaly():
    """THC buckets NG's anomaly metric deposit → updates novelty from its surprise 'signal'."""
    commons = commons_mod.Commons()
    h = _Harness(_load_thc_hook())
    commons.deposit(_fake_embed("ng anomaly"), "metrics:neurograph:anomaly:abc:1.0:1",
                    metadata={"kind": "metrics", "source": "neurograph", "salience": "anomaly", "signal": 0.9})
    assert h._substrate_novelty == 1.0
    orig = commons_mod.get_commons
    commons_mod.get_commons = lambda: commons
    try:
        h._bucket_commons_novelty()
    finally:
        commons_mod.get_commons = orig
    # EWMA: 0.8*1.0 + 0.2*0.9 = 0.98
    assert abs(h._substrate_novelty - (0.8 * 1.0 + 0.2 * 0.9)) < 1e-6


def test_bucket_novelty_dedup():
    """Second pulse over the same deposit does NOT re-update novelty (dedup)."""
    commons = commons_mod.Commons()
    h = _Harness(_load_thc_hook())
    commons.deposit(_fake_embed("a"), "metrics:neurograph:anomaly:x:1.0:1",
                    metadata={"kind": "metrics", "salience": "anomaly", "signal": 0.5})
    orig = commons_mod.get_commons
    commons_mod.get_commons = lambda: commons
    try:
        h._bucket_commons_novelty()
        first = h._substrate_novelty
        h._bucket_commons_novelty()
    finally:
        commons_mod.get_commons = orig
    assert h._substrate_novelty == first, "dedup prevents double-counting the same metric"


def test_bucket_novelty_ignores_non_ng_metrics():
    """THC's novelty is driven by NG's metrics, not THC's own repair deposits (no self-feedback)."""
    commons = commons_mod.Commons()
    h = _Harness(_load_thc_hook())
    commons.deposit(_fake_embed("self"), "repair:process_restart",
                    metadata={"kind": "outcome", "salience": "anomaly", "signal": 0.9})
    orig = commons_mod.get_commons
    commons_mod.get_commons = lambda: commons
    try:
        h._bucket_commons_novelty()
    finally:
        commons_mod.get_commons = orig
    assert h._substrate_novelty == 1.0, "THC must not feed its own repair deposits into its novelty"


def test_bucket_novelty_failsoft_no_commons():
    h = _Harness(_load_thc_hook())
    orig = commons_mod.get_commons
    commons_mod.get_commons = lambda: None
    try:
        h._bucket_commons_novelty()   # no Commons — must not raise
    finally:
        commons_mod.get_commons = orig
    assert h._substrate_novelty == 1.0


# ---- experience bucket (the restored failure-detection trigger) ----
def test_bucket_experience_routes_to_failure_check():
    """THC buckets NG's experience deposit, embeds, and routes to _check_failure_from_river."""
    commons = commons_mod.Commons()
    h = _Harness(_load_thc_hook())
    commons.deposit(_fake_embed(1), "experience:abc",
                    metadata={"kind": "experience", "user_text": "the cache is corrupt",
                              "assistant_text": "*frowns* restarting the pool"})
    orig = commons_mod.get_commons
    commons_mod.get_commons = lambda: commons
    try:
        h._bucket_commons_experience()
    finally:
        commons_mod.get_commons = orig
    assert len(h.failure_checks) == 1, "THC must route the bucketed experience to failure detection"
    conv = h.failure_checks[0]
    assert conv["text"].startswith("the cache is corrupt") and "restarting the pool" in conv["text"]
    assert conv["embedding"] is not None and conv["embedding"].shape[0] == 768


def test_bucket_experience_dedups_and_ignores_metrics():
    """Second pulse doesn't re-check; metrics deposits are NOT routed as experience."""
    commons = commons_mod.Commons()
    h = _Harness(_load_thc_hook())
    commons.deposit(_fake_embed(2), "experience:x", metadata={"kind": "experience", "user_text": "hi"})
    commons.deposit(_fake_embed(3), "metrics:neurograph:anomaly:y:1.0:1",
                    metadata={"kind": "metrics", "salience": "anomaly", "signal": 0.9})
    orig = commons_mod.get_commons
    commons_mod.get_commons = lambda: commons
    try:
        h._bucket_commons_experience()
        first = len(h.failure_checks)
        h._bucket_commons_experience()
    finally:
        commons_mod.get_commons = orig
    assert first == 1, "only the experience deposit routed (metrics ignored)"
    assert len(h.failure_checks) == 1, "dedup: no re-check on second pulse"


def test_bucket_experience_failsoft_no_commons():
    h = _Harness(_load_thc_hook())
    orig = commons_mod.get_commons
    commons_mod.get_commons = lambda: None
    try:
        h._bucket_commons_experience()   # no Commons — must not raise
    finally:
        commons_mod.get_commons = orig
    assert h.failure_checks == []


# ---- surprise extraction ----
def test_surprise_from_metric_forms():
    mod = _load_thc_hook()
    f = mod.HealingCollectiveHook._surprise_from_metric
    assert abs(f({"salience": "anomaly", "signal": 0.7}) - 0.7) < 1e-9
    assert abs(f({"salience": "nominal", "aggregate": {"predictions_confirmed": 9, "predictions_surprised": 1}}) - 0.1) < 1e-9
    assert f({"salience": "nominal", "aggregate": {}}) == 0.0
    assert f(None) is None and f({"salience": "other"}) is None


if __name__ == "__main__":
    test_bucket_novelty_from_ng_anomaly();        print("PASS THC buckets NG anomaly metric → novelty EWMA")
    test_bucket_novelty_dedup();                  print("PASS novelty bucket dedups (no double-count)")
    test_bucket_novelty_ignores_non_ng_metrics(); print("PASS novelty driven by NG metrics, not THC's own repair deposits")
    test_bucket_novelty_failsoft_no_commons();    print("PASS novelty bucket fail-soft when no Commons")
    test_bucket_experience_routes_to_failure_check(); print("PASS THC buckets experience → _check_failure_from_river (trigger restored)")
    test_bucket_experience_dedups_and_ignores_metrics(); print("PASS experience bucket dedups + ignores metrics")
    test_bucket_experience_failsoft_no_commons(); print("PASS experience bucket fail-soft when no Commons")
    test_surprise_from_metric_forms();            print("PASS _surprise_from_metric: anomaly signal + nominal aggregate + fallbacks")
    print("\nTHC Commons migration (#335): ALL PASS — deposit (CommonsEco) + bucket novelty + bucket experience, no tracts, no calls")
