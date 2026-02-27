"""
The Healing Collective — Pattern Compression

Periodically compresses similar DVS entries into synthetic
COMPRESSED_PATTERN entries.  This reduces storage while preserving
the Collective's learned knowledge — the substrate remembers the
patterns, the DVS just carries less redundant detail.

Compression is triggered by two conditions (either is sufficient):
  1. DVS fullness exceeds trigger_pct (default 80%)
  2. Time since last compression exceeds cycle_days (default 7)

The algorithm:
  1. Gather compressible entries (FAILURE_SIGNATURE and REPAIR_RECORD)
  2. Cluster by embedding similarity (threshold: 0.85)
  3. Clusters below min_cluster (3) are skipped
  4. Synthesize a COMPRESSED_PATTERN from each qualifying cluster
  5. Remove originals, add compressed entries

# ---- Changelog ----
# [2026-02-27] Claude (Opus 4.6) — Initial creation.
#   What: PatternCompressor with greedy clustering, pattern synthesis,
#         and periodic/threshold-triggered compression cycles.
#   Why:  PRD specifies DVS compression to prevent unbounded growth.
#         A system that runs for months accumulates thousands of similar
#         failure records; compression keeps the DVS lean without
#         forgetting learned patterns.
#   Settings: compression.trigger_pct=0.80, similarity_threshold=0.85,
#         min_cluster=3, cycle_days=7.
#   How:  Greedy agglomerative clustering on normalized embeddings.
#         Centroid embedding + aggregated metadata for each cluster.
#         Original entry IDs stored in compression_sources field.
# -------------------
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.config import CompressionConfig
from core.dvs import DiagnosticVectorStore, DVSEntry, DVSEntryType

logger = logging.getLogger("healing_collective.compression")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    """Result of a compression cycle."""
    clusters_found: int
    entries_compressed: int
    entries_created: int
    entries_removed: int
    duration_ms: float
    dvs_size_before: int
    dvs_size_after: int


# ---------------------------------------------------------------------------
# Pattern Compressor
# ---------------------------------------------------------------------------

class PatternCompressor:
    """DVS pattern compression engine.

    Clusters similar DVS entries and replaces them with synthetic
    COMPRESSED_PATTERN entries that preserve the aggregate knowledge.
    """

    def __init__(
        self,
        config: CompressionConfig,
        dvs: DiagnosticVectorStore,
    ):
        """
        Args:
            config: Compression configuration.
            dvs: DiagnosticVectorStore instance.
        """
        self._config = config
        self._dvs = dvs
        self._last_compression: float = 0.0
        self._compression_count = 0
        self._total_compressed = 0

    def should_compress(self) -> bool:
        """Check if compression should run.

        Returns True if DVS fullness exceeds trigger_pct OR
        time since last compression exceeds cycle_days.
        """
        dvs_stats = self._dvs.stats()
        fullness = dvs_stats.get("fullness_pct", 0.0)

        if fullness >= self._config.trigger_pct:
            return True

        if self._last_compression > 0:
            days_since = (time.time() - self._last_compression) / 86400
            if days_since >= self._config.cycle_days:
                return True

        return False

    def compress(self) -> CompressionResult:
        """Run a compression cycle.

        Clusters similar entries, synthesizes compressed patterns,
        and replaces originals with compressed versions.

        Returns:
            CompressionResult with cycle statistics.
        """
        start = time.monotonic()
        size_before = self._dvs.size

        # Gather compressible entries
        compressible = self._get_compressible_entries()
        if len(compressible) < self._config.min_cluster:
            return CompressionResult(
                clusters_found=0,
                entries_compressed=0,
                entries_created=0,
                entries_removed=0,
                duration_ms=(time.monotonic() - start) * 1000,
                dvs_size_before=size_before,
                dvs_size_after=self._dvs.size,
            )

        # Cluster by embedding similarity
        clusters = self._find_clusters(compressible)

        # Filter to qualifying clusters
        qualifying = [c for c in clusters if len(c) >= self._config.min_cluster]

        entries_removed = 0
        entries_created = 0

        for cluster in qualifying:
            # Synthesize compressed entry
            compressed = self._synthesize_pattern(cluster)
            if compressed is None:
                continue

            # Remove originals
            for entry in cluster:
                self._dvs.delete(entry.entry_id)
                entries_removed += 1

            # Add compressed entry
            self._dvs.add(compressed)
            entries_created += 1

        self._last_compression = time.time()
        self._compression_count += 1
        self._total_compressed += entries_removed

        elapsed_ms = (time.monotonic() - start) * 1000

        result = CompressionResult(
            clusters_found=len(qualifying),
            entries_compressed=entries_removed,
            entries_created=entries_created,
            entries_removed=entries_removed,
            duration_ms=elapsed_ms,
            dvs_size_before=size_before,
            dvs_size_after=self._dvs.size,
        )

        logger.info(
            "Compression: %d clusters, %d entries -> %d compressed (%.0fms)",
            len(qualifying), entries_removed, entries_created, elapsed_ms,
        )

        return result

    def _get_compressible_entries(self) -> List[DVSEntry]:
        """Gather entries eligible for compression.

        COMPRESSED_PATTERN entries are excluded (already compressed).
        STATE_SNAPSHOT and DIAGNOSTIC_LOG entries are excluded
        (ephemeral, not worth compressing).
        """
        compressible_types = {
            DVSEntryType.FAILURE_SIGNATURE.value,
            DVSEntryType.REPAIR_RECORD.value,
        }
        return [
            entry for entry in self._dvs._entries.values()
            if entry.entry_type in compressible_types
        ]

    def _find_clusters(self, entries: List[DVSEntry]) -> List[List[DVSEntry]]:
        """Cluster entries by embedding similarity.

        Uses greedy agglomerative clustering: for each unassigned entry,
        find all other unassigned entries within similarity_threshold
        and form a cluster.
        """
        if not entries:
            return []

        # Pre-compute normalized embeddings
        embeddings = []
        for entry in entries:
            emb = np.array(entry.embedding, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)

        assigned = set()
        clusters: List[List[DVSEntry]] = []

        for i, entry_i in enumerate(entries):
            if i in assigned:
                continue

            cluster = [entry_i]
            assigned.add(i)

            for j in range(i + 1, len(entries)):
                if j in assigned:
                    continue

                similarity = float(np.dot(embeddings[i], embeddings[j]))
                if similarity >= self._config.similarity_threshold:
                    cluster.append(entries[j])
                    assigned.add(j)

            clusters.append(cluster)

        return clusters

    def _synthesize_pattern(self, cluster: List[DVSEntry]) -> Optional[DVSEntry]:
        """Create a COMPRESSED_PATTERN entry from a cluster.

        The synthetic entry uses:
        - Centroid embedding (mean of cluster embeddings)
        - Aggregated content (counts, best outcome, etc.)
        - compression_sources: list of original entry IDs
        - Highest confidence from the cluster
        - Best repair outcome from the cluster
        """
        if not cluster:
            return None

        # Compute centroid embedding
        embeddings = np.array([e.embedding for e in cluster], dtype=np.float32)
        centroid = np.mean(embeddings, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Aggregate metadata
        source_ids = [e.entry_id for e in cluster]
        types = {}
        primitives = {}
        outcomes = {"success": 0, "partial": 0, "failed": 0}

        for entry in cluster:
            # Count entry types
            types[entry.entry_type] = types.get(entry.entry_type, 0) + 1

            # Count proposed primitives
            content = entry.content or {}
            prim = content.get("proposed_primitive")
            if prim:
                primitives[prim] = primitives.get(prim, 0) + 1

            # Count outcomes
            if entry.repair_outcome in outcomes:
                outcomes[entry.repair_outcome] += 1

        # Best outcome
        best_outcome = None
        if outcomes["success"] > 0:
            best_outcome = "success"
        elif outcomes["partial"] > 0:
            best_outcome = "partial"
        elif outcomes["failed"] > 0:
            best_outcome = "failed"

        # Highest confidence
        max_confidence = max(e.confidence for e in cluster)

        # Most common primitive
        dominant_primitive = None
        if primitives:
            dominant_primitive = max(primitives, key=primitives.get)

        compressed = DVSEntry.create(
            entry_type=DVSEntryType.COMPRESSED_PATTERN,
            source_module="healing_collective",
            embedding=centroid,
            content={
                "cluster_size": len(cluster),
                "entry_types": types,
                "primitives": primitives,
                "outcomes": outcomes,
                "dominant_primitive": dominant_primitive,
                "oldest_timestamp": min(e.timestamp for e in cluster),
                "newest_timestamp": max(e.timestamp for e in cluster),
            },
            confidence=max_confidence,
            repair_outcome=best_outcome,
        )
        compressed.compression_sources = source_ids

        # Carry forward access count (sum of originals)
        compressed.ttl_accesses = sum(e.ttl_accesses for e in cluster)

        return compressed

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return compression statistics."""
        return {
            "compression_count": self._compression_count,
            "total_compressed": self._total_compressed,
            "last_compression": self._last_compression,
            "trigger_pct": self._config.trigger_pct,
            "similarity_threshold": self._config.similarity_threshold,
            "min_cluster": self._config.min_cluster,
            "cycle_days": self._config.cycle_days,
        }
