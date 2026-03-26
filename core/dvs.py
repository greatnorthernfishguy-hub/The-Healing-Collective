"""
The Healing Collective — Diagnostic Vector Store (DVS)

A compact, NG-Lite-augmented vector database for healing operations.
Stores failure signatures, repair transcripts, system state snapshots,
and diagnostic context with semantic embeddings.

Key design: The DVS routes ALL search through the NG-Lite substrate's
learned topology.  When the Collective queries for similar failures,
results are ranked by the substrate's learned associations — what the
system has empirically connected to the query pattern — not merely by
cosine similarity.

Persistence: Primary format is msgpack.
File: ~/.et_modules/healing_collective/dvs.msgpack

# ---- Changelog ----
# [2026-02-26] Claude (Opus 4.6) — Initial creation.
#   What: DiagnosticVectorStore with msgpack persistence, LRU eviction,
#         cosine search (Phase 1), and NG-Lite-augmented search (Phase 2).
#   Why:  PRD §2.2.3 specifies a compact vector DB with substrate-routed
#         search.  The DVS is the episodic memory of the Collective.
#   Settings: dvs_max_entries=10000, dvs_search_top_k=10,
#         dvs_persistence_format=msgpack.
#   How:  DVSEntry dataclass with msgpack serialization.  Search pipeline:
#         Embed → Activate → Propagate → Harvest → Return.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("healing_collective.dvs")


# ---------------------------------------------------------------------------
# Entry types
# ---------------------------------------------------------------------------

class DVSEntryType(str, Enum):
    """Types of DVS entries per PRD §2.2.3."""
    FAILURE_SIGNATURE = "FAILURE_SIGNATURE"
    REPAIR_RECORD = "REPAIR_RECORD"
    STATE_SNAPSHOT = "STATE_SNAPSHOT"
    DIAGNOSTIC_LOG = "DIAGNOSTIC_LOG"
    COMPRESSED_PATTERN = "COMPRESSED_PATTERN"


# ---------------------------------------------------------------------------
# DVS Entry
# ---------------------------------------------------------------------------

@dataclass
class DVSEntry:
    """A single entry in the Diagnostic Vector Store.

    Schema per PRD §2.2.3.
    """
    entry_id: str
    entry_type: str  # DVSEntryType value
    timestamp: float
    source_module: str
    embedding: List[float]
    content: Dict[str, Any]
    ng_node_id: Optional[int] = None
    repair_outcome: Optional[str] = None  # "success" | "partial" | "failed" | None
    confidence: float = 0.0
    compression_sources: Optional[List[str]] = None
    ttl_accesses: int = 0

    @staticmethod
    def create(
        entry_type: DVSEntryType,
        source_module: str,
        embedding: np.ndarray,
        content: Dict[str, Any],
        confidence: float = 0.0,
        repair_outcome: Optional[str] = None,
    ) -> "DVSEntry":
        """Factory method to create a new DVS entry."""
        return DVSEntry(
            entry_id=str(uuid.uuid4()),
            entry_type=entry_type.value,
            timestamp=time.time(),
            source_module=source_module,
            embedding=embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding),
            content=content,
            confidence=confidence,
            repair_outcome=repair_outcome,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for msgpack/JSON."""
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "DVSEntry":
        """Deserialize from dict."""
        return DVSEntry(**d)

    def embedding_array(self) -> np.ndarray:
        """Return embedding as numpy array."""
        return np.array(self.embedding, dtype=np.float32)


# ---------------------------------------------------------------------------
# Diagnostic Vector Store
# ---------------------------------------------------------------------------

class DiagnosticVectorStore:
    """NG-Lite-augmented vector database for healing operations.

    Phase 1: Basic cosine similarity search.
    Phase 2: Search routes through NG-Lite substrate's learned topology.
    """

    def __init__(
        self,
        max_entries: int = 10000,
        persistence_path: Optional[str] = None,
        ng_lite: Optional[Any] = None,
        config: Optional[Any] = None,
    ):
        """
        Args:
            max_entries: Maximum entries before LRU eviction.
            persistence_path: Path to dvs.msgpack file.
            ng_lite: NGLite instance for substrate-augmented search (Phase 2).
            config: HealingCollectiveConfig for SVG Phase 3 tunables.
        """
        self._entries: Dict[str, DVSEntry] = {}
        self._max_entries = max_entries
        self._persistence_path = persistence_path
        self._ng_lite = ng_lite
        self._config = config

        if persistence_path and os.path.exists(persistence_path):
            self._load(persistence_path)

    @property
    def size(self) -> int:
        """Number of entries currently stored."""
        return len(self._entries)

    def set_ng_lite(self, ng_lite: Any) -> None:
        """Attach or replace the NG-Lite substrate for augmented search."""
        self._ng_lite = ng_lite

    # -----------------------------------------------------------------
    # CRUD
    # -----------------------------------------------------------------

    def add(self, entry: DVSEntry) -> str:
        """Add an entry to the DVS.  Returns the entry_id.

        Triggers LRU eviction if at capacity.
        """
        if len(self._entries) >= self._max_entries:
            self._evict_lru()

        self._entries[entry.entry_id] = entry
        return entry.entry_id

    def get(self, entry_id: str) -> Optional[DVSEntry]:
        """Retrieve an entry by ID.  Increments access counter."""
        entry = self._entries.get(entry_id)
        if entry is not None:
            entry.ttl_accesses += 1
        return entry

    def update(self, entry_id: str, **kwargs: Any) -> bool:
        """Update fields on an existing entry."""
        entry = self._entries.get(entry_id)
        if entry is None:
            return False
        for key, val in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, val)
        return True

    def delete(self, entry_id: str) -> bool:
        """Remove an entry by ID."""
        return self._entries.pop(entry_id, None) is not None

    def get_by_type(self, entry_type: DVSEntryType) -> List[DVSEntry]:
        """Return all entries of a specific type."""
        return [
            e for e in self._entries.values()
            if e.entry_type == entry_type.value
        ]

    # -----------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        entry_type: Optional[DVSEntryType] = None,
    ) -> List[Tuple[DVSEntry, float]]:
        """Search DVS using substrate-augmented or cosine similarity.

        Phase 2 (with NG-Lite): Routes through substrate's learned topology.
        Phase 1 (fallback): Plain cosine similarity.

        Args:
            query_embedding: Query vector (384-dim or 768-dim).
            top_k: Max results to return.
            entry_type: Filter to specific entry type.

        Returns:
            List of (entry, score) tuples sorted by relevance descending.
            Each returned entry has its ttl_accesses incremented.
        """
        if self._ng_lite is not None:
            return self._substrate_search(query_embedding, top_k, entry_type)
        return self._cosine_search(query_embedding, top_k, entry_type)

    def _cosine_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        entry_type: Optional[DVSEntryType],
    ) -> List[Tuple[DVSEntry, float]]:
        """Phase 1 fallback: plain cosine similarity search."""
        query = query_embedding.astype(np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        query = query / query_norm

        scored: List[Tuple[DVSEntry, float]] = []
        for entry in self._entries.values():
            if entry_type and entry.entry_type != entry_type.value:
                continue
            entry_emb = entry.embedding_array()
            entry_norm = np.linalg.norm(entry_emb)
            if entry_norm == 0:
                continue
            entry_emb = entry_emb / entry_norm
            sim = float(np.dot(query, entry_emb))
            scored.append((entry, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        results = scored[:top_k]

        for entry, _ in results:
            entry.ttl_accesses += 1

        return results

    def _substrate_search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        entry_type: Optional[DVSEntryType],
    ) -> List[Tuple[DVSEntry, float]]:
        """Phase 2: NG-Lite-augmented search.

        Pipeline per PRD §5.2:
        1. Embed: query_embedding already provided
        2. Activate: find_or_create_node() on substrate
        3. Propagate: traverse outgoing synapses weighted by confidence
        4. Harvest: collect DVS entries whose nodes were activated
        5. Return: multi-factor ranking
        """
        try:
            # Step 2: Activate — find the closest node in substrate
            node = self._ng_lite.find_or_create_node(query_embedding)

            # Step 3: Propagate — get recommendations (traverses learned topology)
            recs = self._ng_lite.get_recommendations(query_embedding, top_k=top_k * 2)

            # Build activation map: target_id -> confidence
            activation_map: Dict[str, float] = {}
            if recs:
                for target_id, confidence, _reasoning in recs:
                    activation_map[target_id] = confidence

            # Step 4: Harvest — match activated targets to DVS entries
            # Also get novelty for scoring
            novelty = self._ng_lite.detect_novelty(query_embedding)

            # Score all entries using multi-factor ranking
            scored: List[Tuple[DVSEntry, float]] = []
            for entry in self._entries.values():
                if entry_type and entry.entry_type != entry_type.value:
                    continue

                # Factor 1: Substrate activation (learned relevance)
                activation = 0.0
                if entry.ng_node_id is not None:
                    # Check if any recommendation target maps to this entry
                    for target_id, conf in activation_map.items():
                        if str(entry.ng_node_id) in target_id or entry.entry_id in target_id:
                            activation = max(activation, conf)

                # Factor 2: Cosine similarity (geometric relevance)
                entry_emb = entry.embedding_array()
                e_norm = np.linalg.norm(entry_emb)
                q_norm = np.linalg.norm(query_embedding)
                if e_norm > 0 and q_norm > 0:
                    cosine_sim = float(np.dot(
                        query_embedding / q_norm,
                        entry_emb / e_norm,
                    ))
                else:
                    cosine_sim = 0.0

                # Factor 3: Recency (newer is better)
                age = time.time() - entry.timestamp
                recency_days = getattr(self._config, 'dvs_recency_days', 30) if self._config else 30
                recency = max(0.0, 1.0 - (age / (86400 * recency_days)))

                # Factor 4: Repair success rate
                s_bonus = getattr(self._config, 'dvs_success_bonus', 0.2) if self._config else 0.2
                p_bonus = getattr(self._config, 'dvs_partial_bonus', 0.1) if self._config else 0.1
                success_bonus = 0.0
                if entry.repair_outcome == "success":
                    success_bonus = s_bonus
                elif entry.repair_outcome == "partial":
                    success_bonus = p_bonus

                # Multi-factor score (substrate activation weighted highest)
                w_a = getattr(self._config, 'dvs_weight_activation', 0.4) if self._config else 0.4
                w_c = getattr(self._config, 'dvs_weight_cosine', 0.3) if self._config else 0.3
                w_r = getattr(self._config, 'dvs_weight_recency', 0.15) if self._config else 0.15
                w_s = getattr(self._config, 'dvs_weight_success', 0.15) if self._config else 0.15
                score = (
                    activation * w_a +
                    cosine_sim * w_c +
                    recency * w_r +
                    success_bonus * w_s
                )

                scored.append((entry, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            results = scored[:top_k]

            for entry, _ in results:
                entry.ttl_accesses += 1

            return results

        except Exception as exc:
            logger.warning("Substrate search failed, falling back to cosine: %s", exc)
            return self._cosine_search(query_embedding, top_k, entry_type)

    # -----------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Save DVS to msgpack file.

        Uses atomic write (temp file + rename) per PRD §6.2.
        """
        path = path or self._persistence_path
        if not path:
            return

        try:
            import msgpack

            data = [entry.to_dict() for entry in self._entries.values()]
            serialized = msgpack.packb(data, use_bin_type=True)

            # Atomic write
            tmp_path = path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(serialized)
            os.replace(tmp_path, path)

            logger.debug("DVS saved: %d entries to %s", len(data), path)

        except Exception as exc:
            logger.warning("DVS save failed: %s", exc)

    def _load(self, path: str) -> None:
        """Load DVS from msgpack file."""
        try:
            import msgpack

            with open(path, "rb") as f:
                data = msgpack.unpackb(f.read(), raw=False)

            for entry_dict in data:
                entry = DVSEntry.from_dict(entry_dict)
                self._entries[entry.entry_id] = entry

            logger.info("DVS loaded: %d entries from %s", len(self._entries), path)

        except Exception as exc:
            logger.warning("DVS load failed (starting empty): %s", exc)
            self._entries = {}

    def export_json(self, path: Optional[str] = None) -> None:
        """Export DVS to JSON for debugging.  Never loaded at startup."""
        if path is None and self._persistence_path:
            base = os.path.dirname(self._persistence_path)
            path = os.path.join(base, "dvs_debug.json")

        if not path:
            return

        try:
            data = [entry.to_dict() for entry in self._entries.values()]
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug("DVS exported to JSON: %s", path)
        except Exception as exc:
            logger.warning("DVS JSON export failed: %s", exc)

    # -----------------------------------------------------------------
    # Eviction
    # -----------------------------------------------------------------

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry based on ttl_accesses."""
        if not self._entries:
            return

        # Find entry with lowest access count (ties broken by oldest timestamp)
        victim = min(
            self._entries.values(),
            key=lambda e: (e.ttl_accesses, e.timestamp),
        )
        del self._entries[victim.entry_id]
        logger.debug("DVS evicted entry %s (accesses=%d)", victim.entry_id, victim.ttl_accesses)

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return DVS statistics."""
        type_counts: Dict[str, int] = {}
        for entry in self._entries.values():
            type_counts[entry.entry_type] = type_counts.get(entry.entry_type, 0) + 1

        return {
            "total_entries": len(self._entries),
            "max_entries": self._max_entries,
            "fullness_pct": round(len(self._entries) / self._max_entries, 4) if self._max_entries > 0 else 0,
            "type_counts": type_counts,
            "substrate_augmented": self._ng_lite is not None,
        }
