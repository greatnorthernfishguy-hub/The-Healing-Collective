"""
The Healing Collective — Tier 3 Coordinator (Cluster Learning)

Extends the Collective's learning from dyadic peer pairs (Tier 2) to
full cluster knowledge sharing (Tier 3).  While the NGPeerBridge handles
raw pattern sharing at the substrate level, the Tier3Coordinator shares
healing-specific knowledge: which repairs worked for which failures,
aggregated across all modules in the cluster.

The shared knowledge format is a JSONL file per module in the shared
learning directory, containing repair outcome records.  At startup and
periodically, the coordinator reads all peer repair records and merges
relevant ones into the local DVS — giving the Collective the benefit
of the entire cluster's repair experience.

# ---- Changelog ----
# [2026-03-23] Claude Code (Opus 4.6) — Fix _shared_dir AttributeError (#101)
# What: Guard _peer_bridge._shared_dir access with hasattr check.
# Why:  NGTractBridge (v0.3+) replaced NGPeerBridge but doesn't have
#   _shared_dir attribute. Crashed during hook init, preventing THC from
#   loading in the fan-out. Law 4: fix at the source.
# How:  Added hasattr check before accessing _shared_dir. Falls through
#   to the existing default path fallback on line 118.
# [2026-02-27] Claude (Opus 4.6) — Initial creation.
#   What: Tier3Coordinator with repair broadcast, cluster sync,
#         and cluster-wide confidence aggregation.
#   Why:  PRD specifies Tier 3 as full cluster learning.  Individual
#         modules learning in isolation converge slowly; sharing repair
#         outcomes across the cluster accelerates convergence for
#         all modules.  The et_module.json declares tier3_upgrade: true.
#   Settings: Uses peer bridge shared_dir for I/O.  Sync triggered
#         at startup, after successful repairs, and periodically.
#   How:  Writes repair outcomes to shared_learning/healing_collective_repairs.jsonl.
#         Reads peer repair files (*.repairs.jsonl) and imports matching
#         entries into local DVS.  Cluster confidence = weighted average
#         of local + all-peer success rates for a given primitive.
# -------------------
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.dvs import DiagnosticVectorStore, DVSEntry, DVSEntryType

logger = logging.getLogger("healing_collective.tier3")


# ---------------------------------------------------------------------------
# Repair broadcast record (written to shared learning directory)
# ---------------------------------------------------------------------------

@dataclass
class RepairBroadcast:
    """A repair outcome broadcast to the cluster."""
    module_id: str
    timestamp: float
    failure_description: str
    embedding: List[float]
    proposed_primitive: str
    confidence: float
    outcome: str  # "success" | "partial" | "failed"
    tracking_id: str = ""


# ---------------------------------------------------------------------------
# Sync result
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    """Result of a cluster sync operation."""
    peers_read: int
    records_imported: int
    records_skipped: int
    duration_ms: float


# ---------------------------------------------------------------------------
# Tier 3 Coordinator
# ---------------------------------------------------------------------------

class Tier3Coordinator:
    """Cluster-wide repair knowledge sharing.

    Enables the Collective to learn from all modules' repair
    experience, not just its own.
    """

    def __init__(
        self,
        module_id: str,
        ng_ecosystem: Any,
        dvs: DiagnosticVectorStore,
        embed_fn: Any,
        shared_dir: Optional[str] = None,
    ):
        """
        Args:
            module_id: This module's identifier.
            ng_ecosystem: NGEcosystem instance.
            dvs: DiagnosticVectorStore instance.
            embed_fn: Callable that embeds text to np.ndarray.
            shared_dir: Path to shared learning directory.
                        Defaults to ~/.et_modules/shared_learning/
        """
        self._module_id = module_id
        self._eco = ng_ecosystem
        self._dvs = dvs
        self._embed_fn = embed_fn

        if shared_dir:
            self._shared_dir = Path(shared_dir)
        elif (ng_ecosystem
              and getattr(ng_ecosystem, "_peer_bridge", None)
              and hasattr(ng_ecosystem._peer_bridge, "_shared_dir")):
            self._shared_dir = ng_ecosystem._peer_bridge._shared_dir
        else:
            self._shared_dir = Path.home() / ".et_modules" / "shared_learning"

        self._shared_dir.mkdir(parents=True, exist_ok=True)

        # This module's repair broadcast file
        self._broadcast_file = self._shared_dir / f"{module_id}_repairs.jsonl"

        # Track read positions in peer files
        self._peer_read_positions: Dict[str, int] = {}

        # Stats
        self._broadcasts = 0
        self._syncs = 0
        self._total_imported = 0

    # -----------------------------------------------------------------
    # Broadcast
    # -----------------------------------------------------------------

    def broadcast_repair(
        self,
        failure_description: str,
        embedding: np.ndarray,
        proposed_primitive: str,
        confidence: float,
        outcome: str,
        tracking_id: str = "",
    ) -> None:
        """Broadcast a repair outcome to the cluster.

        Writes the outcome to the shared learning directory so other
        modules can benefit from this module's experience.

        Args:
            failure_description: What failed.
            embedding: Failure embedding vector.
            proposed_primitive: Which repair was applied.
            confidence: Confidence at time of repair.
            outcome: "success", "partial", or "failed".
            tracking_id: Optional tracking ID.
        """
        record = {
            "module_id": self._module_id,
            "timestamp": time.time(),
            "failure_description": failure_description,
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding),
            "proposed_primitive": proposed_primitive,
            "confidence": confidence,
            "outcome": outcome,
            "tracking_id": tracking_id,
        }

        try:
            with open(self._broadcast_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()
            self._broadcasts += 1
            logger.debug(
                "Broadcast repair: %s -> %s (%s)",
                proposed_primitive, outcome, failure_description[:50],
            )
        except OSError as exc:
            logger.warning("Failed to broadcast repair: %s", exc)

    # -----------------------------------------------------------------
    # Sync
    # -----------------------------------------------------------------

    def sync_cluster_knowledge(self) -> SyncResult:
        """Download and merge repair knowledge from all peers.

        Reads peer modules' repair broadcast files and imports
        relevant records into the local DVS.  Records are imported
        as REPAIR_RECORD entries with the peer's module_id as source.

        Returns:
            SyncResult with import statistics.
        """
        start = time.monotonic()
        self._syncs += 1

        peers_read = 0
        records_imported = 0
        records_skipped = 0

        try:
            for repair_file in self._shared_dir.glob("*_repairs.jsonl"):
                peer_module = repair_file.stem.replace("_repairs", "")
                if peer_module == self._module_id:
                    continue  # Skip own file

                new_records = self._read_peer_repairs(repair_file, peer_module)
                peers_read += 1

                for record in new_records:
                    imported = self._import_repair_record(record)
                    if imported:
                        records_imported += 1
                    else:
                        records_skipped += 1

        except Exception as exc:
            logger.warning("Cluster sync failed: %s", exc)

        self._total_imported += records_imported
        elapsed_ms = (time.monotonic() - start) * 1000

        result = SyncResult(
            peers_read=peers_read,
            records_imported=records_imported,
            records_skipped=records_skipped,
            duration_ms=elapsed_ms,
        )

        if records_imported > 0:
            logger.info(
                "Cluster sync: %d peers, %d imported, %d skipped (%.0fms)",
                peers_read, records_imported, records_skipped, elapsed_ms,
            )

        return result

    def _read_peer_repairs(
        self,
        repair_file: Path,
        peer_module: str,
    ) -> List[Dict[str, Any]]:
        """Read new repair records from a peer's broadcast file."""
        records: List[Dict[str, Any]] = []
        last_pos = self._peer_read_positions.get(peer_module, 0)

        try:
            with open(repair_file, "r") as f:
                f.seek(last_pos)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        pass
                self._peer_read_positions[peer_module] = f.tell()
        except OSError as exc:
            logger.debug("Failed to read peer repairs from %s: %s", repair_file, exc)

        return records

    def _import_repair_record(self, record: Dict[str, Any]) -> bool:
        """Import a peer repair record into the local DVS.

        Only imports records for primitives we have registered
        and with embeddings of the correct dimensionality.
        """
        embedding_raw = record.get("embedding")
        if not embedding_raw:
            return False

        embedding = np.array(embedding_raw, dtype=np.float32)

        entry = DVSEntry.create(
            entry_type=DVSEntryType.REPAIR_RECORD,
            source_module=record.get("module_id", "unknown_peer"),
            embedding=embedding,
            content={
                "proposed_primitive": record.get("proposed_primitive"),
                "failure_description": record.get("failure_description"),
                "tracking_id": record.get("tracking_id"),
                "source": "tier3_sync",
                "original_module": record.get("module_id"),
            },
            confidence=record.get("confidence", 0.0),
            repair_outcome=record.get("outcome"),
        )

        self._dvs.add(entry)
        return True

    # -----------------------------------------------------------------
    # Cluster confidence
    # -----------------------------------------------------------------

    def get_cluster_confidence(
        self,
        embedding: np.ndarray,
        primitive: str,
        local_confidence: float,
    ) -> float:
        """Aggregate confidence from cluster knowledge.

        Searches the DVS for repair records from peer modules
        (imported via sync) that match the failure embedding and
        proposed primitive.  Returns a blended confidence that
        incorporates cluster-wide experience.

        Args:
            embedding: Failure embedding.
            primitive: Proposed repair primitive.
            local_confidence: Local module's confidence.

        Returns:
            Cluster-adjusted confidence.
        """
        # Search DVS for peer-sourced repair records
        results = self._dvs.search(
            embedding, top_k=20,
            entry_type=DVSEntryType.REPAIR_RECORD,
        )

        peer_successes = 0
        peer_total = 0

        for entry, score in results:
            if score < 0.3:
                continue

            content = entry.content or {}
            if content.get("source") != "tier3_sync":
                continue  # Only consider peer-imported records

            if content.get("proposed_primitive") != primitive:
                continue

            peer_total += 1
            if entry.repair_outcome == "success":
                peer_successes += 1

        if peer_total == 0:
            return local_confidence

        # Peer success rate
        peer_rate = peer_successes / peer_total

        # Blend: local confidence weighted 0.7, peer consensus 0.3
        blended = local_confidence * 0.7 + peer_rate * 0.3

        return float(np.clip(blended, 0.0, 1.0))

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return Tier 3 coordinator statistics."""
        return {
            "broadcasts": self._broadcasts,
            "syncs": self._syncs,
            "total_imported": self._total_imported,
            "shared_dir": str(self._shared_dir),
            "broadcast_file": str(self._broadcast_file),
        }
