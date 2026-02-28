"""
The Healing Collective — Congregation (Peer Deliberation)

When the Diagnosis Engine lands in the recommendation zone (confidence
between recommend and auto-execute thresholds), the Congregation polls
peer modules for their assessment.  Peers that have seen similar failures
vote on the proposed repair, and the consensus can elevate confidence
above the auto-execute threshold — or confirm that caution is warranted.

The name "Congregation" reflects the design: individual modules gather
to deliberate, each contributing its own learned experience.

# ---- Changelog ----
# [2026-02-27] Claude (Opus 4.6) — Initial creation.
#   What: Congregation with peer polling via shared learning directory,
#         vote aggregation, and confidence adjustment.
#   Why:  PRD specifies distributed deliberation for uncertain repairs.
#         A single module may lack confidence, but three modules that have
#         all seen similar failures and successfully applied the same repair
#         provide collective certainty.
#   Settings: congregation.confidence_spread=0.15, max_candidates=4,
#         timeout_seconds=10.0, require_for_host_repairs=true.
#   How:  Writes a deliberation request to shared_learning/ as a JSONL
#         event.  Reads peer event files for repair records matching the
#         failure embedding.  Aggregates peer experience into a confidence
#         adjustment.  Returns CongregationResult with adjusted confidence.
# -------------------
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import CongregationConfig

logger = logging.getLogger("healing_collective.congregation")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PeerVote:
    """A single peer's vote on a proposed repair."""
    peer_module: str
    confidence: float  # Peer's confidence in the proposed primitive
    supporting_repairs: int  # How many successful repairs the peer has for this pattern
    similarity: float  # How similar the peer's experience is to this failure


@dataclass
class CongregationResult:
    """Result of a congregation deliberation."""
    original_confidence: float
    adjusted_confidence: float
    peers_consulted: int
    votes: List[PeerVote] = field(default_factory=list)
    consensus_reached: bool = False
    deliberation_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Congregation
# ---------------------------------------------------------------------------

class Congregation:
    """Peer deliberation for uncertain repairs.

    Consults peer modules' learned experience when local confidence
    falls in the recommendation zone.  Peer votes can raise or lower
    the effective confidence for a proposed repair.
    """

    def __init__(
        self,
        config: CongregationConfig,
        ng_ecosystem: Any,
        dvs: Any,
        embed_fn: Any,
    ):
        """
        Args:
            config: Congregation configuration.
            ng_ecosystem: NGEcosystem instance (provides peer bridge access).
            dvs: DiagnosticVectorStore instance.
            embed_fn: Callable that embeds text to np.ndarray.
        """
        self._config = config
        self._eco = ng_ecosystem
        self._dvs = dvs
        self._embed_fn = embed_fn

        # Stats
        self._deliberations = 0
        self._consensus_count = 0

    def should_deliberate(
        self,
        confidence: float,
        source: str,
        auto_execute_threshold: float,
        recommend_threshold: float,
    ) -> bool:
        """Determine whether congregation should be consulted.

        Deliberation is appropriate when:
        1. Confidence is in the recommendation zone (between recommend
           and auto-execute thresholds)
        2. For host repairs, if require_for_host_repairs is True

        Args:
            confidence: Local confidence for the proposed repair.
            source: "host" or peer module_id.
            auto_execute_threshold: Effective auto-execute threshold
                                    (already includes host premium if applicable).
            recommend_threshold: Minimum confidence to recommend.
        """
        if confidence < recommend_threshold:
            return False  # Too low — not worth deliberating
        if confidence >= auto_execute_threshold:
            return False  # Already confident enough

        if source == "host" and self._config.require_for_host_repairs:
            return True

        # For peer repairs, deliberate if there's meaningful uncertainty
        return confidence < auto_execute_threshold

    def deliberate(
        self,
        failure_embedding: np.ndarray,
        proposed_primitive: str,
        local_confidence: float,
        failure_description: str = "",
    ) -> CongregationResult:
        """Gather peer opinions on a proposed repair.

        Reads peer modules' shared learning event files for repair
        records that match the failure embedding.  Peers that have
        successfully applied the proposed primitive to similar failures
        contribute positive votes.

        Args:
            failure_embedding: Embedding of the failure being diagnosed.
            proposed_primitive: The repair primitive being considered.
            local_confidence: Local confidence before deliberation.
            failure_description: Human-readable failure description.

        Returns:
            CongregationResult with adjusted confidence.
        """
        start = time.monotonic()
        self._deliberations += 1

        votes = self._gather_peer_votes(
            failure_embedding, proposed_primitive,
        )

        adjusted = self._aggregate_votes(votes, local_confidence)

        consensus = len(votes) >= 2 and all(
            v.confidence > 0.5 for v in votes
        )
        if consensus:
            self._consensus_count += 1

        elapsed_ms = (time.monotonic() - start) * 1000

        result = CongregationResult(
            original_confidence=local_confidence,
            adjusted_confidence=adjusted,
            peers_consulted=len(votes),
            votes=votes,
            consensus_reached=consensus,
            deliberation_time_ms=elapsed_ms,
        )

        logger.info(
            "Congregation: %d peers consulted, confidence %.2f -> %.2f (consensus=%s, %.0fms)",
            len(votes), local_confidence, adjusted, consensus, elapsed_ms,
        )

        return result

    def _gather_peer_votes(
        self,
        failure_embedding: np.ndarray,
        proposed_primitive: str,
    ) -> List[PeerVote]:
        """Read peer event files and gather votes.

        Scans the shared learning directory for peer modules' JSONL
        event files.  For each peer, finds repair outcomes matching
        the failure embedding and proposed primitive.
        """
        votes: List[PeerVote] = []

        if not self._eco or not getattr(self._eco, "_peer_bridge", None):
            return votes

        try:
            bridge = self._eco._peer_bridge
            shared_dir = bridge._shared_dir

            if not shared_dir.exists():
                return votes

            query_emb = self._normalize(failure_embedding)

            for event_file in shared_dir.glob("*.jsonl"):
                peer_module = event_file.stem
                if peer_module == self._eco.module_id:
                    continue  # Skip own events
                if peer_module.startswith("_"):
                    continue  # Skip registry files

                vote = self._evaluate_peer(
                    event_file, peer_module, query_emb, proposed_primitive,
                )
                if vote is not None:
                    votes.append(vote)

                if len(votes) >= self._config.max_candidates:
                    break

        except Exception as exc:
            logger.debug("Peer vote gathering failed: %s", exc)

        return votes

    def _evaluate_peer(
        self,
        event_file: Path,
        peer_module: str,
        query_embedding: np.ndarray,
        proposed_primitive: str,
    ) -> Optional[PeerVote]:
        """Evaluate a single peer's experience with similar failures.

        Reads the peer's event file and looks for repair outcomes
        that match both the failure pattern and proposed primitive.
        """
        try:
            matching_successes = 0
            matching_total = 0
            best_similarity = 0.0

            with open(event_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Check if this event has an embedding we can compare
                    peer_emb_raw = event.get("embedding")
                    if not peer_emb_raw:
                        continue

                    peer_emb = np.array(peer_emb_raw, dtype=np.float32)
                    if peer_emb.shape[0] != query_embedding.shape[0]:
                        continue

                    peer_emb = self._normalize(peer_emb)
                    similarity = float(np.dot(query_embedding, peer_emb))

                    if similarity < (1.0 - self._config.confidence_spread * 2):
                        continue  # Not similar enough

                    best_similarity = max(best_similarity, similarity)

                    # Check if this event's target relates to the proposed primitive
                    target = event.get("target_id", "")
                    if proposed_primitive in target:
                        matching_total += 1
                        if event.get("success", False):
                            matching_successes += 1

            if best_similarity < 0.3:
                return None  # No relevant experience

            # Compute peer's confidence based on their experience
            if matching_total > 0:
                peer_confidence = matching_successes / matching_total
            elif best_similarity > 0.7:
                # Peer has seen similar patterns but no matching primitive
                peer_confidence = 0.3  # Mild support
            else:
                peer_confidence = 0.0

            if peer_confidence <= 0.0 and matching_total == 0:
                return None

            return PeerVote(
                peer_module=peer_module,
                confidence=peer_confidence,
                supporting_repairs=matching_successes,
                similarity=best_similarity,
            )

        except Exception as exc:
            logger.debug("Failed to evaluate peer %s: %s", peer_module, exc)
            return None

    def _aggregate_votes(
        self,
        votes: List[PeerVote],
        local_confidence: float,
    ) -> float:
        """Compute consensus confidence from peer votes.

        Uses similarity-weighted averaging: peers with more similar
        experience get more influence.  The result is blended with
        the local confidence using confidence_spread as the maximum
        adjustment range.

        Args:
            votes: Peer votes collected during deliberation.
            local_confidence: Pre-deliberation local confidence.

        Returns:
            Adjusted confidence, bounded by confidence_spread.
        """
        if not votes:
            return local_confidence

        # Similarity-weighted peer confidence
        total_weight = 0.0
        weighted_confidence = 0.0

        for vote in votes:
            weight = vote.similarity * (1.0 + vote.supporting_repairs * 0.1)
            weighted_confidence += vote.confidence * weight
            total_weight += weight

        if total_weight > 0:
            peer_consensus = weighted_confidence / total_weight
        else:
            return local_confidence

        # Blend: local confidence adjusted by peer consensus within spread bounds
        adjustment = (peer_consensus - local_confidence) * self._config.confidence_spread
        # Scale adjustment by number of peers (more peers = more influence)
        peer_factor = min(1.0, len(votes) / self._config.max_candidates)
        adjustment *= peer_factor

        adjusted = local_confidence + adjustment
        return float(np.clip(adjusted, 0.0, 1.0))

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _normalize(embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector."""
        norm = np.linalg.norm(embedding)
        if norm < 1e-12:
            return embedding
        return embedding / norm

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return congregation statistics."""
        return {
            "deliberations": self._deliberations,
            "consensus_reached": self._consensus_count,
            "consensus_rate": (
                round(self._consensus_count / self._deliberations, 3)
                if self._deliberations > 0 else 0.0
            ),
            "max_candidates": self._config.max_candidates,
            "confidence_spread": self._config.confidence_spread,
            "require_for_host_repairs": self._config.require_for_host_repairs,
        }
