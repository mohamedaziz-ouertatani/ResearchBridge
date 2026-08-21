"""Sec 32's "implicit cross-paper gaps": recurring limitation/gap patterns
across a set of related papers' claims.

Deliberately narrow in what it produces: a cluster of similar limitation/
research_gap claims from 3+ distinct papers is the OBSERVATION ("N papers
state a similar limitation") - not a synthesized opportunity statement.
Sec 32's own example distinguishes the observation ("Most evaluated
systems are offline") from the potential research gap it suggests ("Real-
time deployment under production constraints"); generating that second,
more interpretive leap would mean inventing content beyond what's grounded
in the source papers, which this project has avoided everywhere else
(extraction stays extractive for exactly this reason - see semantic.py).
That final framing step is left to the human reviewer (Sec 35).

DEFAULT_SIMILARITY_THRESHOLD is calibrated against real output, not
guessed - a first attempt at 0.5 with average linkage found nothing on a
real 21-paper neighborhood (PathCAS, cs.DC), and closer inspection showed
why: average linkage let a claim at only 0.204 similarity to another
member into the same cluster, purely through a third, intermediate claim -
a real bug (see the linkage note below), not a threshold problem. After
switching to complete linkage, a sweep from 0.30 to 0.50 on that same
neighborhood showed a smooth, sensible falloff (3 clusters -> 2 -> 2 -> 1
-> 0 patterns as the threshold tightened); 0.35 was picked as the point
that surfaces genuine, tightly-coherent patterns (worst pairwise
similarity within a returned cluster was 0.365, comfortably clearing the
threshold) without yet thinning out to nothing.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass

from sklearn.cluster import AgglomerativeClustering

from researchbridge.embedding.base import Embedder

# Which claim types feed the pattern search. Sec 32 explicitly names both:
# limitations (a stated weakness) and research_gap (a stated open question) -
# both are "explicit gap" material an author put in their own abstract.
RELEVANT_CLAIM_TYPES = ("limitations", "research_gap")

DEFAULT_MIN_CLUSTER_SIZE = 3  # matches Sec 32's own example (three papers sharing a pattern)
DEFAULT_SIMILARITY_THRESHOLD = 0.35  # see calibration note above


@dataclass
class ClaimRecord:
    paper_id: uuid.UUID
    evidence_id: uuid.UUID
    text: str


@dataclass
class GapCluster:
    members: list[ClaimRecord]
    representative_text: str
    """The member whose average similarity to the rest of the cluster is
    highest (the medoid) - a real claim from a real paper, not a summary
    invented from the group."""

    @property
    def contributing_paper_count(self) -> int:
        return len({m.paper_id for m in self.members})


def find_recurring_patterns(
    claims: list[ClaimRecord],
    embedder: Embedder,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[GapCluster]:
    """Groups claims by embedding similarity (agglomerative, average-link
    cosine distance). Only clusters spanning at least min_cluster_size
    DISTINCT papers are returned - one paper repeating itself, or two
    papers making the same point, isn't yet the recurring pattern Sec 32
    is after. Sorted with the most-corroborated pattern first.
    """
    if len(claims) < min_cluster_size:
        return []

    vectors = embedder.embed_texts([c.text for c in claims])

    # complete linkage, not average: average-link merges based on a
    # cluster-wide mean distance, which lets weakly-related claims chain
    # together transitively through an intermediate one even when they
    # aren't actually similar to each other - measured directly on real
    # output (rb-gaps-detect against PathCAS's neighborhood): one claim at
    # only 0.204 similarity to another ended up in the same cluster this
    # way. Complete linkage merges on the WORST pairwise distance in the
    # cluster, so every pair within a returned cluster is guaranteed to
    # individually clear similarity_threshold, not just clear it on average.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - similarity_threshold,
        metric="cosine",
        linkage="complete",
    )
    labels = clustering.fit_predict(vectors)

    groups: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)

    clusters = []
    for indices in groups.values():
        members = [claims[i] for i in indices]
        if len({m.paper_id for m in members}) < min_cluster_size:
            continue
        representative_index = _medoid_index(indices, vectors)
        clusters.append(GapCluster(members=members, representative_text=claims[representative_index].text))

    clusters.sort(key=lambda c: c.contributing_paper_count, reverse=True)
    return clusters


def _medoid_index(indices: list[int], vectors: list[list[float]]) -> int:
    best_index, best_score = indices[0], -1.0
    for i in indices:
        others = [j for j in indices if j != i]
        avg = sum(_cosine(vectors[i], vectors[j]) for j in others) / len(others) if others else 1.0
        if avg > best_score:
            best_index, best_score = i, avg
    return best_index


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
