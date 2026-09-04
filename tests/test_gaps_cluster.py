from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field

from researchbridge.gaps.cluster import ClaimRecord, find_recurring_patterns

_WORD = re.compile(r"[a-z]+")


@dataclass
class WordOverlapEmbedder:
    """Bag-of-words cosine similarity - deterministic and legible, matching
    the fake embedder pattern used across the extraction/retrieval tests."""

    model_name: str = "word-overlap-fake"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        vocab = sorted({w for t in texts for w in _WORD.findall(t.lower())})
        vectors = []
        for t in texts:
            words = set(_WORD.findall(t.lower()))
            raw = [1.0 if w in words else 0.0 for w in vocab]
            norm = math.sqrt(sum(x * x for x in raw)) or 1.0
            vectors.append([x / norm for x in raw])
        return vectors


def _claim(paper_id: uuid.UUID | None, text: str) -> ClaimRecord:
    return ClaimRecord(paper_id=paper_id or uuid.uuid4(), evidence_id=uuid.uuid4(), text=text)


def test_below_min_claims_returns_nothing() -> None:
    claims = [_claim(None, "evaluated only offline") for _ in range(2)]
    assert find_recurring_patterns(claims, WordOverlapEmbedder(), min_cluster_size=3) == []


def test_recurring_pattern_across_distinct_papers_is_found() -> None:
    # verified pairwise cosine under WordOverlapEmbedder: all three "offline"
    # sentences sit at 0.35-0.56 (avg-link distance ~0.54, clears a 0.3
    # threshold i.e. distance_threshold=0.7); the dataset sentence sits at
    # 0.0-0.24 against all three, well outside - real, literal word overlap,
    # not assumed paraphrase understanding a bag-of-words fake doesn't have
    claims = [
        _claim(None, "the system is tested only offline in this setup"),
        _claim(None, "we test the model only offline in our setup"),
        _claim(None, "testing here happens only offline within this setup"),
        _claim(None, "the dataset used is small and manually curated"),
    ]

    clusters = find_recurring_patterns(claims, WordOverlapEmbedder(), min_cluster_size=3, similarity_threshold=0.3)

    assert len(clusters) == 1
    assert clusters[0].contributing_paper_count == 3


def test_same_paper_repeating_itself_does_not_count_as_recurring() -> None:
    same_paper = uuid.uuid4()
    claims = [
        _claim(same_paper, "the system is tested only offline in this setup"),
        _claim(same_paper, "we test the model only offline in our setup"),
        _claim(same_paper, "testing here happens only offline within this setup"),
    ]

    clusters = find_recurring_patterns(claims, WordOverlapEmbedder(), min_cluster_size=3, similarity_threshold=0.3)

    assert clusters == []  # these would cluster together (verified above) - but from only 1 distinct paper


def test_unrelated_claims_are_not_forced_into_one_cluster() -> None:
    claims = [
        _claim(None, "the system is evaluated only offline"),
        _claim(None, "training requires substantial gpu resources"),
        _claim(None, "the dataset used is small and manually curated"),
    ]

    clusters = find_recurring_patterns(claims, WordOverlapEmbedder(), min_cluster_size=3, similarity_threshold=0.5)

    assert clusters == []  # each claim is about a different topic


def test_representative_text_is_a_real_member_not_invented() -> None:
    claims = [
        _claim(None, "the system is tested only offline in this setup"),
        _claim(None, "we test the model only offline in our setup"),
        _claim(None, "testing here happens only offline within this setup"),
    ]

    clusters = find_recurring_patterns(claims, WordOverlapEmbedder(), min_cluster_size=3, similarity_threshold=0.3)

    assert clusters[0].representative_text in [c.text for c in claims]


@dataclass
class FixedVectorEmbedder:
    """Maps specific texts to hand-picked 2D unit vectors, for exact control
    over pairwise cosine similarity - bag-of-words overlap can't reliably
    hit a precise geometry like "A-B similar, B-C similar, A-C NOT similar"
    on demand."""

    vectors_by_text: dict[str, list[float]]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors_by_text[t] for t in texts]


def _unit_vector(angle_degrees: float) -> list[float]:
    radians = math.radians(angle_degrees)
    return [math.cos(radians), math.sin(radians)]


def test_weakly_related_claims_do_not_chain_through_an_intermediate() -> None:
    # Regression test for a real bug found on live output (rb-gaps-detect
    # against a real paper's neighborhood): average linkage merges based on
    # a cluster-wide MEAN distance, so it can pull in a claim that's only
    # weakly related to the group as long as it's strongly related to one
    # member. Constructed so the two linkage strategies provably diverge:
    # A-B=0.899, B-C=0.5, A-C=0.07 (verified directly below). At distance
    # threshold 0.8 (similarity 0.2), average linkage's mean distance for
    # merging C into {A,B} is (0.93+0.5)/2=0.715 <= 0.8 -> merges all three;
    # complete linkage's max distance is max(0.93,0.5)=0.93 > 0.8 -> keeps C
    # out, which is the correct behavior (A and C are barely related at all).
    a, b, c = "claim A", "claim B", "claim C"
    embedder = FixedVectorEmbedder({a: _unit_vector(0), b: _unit_vector(26), c: _unit_vector(86)})

    assert _cosine_of(embedder, a, b) > 0.85
    assert _cosine_of(embedder, b, c) > 0.45
    assert _cosine_of(embedder, a, c) < 0.1  # the pair that must NOT end up clustered together

    claims = [_claim(None, a), _claim(None, b), _claim(None, c)]
    clusters = find_recurring_patterns(claims, embedder, min_cluster_size=2, similarity_threshold=0.2)

    for cluster in clusters:
        texts_in_cluster = {m.text for m in cluster.members}
        assert not ({a, c} <= texts_in_cluster)


def _cosine_of(embedder: FixedVectorEmbedder, text_a: str, text_b: str) -> float:
    va, vb = embedder.embed_texts([text_a, text_b])
    return sum(x * y for x, y in zip(va, vb, strict=True))


def test_cluster_with_shared_vocabulary_across_every_pair_is_tier_strong_gap() -> None:
    # every pair shares real content words ("heterogeneous", "non-iid",
    # "data", "distribution", "clients") - min pairwise Jaccard 0.45, well
    # above STRONG_KEYWORD_OVERLAP_THRESHOLD (0.15)
    texts = [
        "clients face heterogeneous non-iid data distribution",
        "the heterogeneous non-iid data distribution across clients remains unsolved",
        "heterogeneous non-iid client data distribution is a persistent challenge",
    ]
    embedder = FixedVectorEmbedder(dict(zip(texts, [_unit_vector(0), _unit_vector(5), _unit_vector(10)], strict=True)))
    claims = [_claim(None, t) for t in texts]

    clusters = find_recurring_patterns(claims, embedder, min_cluster_size=3, similarity_threshold=0.3)

    assert len(clusters) == 1
    assert clusters[0].tier == "strong_gap"


def test_cluster_without_shared_vocabulary_is_tier_potential_gap() -> None:
    # regression case: real federated-learning assessment (2026-09-04) where
    # 3 papers' limitations are topically related (all "FL has problems")
    # but name different specific problems - privacy/communication/
    # architecture, security/poisoning/privacy, poisoning/right-to-be-
    # forgotten - min pairwise Jaccard 0.0, below the threshold
    texts = [
        "insufficient protection of user privacy and high communication costs",
        "security issues such as single point of failure and model poisoning",
        "existing frameworks remain vulnerable to poisoning attacks on data privacy",
    ]
    embedder = FixedVectorEmbedder(dict(zip(texts, [_unit_vector(0), _unit_vector(5), _unit_vector(10)], strict=True)))
    claims = [_claim(None, t) for t in texts]

    clusters = find_recurring_patterns(claims, embedder, min_cluster_size=3, similarity_threshold=0.3)

    assert len(clusters) == 1
    assert clusters[0].tier == "potential_gap"


def test_clusters_sorted_largest_first() -> None:
    small_pattern = [_claim(None, f"gpu memory limits scale {i}") for i in range(3)]
    large_pattern = [_claim(None, f"offline evaluation only setting run {i}") for i in range(5)]
    claims = small_pattern + large_pattern

    clusters = find_recurring_patterns(claims, WordOverlapEmbedder(), min_cluster_size=3, similarity_threshold=0.2)

    assert len(clusters) == 2
    assert clusters[0].contributing_paper_count >= clusters[1].contributing_paper_count
