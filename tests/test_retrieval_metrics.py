from __future__ import annotations

import uuid

from researchbridge.retrieval.metrics import mrr, ndcg_at_k, precision_at_k, recall_at_k


def _ids(n: int) -> list[uuid.UUID]:
    return [uuid.uuid4() for _ in range(n)]


def test_precision_at_k_counts_relevant_in_top_k() -> None:
    a, b, c, d = _ids(4)
    ranked = [a, b, c, d]

    assert precision_at_k(ranked, {a, c}, k=2) == 0.5
    assert precision_at_k(ranked, {a, c}, k=4) == 0.5


def test_precision_at_k_empty_ranking_is_zero() -> None:
    a = uuid.uuid4()
    assert precision_at_k([], {a}, k=5) == 0.0


def test_recall_at_k_finds_fraction_of_relevant_retrieved() -> None:
    a, b, c, d = _ids(4)
    ranked = [a, b, c, d]

    assert recall_at_k(ranked, {a, b, c}, k=2) == 2 / 3
    assert recall_at_k(ranked, {a, b, c}, k=4) == 1.0


def test_recall_at_k_no_relevant_docs_is_zero_not_divide_by_zero() -> None:
    a = uuid.uuid4()
    assert recall_at_k([a], set(), k=5) == 0.0


def test_ndcg_rewards_relevant_docs_ranked_higher() -> None:
    a, b = uuid.uuid4(), uuid.uuid4()

    perfect = ndcg_at_k([a, b], {a}, k=2)
    worst = ndcg_at_k([b, a], {a}, k=2)

    assert perfect == 1.0  # the one relevant doc is first -> ideal ordering
    assert 0 < worst < perfect


def test_ndcg_no_relevant_docs_is_zero() -> None:
    a = uuid.uuid4()
    assert ndcg_at_k([a], set(), k=5) == 0.0


def test_mrr_is_inverse_rank_of_first_relevant_hit() -> None:
    a, b, c = _ids(3)

    assert mrr([a, b, c], {b}) == 1 / 2
    assert mrr([a, b, c], {a}) == 1.0


def test_mrr_no_hit_is_zero() -> None:
    a, b = uuid.uuid4(), uuid.uuid4()
    assert mrr([a], {b}) == 0.0
