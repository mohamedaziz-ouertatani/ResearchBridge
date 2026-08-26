from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

from researchbridge.assessment.build import build_assessment
from researchbridge.assessment.graph import build_similarity_graph
from researchbridge.db.models import EMBEDDING_DIM, Embedding, Evidence, ExtractedClaim, Paper, ResearchInput
from researchbridge.embedding.pipeline import EMBEDDING_TYPE


def _hash_to_unit_vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    raw = [digest[i % len(digest)] - 128 for i in range(EMBEDDING_DIM)]
    norm = sum(x * x for x in raw) ** 0.5
    return [x / norm for x in raw]


@dataclass
class FakeEmbedder:
    model_name: str = "fake-embedder-v1"
    calls: list[list[str]] = field(default_factory=list)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [_hash_to_unit_vector(t) for t in texts]


def _paper(session, embedder, source_id: str, title: str) -> Paper:
    paper = Paper(
        id=uuid.uuid4(), source="fake", source_id=source_id, title=title, abstract="",
        raw_metadata={}, ingestion_metadata={},
    )
    session.add(paper)
    session.flush()
    [vector] = embedder.embed_texts([title])
    session.add(
        Embedding(paper_id=paper.id, embedding_type=EMBEDDING_TYPE, model_name=embedder.model_name, vector=vector)
    )
    return paper


def _claim(session, paper: Paper, claim_type: str, text: str, extraction_method: str = "hybrid") -> None:
    evidence = Evidence(
        paper_id=paper.id, evidence_type=claim_type, section=None, text=text,
        extraction_method=extraction_method, model_version="v1", confidence="medium",
    )
    session.add(evidence)
    session.flush()
    session.add(
        ExtractedClaim(paper_id=paper.id, claim_type=claim_type, text=text, evidence_id=evidence.id, confidence="medium")
    )


def _research_input(session, raw_text: str) -> ResearchInput:
    ri = ResearchInput(id=uuid.uuid4(), input_type="idea", raw_text=raw_text)
    session.add(ri)
    session.flush()
    return ri


def test_graph_has_input_node_plus_one_node_per_retrieved_paper(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _paper(session, embedder, "p2", "graph transformers for fraud detection variant")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder)
    graph = build_similarity_graph(session, assessment, ri, embedder)

    assert len(graph.nodes) == 1 + len(assessment.retrieved_paper_ids)
    assert graph.nodes[0].id == "input"
    assert graph.nodes[0].type == "input"
    assert graph.nodes[0].distance_to_input is None
    session.close()


def test_input_to_paper_distance_is_zero_for_exact_text_match(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    _paper(session, embedder, "p1", "the exact same text")
    ri = _research_input(session, "the exact same text")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder)
    graph = build_similarity_graph(session, assessment, ri, embedder)

    paper_node = next(n for n in graph.nodes if n.type == "paper")
    assert paper_node.distance_to_input == 0.0
    session.close()


def test_paper_to_paper_edges_are_symmetric(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _paper(session, embedder, "p2", "unrelated topic about weather prediction")
    ri = _research_input(session, "graph transformers for fraud")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder, top_k=2)
    graph = build_similarity_graph(session, assessment, ri, embedder)

    paper_ids = [n.id for n in graph.nodes if n.type == "paper"]
    assert len(paper_ids) == 2
    edge = next(e for e in graph.edges if {e.source, e.target} == set(paper_ids))
    # only one edge per unordered pair - not a duplicate reverse edge
    assert sum(1 for e in graph.edges if {e.source, e.target} == set(paper_ids)) == 1
    assert 0.0 <= edge.distance <= 2.0
    session.close()


def test_claim_counts_grouped_by_claim_type(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "method", "uses a graph transformer")
    _claim(session, paper, "method", "uses attention pooling")
    _claim(session, paper, "limitations", "evaluated offline only")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder)
    graph = build_similarity_graph(session, assessment, ri, embedder)

    paper_node = next(n for n in graph.nodes if n.type == "paper")
    assert paper_node.claim_counts == {"method": 2, "limitations": 1}
    session.close()


def test_stub_claims_excluded_from_claim_counts(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    _claim(session, paper, "method", "synthetic placeholder", extraction_method="stub")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder)
    graph = build_similarity_graph(session, assessment, ri, embedder)

    paper_node = next(n for n in graph.nodes if n.type == "paper")
    assert paper_node.claim_counts == {}
    session.close()


def test_empty_retrieval_returns_only_input_node(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    ri = _research_input(session, "a topic with nothing in the corpus")
    session.commit()

    assessment = build_assessment(session, ri.id, embedder)
    graph = build_similarity_graph(session, assessment, ri, embedder)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].type == "input"
    assert graph.edges == []
    session.close()


def test_paper_missing_embedding_for_current_model_is_skipped(session_factory) -> None:
    session = session_factory()
    embedder = FakeEmbedder()
    paper = _paper(session, embedder, "p1", "graph transformers for fraud detection")
    ri = _research_input(session, "graph transformers for fraud detection")
    session.commit()
    assessment = build_assessment(session, ri.id, embedder)

    # simulate a model change after the assessment was built: no Embedding row
    # exists for "a-different-model", so the paper should be skipped, not error
    other_embedder = FakeEmbedder(model_name="a-different-model")
    graph = build_similarity_graph(session, assessment, ri, other_embedder)

    assert len(graph.nodes) == 1
    assert graph.nodes[0].type == "input"
    assert graph.edges == []
    session.close()
