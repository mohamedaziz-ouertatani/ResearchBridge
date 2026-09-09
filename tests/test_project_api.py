from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from researchbridge.api.app import create_app
from researchbridge.api.deps import get_session
from researchbridge.db.models import ResearchAssessment, ResearchInput


@pytest.fixture()
def client(session_factory):
    app = create_app()

    def _session_override():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_session] = _session_override
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def assessment_thread(session_factory):
    session = session_factory()
    research_input = ResearchInput(input_type="idea", raw_text="graph fraud detection")
    session.add(research_input)
    session.flush()
    first = ResearchAssessment(
        research_input_id=research_input.id,
        status="completed",
        created_at=datetime.now(UTC) - timedelta(minutes=1),
        recommendation="LOW PRIORITY",
    )
    latest = ResearchAssessment(
        research_input_id=research_input.id,
        status="completed",
        created_at=datetime.now(UTC),
        recommendation="HIGH PRIORITY",
    )
    session.add_all([first, latest])
    session.commit()
    yield research_input, first, latest
    session.close()


def test_project_lifecycle_and_latest_assessment_membership(client, session_factory, assessment_thread) -> None:
    research_input, _first, latest = assessment_thread
    created = client.post(
        "/api/projects", json={"name": "Graph fraud", "notes": "Track the idea.", "tags": ["Fraud", "graphs", "fraud"]}
    )
    assert created.status_code == 201
    project = created.json()
    assert project["tags"] == ["fraud", "graphs"]

    project_id = project["id"]
    attach_url = f"/api/projects/{project_id}/assessments/{research_input.id}"
    assert client.put(attach_url).status_code == 204
    assert client.put(attach_url).status_code == 204

    detail = client.get(f"/api/projects/{project_id}")
    assert detail.status_code == 200
    assert len(detail.json()["assessments"]) == 1
    assert detail.json()["assessments"][0]["id"] == str(latest.id)

    updated = client.patch(
        f"/api/projects/{project_id}", json={"name": "Updated graph fraud", "notes": "New note", "tags": ["ml"]}
    )
    assert updated.status_code == 200
    assert updated.json()["name"] == "Updated graph fraud"
    assert updated.json()["tags"] == ["ml"]

    assert client.delete(attach_url).status_code == 204
    assert client.get(f"/api/projects/{project_id}").json()["assessments"] == []


def test_deleting_project_preserves_assessment(client, session_factory, assessment_thread) -> None:
    research_input, _first, latest = assessment_thread
    project_id = client.post("/api/projects", json={"name": "Temporary"}).json()["id"]
    assert client.put(f"/api/projects/{project_id}/assessments/{research_input.id}").status_code == 204
    assert client.delete(f"/api/projects/{project_id}").status_code == 204
    assert client.get(f"/api/projects/{project_id}").status_code == 404

    session = session_factory()
    try:
        assert session.get(ResearchAssessment, latest.id) is not None
        assert session.get(ResearchInput, research_input.id) is not None
    finally:
        session.close()


def test_project_rejects_blank_name_and_unknown_input(client) -> None:
    assert client.post("/api/projects", json={"name": "   "}).status_code == 422
    project_id = client.post("/api/projects", json={"name": "Project"}).json()["id"]
    assert client.put(f"/api/projects/{project_id}/assessments/{uuid.uuid4()}").status_code == 404
