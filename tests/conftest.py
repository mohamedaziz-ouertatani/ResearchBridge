from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from researchbridge.db.models import Base
from researchbridge.db.session import make_engine, make_session_factory

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://researchbridge:researchbridge@localhost:5433/researchbridge",
)


@pytest.fixture(scope="session")
def engine():
    eng = make_engine(TEST_DATABASE_URL)
    try:
        with eng.connect():
            pass
    except OperationalError:
        pytest.skip("Postgres not reachable at TEST_DATABASE_URL - run `docker compose up -d`")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _truncate_all(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE extraction_errors, extraction_runs, extracted_claims, evidence, "
                "ingestion_errors, ingestion_runs, paper_citations, "
                "paper_categories, paper_authors, authors, papers RESTART IDENTITY CASCADE"
            )
        )


@pytest.fixture()
def session_factory(engine):
    _truncate_all(engine)  # in case leftover data exists from a manual/CLI run against the same DB
    factory = make_session_factory(engine)
    yield factory
    _truncate_all(engine)
