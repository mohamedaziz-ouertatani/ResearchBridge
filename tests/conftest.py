from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from researchbridge.db.models import Base
from researchbridge.db.session import make_engine, make_session_factory

# Deliberately a DIFFERENT database from .env's DATABASE_URL: the fixtures
# below TRUNCATE every table, so pointing this at the dev database destroys
# the working corpus (which is exactly what happened once).
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+psycopg://researchbridge:researchbridge@localhost:5433/researchbridge_test",
)


def _refuse_to_truncate_the_dev_database() -> None:
    """Abort the whole run if the test DB is also the dev DB.

    The fixtures here TRUNCATE every table. Sharing a database with .env's
    DATABASE_URL means running the suite silently destroys the ingested
    corpus, which costs a full re-ingest to rebuild.
    """
    dev_url = os.environ.get("DATABASE_URL")
    if dev_url and dev_url == TEST_DATABASE_URL:
        pytest.exit(
            "TEST_DATABASE_URL is the same database as DATABASE_URL. The test "
            "fixtures TRUNCATE every table and would destroy the dev corpus. "
            "Point TEST_DATABASE_URL at a separate database (e.g. ..._test).",
            returncode=1,
        )


@pytest.fixture(scope="session")
def engine():
    _refuse_to_truncate_the_dev_database()
    eng = make_engine(TEST_DATABASE_URL)
    try:
        with eng.connect():
            pass
    except OperationalError:
        pytest.skip("Postgres not reachable at TEST_DATABASE_URL - run `docker compose up -d`")
    with eng.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


def _truncate_all(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE TABLE embedding_runs, embeddings, gap_detection_runs, "
                "fulltext_fetch_errors, fulltext_fetch_runs, paper_fulltext, "
                "extraction_errors, extraction_runs, extracted_claims, evidence, "
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
