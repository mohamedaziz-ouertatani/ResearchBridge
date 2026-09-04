from __future__ import annotations

import logging

from researchbridge import pipeline_logging as pl


def _reset_root_logger() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


def test_opens_its_own_log_file_when_not_panel_triggered(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pl, "LOGS_DIR", tmp_path)
    monkeypatch.delenv(pl.LOG_FILE_ENV_VAR, raising=False)
    _reset_root_logger()

    pl.configure_pipeline_logging("extraction")

    matches = list(tmp_path.glob("extraction-*.log"))
    assert len(matches) == 1

    logging.getLogger(__name__).info("hello from a direct CLI run")
    _reset_root_logger()

    assert "hello from a direct CLI run" in matches[0].read_text()


def test_does_not_open_a_second_log_file_when_panel_triggered(tmp_path, monkeypatch) -> None:
    # trigger() already redirected this process's stdout/stderr to a file
    # at the OS level and told us so via the env var - opening a second,
    # independently-timestamped FileHandler here would just be a
    # needless duplicate of content the redirect already captures.
    monkeypatch.setattr(pl, "LOGS_DIR", tmp_path)
    monkeypatch.setenv(pl.LOG_FILE_ENV_VAR, str(tmp_path / "extraction-20260101T000000Z.log"))
    _reset_root_logger()

    pl.configure_pipeline_logging("extraction")

    assert list(tmp_path.glob("extraction-*.log")) == []
    _reset_root_logger()


def test_respects_the_requested_level(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pl, "LOGS_DIR", tmp_path)
    monkeypatch.delenv(pl.LOG_FILE_ENV_VAR, raising=False)
    _reset_root_logger()

    pl.configure_pipeline_logging("citations_fetch", logging.WARNING)

    assert logging.getLogger().level == logging.WARNING
    _reset_root_logger()
