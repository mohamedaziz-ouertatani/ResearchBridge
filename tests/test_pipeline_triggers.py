from __future__ import annotations

import psutil
import pytest

from researchbridge.api import pipeline_triggers as pt


class FakeProcess:
    def __init__(self, poll_result: int | None = None, hangs: bool = False) -> None:
        self._poll_result = poll_result
        self.terminated = False
        self.killed = False
        self._hangs = hangs  # simulate a process that ignores terminate() and needs kill()

    def poll(self) -> int | None:
        return self._poll_result

    def terminate(self) -> None:
        self.terminated = True
        if not self._hangs:
            self._poll_result = -15

    def kill(self) -> None:
        self.killed = True
        self._poll_result = -9

    def wait(self, timeout: float | None = None) -> int:
        import subprocess

        if self._poll_result is None:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
        return self._poll_result


@pytest.fixture(autouse=True)
def _clear_registry():
    pt._RUNNING.clear()
    yield
    pt._RUNNING.clear()


def test_trigger_spawns_a_subprocess_with_the_expected_command(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return FakeProcess()

    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(pt.subprocess, "Popen", fake_popen)

    log_path = pt.trigger("ingestion_arxiv", "researchbridge.ingestion.cli", ["--max-pages", "3"])

    assert len(calls) == 1
    cmd, kwargs = calls[0]
    assert cmd == [pt.sys.executable, "-u", "-m", "researchbridge.ingestion.cli", "--max-pages", "3"]
    assert kwargs["cwd"] == pt.REPO_ROOT
    assert log_path.parent == tmp_path
    assert log_path.exists()


def test_trigger_detaches_the_subprocess_from_the_parent_console(monkeypatch, tmp_path) -> None:
    # Without this, an interrupt delivered to the API server's own console
    # (uvicorn's --reload restarting on a source-file change, or a Ctrl+C
    # in whatever terminal hosts it) broadcasts to every process attached
    # to that console - including this subprocess - killing a multi-hour
    # pipeline run with a bare KeyboardInterrupt unrelated to the pipeline
    # itself. This is a regression test for a real production incident.
    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append(kwargs)
        return FakeProcess()

    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(pt.subprocess, "Popen", fake_popen)

    pt.trigger("extraction", "researchbridge.extraction.cli", [])

    kwargs = calls[0]
    if pt.sys.platform == "win32":
        assert kwargs["creationflags"] == pt.subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert kwargs["start_new_session"] is True


def test_trigger_raises_when_already_running(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(pt.subprocess, "Popen", lambda *a, **k: FakeProcess(poll_result=None))

    pt.trigger("extraction", "researchbridge.extraction.cli", [])

    with pytest.raises(pt.PipelineAlreadyRunning):
        pt.trigger("extraction", "researchbridge.extraction.cli", [])


def test_trigger_allowed_again_after_the_process_finishes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)

    first = FakeProcess(poll_result=None)
    monkeypatch.setattr(pt.subprocess, "Popen", lambda *a, **k: first)
    pt.trigger("embedding", "researchbridge.embedding.cli_embed", [])

    first._poll_result = 0  # process finished

    second = FakeProcess(poll_result=None)
    monkeypatch.setattr(pt.subprocess, "Popen", lambda *a, **k: second)
    pt.trigger("embedding", "researchbridge.embedding.cli_embed", [])  # should not raise

    assert pt._RUNNING["embedding"] is second


def test_is_running_reflects_the_live_poll_result(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)
    proc = FakeProcess(poll_result=None)
    monkeypatch.setattr(pt.subprocess, "Popen", lambda *a, **k: proc)

    assert pt.is_running("ingestion_springer") is False

    pt.trigger("ingestion_springer", "researchbridge.ingestion.cli_springer", [])
    assert pt.is_running("ingestion_springer") is True

    proc._poll_result = 0
    assert pt.is_running("ingestion_springer") is False


def test_different_pipeline_keys_do_not_conflict(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(pt.subprocess, "Popen", lambda *a, **k: FakeProcess(poll_result=None))

    pt.trigger("ingestion_arxiv", "researchbridge.ingestion.cli", [])
    pt.trigger("ingestion_springer", "researchbridge.ingestion.cli_springer", [])  # should not raise

    assert pt.is_running("ingestion_arxiv") is True
    assert pt.is_running("ingestion_springer") is True


def test_tail_log_returns_empty_string_when_no_log_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)

    assert pt.tail_log("extraction") == ""


def test_tail_log_reads_the_most_recent_matching_log(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)
    (tmp_path / "extraction-20260101T000000Z.log").write_text("older run\n")
    (tmp_path / "extraction-20260822T120000Z.log").write_text("newest run\nline two\n")
    (tmp_path / "embedding-20260822T130000Z.log").write_text("a different pipeline\n")

    assert pt.tail_log("extraction") == "newest run\nline two"


def test_tail_log_truncates_to_the_requested_number_of_lines(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)
    lines = "\n".join(f"line {i}" for i in range(10))
    (tmp_path / "embedding-20260822T120000Z.log").write_text(lines)

    assert pt.tail_log("embedding", lines=3) == "line 7\nline 8\nline 9"


def test_stop_terminates_a_running_process_and_returns_true(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)
    proc = FakeProcess(poll_result=None)
    monkeypatch.setattr(pt.subprocess, "Popen", lambda *a, **k: proc)
    pt.trigger("extraction", "researchbridge.extraction.cli", [])

    result = pt.stop("extraction")

    assert result is True
    assert proc.terminated is True
    assert proc.killed is False
    assert pt.is_running("extraction") is False


def test_stop_escalates_to_kill_when_terminate_does_not_stop_it(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)
    proc = FakeProcess(poll_result=None, hangs=True)
    monkeypatch.setattr(pt.subprocess, "Popen", lambda *a, **k: proc)
    pt.trigger("embedding", "researchbridge.embedding.cli_embed", [])

    result = pt.stop("embedding")

    assert result is True
    assert proc.terminated is True
    assert proc.killed is True
    assert pt.is_running("embedding") is False


def test_stop_returns_false_when_nothing_is_running(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)

    assert pt.stop("ingestion_arxiv") is False


def test_stop_returns_false_when_process_already_finished(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)
    proc = FakeProcess(poll_result=None)
    monkeypatch.setattr(pt.subprocess, "Popen", lambda *a, **k: proc)
    pt.trigger("ingestion_springer", "researchbridge.ingestion.cli_springer", [])
    proc._poll_result = 0  # finished on its own before stop() is called

    assert pt.stop("ingestion_springer") is False


class FakePsutilProcess:
    """psutil.Process's terminate/kill/wait shape, for the fallback_pid path -
    a separate fake from FakeProcess since psutil raises its own
    TimeoutExpired/NoSuchProcess types, not subprocess's."""

    def __init__(self, hangs: bool = False) -> None:
        self.terminated = False
        self.killed = False
        self._hangs = hangs
        self._waited_once = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        if self._hangs and not self._waited_once:
            self._waited_once = True
            raise psutil.TimeoutExpired(seconds=timeout)
        return 0


def test_stop_falls_back_to_the_db_pid_when_not_in_the_registry(monkeypatch) -> None:
    """Regression test for a real incident: the API server restarted (this
    registry resets on restart - see module docstring) while a Springer
    ingestion subprocess kept running underneath it (deliberately detached
    so it survives exactly that). The admin "stop" button then 409'd
    against a process that was very much still alive."""
    fake_proc = FakePsutilProcess()
    monkeypatch.setattr(pt.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(pt.psutil, "Process", lambda pid: fake_proc)

    result = pt.stop("ingestion_springer", fallback_pid=36808)

    assert result is True
    assert fake_proc.terminated is True
    assert fake_proc.killed is False


def test_stop_fallback_escalates_to_kill_when_terminate_does_not_stop_it(monkeypatch) -> None:
    fake_proc = FakePsutilProcess(hangs=True)
    monkeypatch.setattr(pt.psutil, "pid_exists", lambda pid: True)
    monkeypatch.setattr(pt.psutil, "Process", lambda pid: fake_proc)

    result = pt.stop("ingestion_springer", fallback_pid=36808)

    assert result is True
    assert fake_proc.terminated is True
    assert fake_proc.killed is True


def test_stop_returns_false_when_fallback_pid_is_not_alive(monkeypatch) -> None:
    monkeypatch.setattr(pt.psutil, "pid_exists", lambda pid: False)

    assert pt.stop("ingestion_springer", fallback_pid=99999) is False


def test_stop_prefers_the_in_memory_registry_over_the_fallback_pid(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)
    proc = FakeProcess(poll_result=None)
    monkeypatch.setattr(pt.subprocess, "Popen", lambda *a, **k: proc)
    pt.trigger("ingestion_springer", "researchbridge.ingestion.cli_springer", [])

    pid_exists_called = []
    monkeypatch.setattr(pt.psutil, "pid_exists", lambda pid: pid_exists_called.append(pid) or True)

    result = pt.stop("ingestion_springer", fallback_pid=36808)

    assert result is True
    assert proc.terminated is True
    assert pid_exists_called == []  # never consulted - the in-memory handle was enough


def test_trigger_allowed_again_after_stop(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pt, "LOGS_DIR", tmp_path)
    first = FakeProcess(poll_result=None)
    monkeypatch.setattr(pt.subprocess, "Popen", lambda *a, **k: first)
    pt.trigger("extraction", "researchbridge.extraction.cli", [])
    pt.stop("extraction")

    second = FakeProcess(poll_result=None)
    monkeypatch.setattr(pt.subprocess, "Popen", lambda *a, **k: second)
    pt.trigger("extraction", "researchbridge.extraction.cli", [])  # should not raise

    assert pt._RUNNING["extraction"] is second
