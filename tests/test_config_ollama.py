from __future__ import annotations

import pytest

from researchbridge.config import ollama_enabled, ollama_host, ollama_model, ollama_timeout_seconds


def test_ollama_enabled_defaults_to_true_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_ENABLED", raising=False)

    assert ollama_enabled() is True


@pytest.mark.parametrize("value,expected", [("false", False), ("FALSE", False), ("true", True), ("True", True)])
def test_ollama_enabled_reads_the_env_var_case_insensitively(monkeypatch, value, expected) -> None:
    monkeypatch.setenv("OLLAMA_ENABLED", value)

    assert ollama_enabled() is expected


def test_ollama_host_defaults_to_localhost(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    assert ollama_host() == "http://localhost:11434"


def test_ollama_timeout_defaults_to_twenty_seconds(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_TIMEOUT_SECONDS", raising=False)

    assert ollama_timeout_seconds() == 20.0


def test_ollama_model_falls_back_to_the_callers_own_default(monkeypatch) -> None:
    """The Q&A summary stage and the assessment stages historically shipped
    different fallback models behind the same OLLAMA_MODEL var. Callers
    keep their own default so consolidating these accessors changes no
    behaviour; the divergence is now visible in one place instead of four."""
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    assert ollama_model("phi3:mini") == "phi3:mini"
    assert ollama_model("qwen2.5:3b") == "qwen2.5:3b"


def test_ollama_model_prefers_the_env_var_over_the_default(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3:8b")

    assert ollama_model("phi3:mini") == "llama3:8b"


def test_every_module_shares_one_ollama_enabled_implementation() -> None:
    """Four independent copies of this predicate existed; a change to one
    silently diverged from the rest."""
    from researchbridge.assessment import application_relevance, dimensions_llm, opportunity_synthesis
    from researchbridge.qa import summarize

    for module in (application_relevance, dimensions_llm, opportunity_synthesis, summarize):
        assert module.ollama_enabled is ollama_enabled
