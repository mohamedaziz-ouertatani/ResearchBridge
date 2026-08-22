from __future__ import annotations

import pytest

from researchbridge.benchmark.fulltext import _tidy, fulltext_path


def test_control_characters_from_broken_math_fonts_are_stripped() -> None:
    # \x11 and \x00 are the kind of raw control bytes a symbol-font glyph
    # (e.g. a tensor-product operator) decodes to when a text extractor has
    # no Unicode mapping for it - not real content, just noise.
    garbled = "x⊗k\x11 and \x00d+p−1 remain readable"

    tidied = _tidy(garbled)

    assert "\x11" not in tidied
    assert "\x00" not in tidied
    assert "x⊗k" in tidied
    assert "d+p−1" in tidied


def test_private_use_area_glyphs_are_stripped() -> None:
    garbled = "beforemiddleafter"

    tidied = _tidy(garbled)

    assert "" not in tidied
    assert "" not in tidied
    assert "beforemiddleafter" in tidied


def test_supplementary_private_use_planes_are_stripped() -> None:
    garbled = "a\U000f0000b\U0010fffdc"

    tidied = _tidy(garbled)

    assert "a" in tidied and "b" in tidied and "c" in tidied
    assert "\U000f0000" not in tidied
    assert "\U0010fffd" not in tidied


def test_replacement_character_is_stripped() -> None:
    assert "�" not in _tidy("broken�glyph")


def test_genuine_math_unicode_survives() -> None:
    # These are real, correctly-decoded symbols - stripping must not touch them.
    text = "≤λ⋆ and Ω(d²) and ∑ and ∫ and p < .001"

    tidied = _tidy(text)

    assert tidied == text.strip()


def test_tab_and_newline_survive_as_before() -> None:
    text = "line one\nline two\tindented"

    assert _tidy(text) == text


def test_whitespace_collapsing_still_works_alongside_stripping() -> None:
    garbled = "para one  \n\n\n\npara two\x11 continues"

    tidied = _tidy(garbled)

    assert "\n\n\n" not in tidied
    assert "\x11" not in tidied


def test_fulltext_path_defaults_to_pymupdf_txt_filename(tmp_path) -> None:
    assert fulltext_path(tmp_path, "1234.5678") == tmp_path / "1234.5678.txt"


def test_fulltext_path_pymupdf_extractor_matches_default(tmp_path) -> None:
    assert fulltext_path(tmp_path, "1234.5678", extractor="pymupdf") == tmp_path / "1234.5678.txt"


def test_fulltext_path_nougat_extractor_uses_markdown_filename(tmp_path) -> None:
    assert fulltext_path(tmp_path, "1234.5678", extractor="nougat") == tmp_path / "1234.5678.nougat.md"


def test_fulltext_path_rejects_unknown_extractor(tmp_path) -> None:
    with pytest.raises(ValueError, match="extractor"):
        fulltext_path(tmp_path, "1234.5678", extractor="bogus")


def test_extract_text_dispatches_to_pymupdf_by_default(monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    monkeypatch.setattr(ft, "_extract_pymupdf", lambda pdf_bytes: "pymupdf output")

    assert ft.extract_text(b"fake-pdf-bytes") == "pymupdf output"


def test_extract_text_dispatches_to_nougat_when_selected(monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    monkeypatch.setattr(ft, "_extract_nougat", lambda pdf_bytes: "nougat output")

    assert ft.extract_text(b"fake-pdf-bytes", extractor="nougat") == "nougat output"


def test_extract_text_rejects_unknown_extractor() -> None:
    import pytest

    from researchbridge.benchmark.fulltext import extract_text

    with pytest.raises(ValueError, match="extractor"):
        extract_text(b"fake-pdf-bytes", extractor="bogus")


def test_extract_nougat_invokes_the_isolated_subprocess(monkeypatch, tmp_path) -> None:
    import researchbridge.benchmark.fulltext as ft

    fake_venv_python = tmp_path / ".nougat-venv" / "Scripts" / "python.exe"
    fake_venv_python.parent.mkdir(parents=True)
    fake_venv_python.write_text("")
    monkeypatch.setattr(ft, "NOUGAT_VENV_PYTHON", fake_venv_python)

    captured_args = []

    class FakeCompletedProcess:
        stdout = "# Title\n\nSome text with $\\alpha_i$ math."
        returncode = 0

    def fake_run(args, **kwargs):
        captured_args.append(args)
        return FakeCompletedProcess()

    monkeypatch.setattr(ft.subprocess, "run", fake_run)

    result = ft._extract_nougat(b"fake-pdf-bytes")

    assert "alpha_i" in result
    # confirm it invoked the isolated venv's interpreter, not the main one
    assert ".nougat-venv" in captured_args[0][0]
    assert "nougat_extract.py" in captured_args[0][1]


def test_extract_nougat_raises_on_subprocess_failure(monkeypatch, tmp_path) -> None:
    import subprocess

    import researchbridge.benchmark.fulltext as ft

    fake_venv_python = tmp_path / ".nougat-venv" / "Scripts" / "python.exe"
    fake_venv_python.parent.mkdir(parents=True)
    fake_venv_python.write_text("")
    monkeypatch.setattr(ft, "NOUGAT_VENV_PYTHON", fake_venv_python)

    def fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd=args, stderr="model crashed")

    monkeypatch.setattr(ft.subprocess, "run", fake_run)

    import pytest

    with pytest.raises(subprocess.CalledProcessError):
        ft._extract_nougat(b"fake-pdf-bytes")


def test_extract_nougat_raises_when_isolated_env_missing(monkeypatch, tmp_path) -> None:
    import researchbridge.benchmark.fulltext as ft

    monkeypatch.setattr(ft, "NOUGAT_VENV_PYTHON", tmp_path / "does-not-exist" / "python.exe")

    import pytest

    with pytest.raises(RuntimeError, match="Isolated Nougat environment not found"):
        ft._extract_nougat(b"fake-pdf-bytes")


def test_fetch_fulltext_uses_extractor_specific_cache_path(tmp_path, monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    (tmp_path / "1234.5678.nougat.md").write_text("cached nougat text", encoding="utf-8")

    result = ft.fetch_fulltext("1234.5678", tmp_path, extractor="nougat")

    assert result == "cached nougat text"


def test_fetch_fulltext_force_bypasses_the_cache(tmp_path, monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    (tmp_path / "1234.5678.txt").write_text("stale cached text", encoding="utf-8")

    class FakeResponse:
        content = b"fresh-pdf-bytes"

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(ft, "extract_text", lambda pdf_bytes, extractor="pymupdf": "freshly extracted text")

    result = ft.fetch_fulltext("1234.5678", tmp_path, session=FakeSession(), force=True)

    assert result == "freshly extracted text"
    assert (tmp_path / "1234.5678.txt").read_text(encoding="utf-8") == "freshly extracted text"


def test_fetch_fulltext_passes_extractor_through_to_extract_text(tmp_path, monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    class FakeResponse:
        content = b"fresh-pdf-bytes"

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    calls = []
    monkeypatch.setattr(
        ft, "extract_text", lambda pdf_bytes, extractor="pymupdf": calls.append(extractor) or "text"
    )

    ft.fetch_fulltext("1234.5678", tmp_path, session=FakeSession(), extractor="nougat")

    assert calls == ["nougat"]
