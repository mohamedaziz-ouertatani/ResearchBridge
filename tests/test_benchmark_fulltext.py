from __future__ import annotations

import pytest

from researchbridge.benchmark.fulltext import (
    _tidy,
    fulltext_path,
    normalize_nougat_markdown,
)


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

    with pytest.raises(RuntimeError, match="model crashed"):
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
    monkeypatch.setattr(ft, "extract_images", lambda pdf_bytes: [])

    ft.fetch_fulltext("1234.5678", tmp_path, session=FakeSession(), extractor="nougat")

    assert calls == ["nougat"]



# --- Nougat structural defect repair ---------------------------------------
# Nougat emits Mathpix Markdown, which parses fine on its own, but its OCR
# also produces structural damage that is not valid MMD in the first place:
# fences that never close, float wrappers missing their partner, tables that
# simply stop. Those defects wreck the whole rest of a document when rendered,
# so they are repaired once here rather than in every consumer.


def test_code_fence_that_never_closes_is_dropped() -> None:
    # An unpaired ``` turns everything after it into one code block, so the
    # prose, tables and math that follow all render as literal source.
    text = "intro\n\n```\ncode\n```\n\nmiddle\n\n```\nrest of the paper\n"

    normalized = normalize_nougat_markdown(text)

    assert normalized.count("```") == 2
    assert "rest of the paper" in normalized


def test_balanced_code_fences_are_left_alone() -> None:
    text = "intro\n\n```\ncode\n```\n\nafter\n"

    assert normalize_nougat_markdown(text) == text


def test_orphan_table_float_wrapper_is_removed() -> None:
    # \begin{table} whose \end{table} Nougat dropped. It is layout
    # scaffolding only - the table inside it parses on its own.
    text = "before\n\n" r"\begin{table}" "\n" r"\begin{tabular}{c}" "\n" r"a \\" "\n" r"\end{tabular}" "\n\nafter\n"

    normalized = normalize_nougat_markdown(text)

    assert r"\begin{table}" not in normalized
    assert r"\begin{tabular}{c}" in normalized


def test_tabular_that_never_closes_is_terminated_at_the_blank_line() -> None:
    # Left unterminated, the table swallows the prose that follows it.
    text = "before\n\n" r"\begin{tabular}{c}" "\n" r"row \\" "\n\nafter the table\n"

    normalized = normalize_nougat_markdown(text)

    assert normalized.count(r"\end{tabular}") == 1
    assert normalized.index(r"\end{tabular}") < normalized.index("after the table")


def test_tabular_trapped_in_a_code_fence_is_lifted_out() -> None:
    # Nougat's fences land in the wrong place often enough that a real table
    # ends up inside one, where it renders as literal LaTeX.
    text = (
        "```\nlisting\n"
        r"\begin{tabular}{c}" "\n" r"a \\" "\n" r"\end{tabular}" "\n"
        "more listing\n```\n"
    )

    normalized = normalize_nougat_markdown(text)

    fence_before = normalized[: normalized.index(r"\begin{tabular}")].count("```")
    assert fence_before % 2 == 0, "table should not sit inside an open fence"


def test_normalizing_already_normalized_text_changes_nothing() -> None:
    # Not driven by a failing test - this documents a property verified
    # against the real 40-paper cache. It matters because the repair now runs
    # at extraction time: re-extracting a paper, or re-running the migration
    # over an already-repaired cache, must not compound the edits.
    text = "intro\n\n```\ncode\n```\n\n" r"\begin{table}" "\n" r"\begin{tabular}{c}" "\n" r"a \\" "\n" r"\end{tabular}" "\n\ntail\n"

    once = normalize_nougat_markdown(text)

    assert normalize_nougat_markdown(once) == once


# Nougat fences algorithm pseudocode as a code block, but the steps are full
# of inline math. Nothing renders inside a code fence, so the reader gets raw
# \(\mathcal{U}_{0}\)... instead of typeset symbols. Genuine code listings
# still belong in a fence, so the two cases are told apart by math density.


def test_math_dense_fenced_block_is_unfenced_so_its_math_can_render() -> None:
    pseudocode = (
        "```\n"
        r"1:\(\mathcal{U}\leftarrow\) set of users" "\n"
        r"2:\(\mathcal{C}\leftarrow\) set of creators" "\n"
        r"3:return \(R\)" "\n"
        "```\n"
    )

    normalized = normalize_nougat_markdown(pseudocode)

    assert "```" not in normalized
    assert r"\(\mathcal{U}\leftarrow\)" in normalized, "math must survive verbatim"
    assert "set of creators" in normalized


def test_unfenced_pseudocode_keeps_one_step_per_line() -> None:
    pseudocode = (
        "```\n"
        r"1:\(a\leftarrow\) first" "\n"
        r"2:\(b\leftarrow\) second" "\n"
        r"3:\(c\leftarrow\) third" "\n"
        "```\n"
    )

    normalized = normalize_nougat_markdown(pseudocode)

    # A markdown hard break is two trailing spaces; without it the steps
    # collapse into one running paragraph.
    assert "first  \n" in normalized
    assert "second  \n" in normalized


def test_plain_code_block_without_math_stays_fenced() -> None:
    listing = "```\nimport numpy as np\nx = np.zeros(3)\nprint(x)\n```\n"

    assert normalize_nougat_markdown(listing) == listing


def test_plain_code_keeps_its_fence_even_when_the_document_also_has_pseudocode() -> None:
    # Unfencing one block must not strip the fences off unrelated listings
    # elsewhere in the same paper.
    doc = (
        "intro\n\n"
        "```\n"
        r"1:\(a\leftarrow\) first" "\n"
        r"2:\(b\leftarrow\) second" "\n"
        r"3:\(c\leftarrow\) third" "\n"
        "```\n\n"
        "and some prose\n\n"
        "```\n"
        "import numpy as np\n"
        "print(np.zeros(3))\n"
        "```\n"
    )

    normalized = normalize_nougat_markdown(doc)

    assert "import numpy as np" in normalized
    assert normalized.count("```") == 2, "the plain listing must keep both fences"
    assert r"\(a\leftarrow\)" in normalized


# A double \hline \hline is common LaTeX for a heavier top/bottom table rule
# (23 of 40 benchmark papers use it). mathpix-markdown-it's tabular parser
# does not accept it and abandons the whole table to raw text - confirmed by
# rendering the real output through the parser, not merely inspecting the
# markdown. \hline is decorative only; collapsing repeats to one changes
# nothing about the table's content or column structure.


def test_doubled_hline_is_collapsed_to_one() -> None:
    text = (
        r"\begin{tabular}{l c} \hline \hline A & B \\ \hline \hline \end{tabular}"
    )

    normalized = normalize_nougat_markdown(text)

    assert r"\hline \hline" not in normalized
    assert normalized.count(r"\hline") == 2, "one rule at the top, one at the bottom"


def test_single_hline_is_left_alone() -> None:
    text = r"\begin{tabular}{l c} \hline A & B \\ \hline \end{tabular}"

    assert normalize_nougat_markdown(text) == text


# --- Figure extraction and inline placement ---------------------------------
# Nougat's OCR transcribes text, math and tables only - it discards every
# embedded figure. Recovering them needs a second, independent PyMuPDF pass
# over the same PDF bytes; nougat_extract.py marks each page boundary in its
# Markdown output so figures can be spliced back in near where they appeared.


def _make_pdf_with_images(page_image_counts: list[int]) -> bytes:
    """Build a real in-memory PDF with the given number of tiny images per page."""
    import pymupdf

    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 4, 4))
    pix.set_rect(pix.irect, (255, 0, 0))
    png_bytes = pix.tobytes("png")

    doc = pymupdf.open()
    for count in page_image_counts:
        page = doc.new_page()
        for i in range(count):
            page.insert_image(pymupdf.Rect(i * 10, 0, i * 10 + 8, 8), stream=png_bytes)
    data = doc.tobytes()
    doc.close()
    return data


def test_extract_images_finds_images_per_page() -> None:
    from researchbridge.benchmark.fulltext import extract_images

    pdf_bytes = _make_pdf_with_images([0, 2, 1])

    pages = sorted(image.page for image in extract_images(pdf_bytes))

    assert pages == [1, 1, 2]


def test_extract_images_returns_empty_for_a_pdf_with_no_images() -> None:
    from researchbridge.benchmark.fulltext import extract_images

    assert extract_images(_make_pdf_with_images([0, 0])) == []


def test_save_images_writes_files_named_by_page_and_index(tmp_path) -> None:
    from researchbridge.benchmark.fulltext import PageImage, save_images

    images = [
        PageImage(page=0, index=0, ext="png", data=b"fake-png-bytes-a"),
        PageImage(page=2, index=0, ext="png", data=b"fake-png-bytes-b"),
    ]

    by_page = save_images(images, tmp_path / "fulltext", "1234.5678")

    assert by_page == {0: ["p0000_0.png"], 2: ["p0002_0.png"]}
    saved = tmp_path / "images" / "1234.5678"
    assert (saved / "p0000_0.png").read_bytes() == b"fake-png-bytes-a"
    assert (saved / "p0002_0.png").read_bytes() == b"fake-png-bytes-b"


def test_splice_images_replaces_marker_with_figure_markdown() -> None:
    from researchbridge.benchmark.fulltext import splice_images

    text = "before<!--PAGE:0-->\n\nsome text\n\n<!--PAGE:1-->\n\nmore text"

    spliced = splice_images(text, {1: ["p0001_0.png"]}, "1234.5678")

    assert "<!--PAGE:" not in spliced
    assert "![figure, page 2](/api/benchmark/papers/1234.5678/images/p0001_0.png)" in spliced
    assert "some text" in spliced and "more text" in spliced


def test_splice_images_drops_markers_for_pages_with_no_images() -> None:
    from researchbridge.benchmark.fulltext import splice_images

    spliced = splice_images("<!--PAGE:0-->\n\ntext only", {}, "1234.5678")

    assert "<!--PAGE:" not in spliced
    assert "![" not in spliced


def test_fetch_fulltext_extracts_and_splices_images_for_nougat(tmp_path, monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    class FakeResponse:
        content = b"fresh-pdf-bytes"

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(
        ft, "extract_text", lambda pdf_bytes, extractor="pymupdf": "page one<!--PAGE:0-->\n\npage two"
    )
    monkeypatch.setattr(
        ft, "extract_images", lambda pdf_bytes: [ft.PageImage(page=0, index=0, ext="png", data=b"img-bytes")]
    )

    result = ft.fetch_fulltext("1234.5678", tmp_path / "fulltext", session=FakeSession(), extractor="nougat")

    assert "![figure, page 1](/api/benchmark/papers/1234.5678/images/p0000_0.png)" in result
    assert (tmp_path / "images" / "1234.5678" / "p0000_0.png").read_bytes() == b"img-bytes"


def test_fetch_fulltext_skips_image_extraction_for_pymupdf(tmp_path, monkeypatch) -> None:
    import researchbridge.benchmark.fulltext as ft

    class FakeResponse:
        content = b"fresh-pdf-bytes"

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setattr(ft, "extract_text", lambda pdf_bytes, extractor="pymupdf": "plain text")
    called = []
    monkeypatch.setattr(ft, "extract_images", lambda pdf_bytes: called.append(1) or [])

    ft.fetch_fulltext("1234.5678", tmp_path, session=FakeSession(), extractor="pymupdf")

    assert called == []
