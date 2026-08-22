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


def test_extract_nougat_returns_the_models_markdown_output(monkeypatch) -> None:
    """Mocks the real call shape verified against the installed nougat-ocr==0.1.17
    source (nougat/model.py, nougat/dataset/rasterize.py, predict.py):

    - _rasterize_pdf_pages(pdf_bytes) turns PDF bytes into a list of PIL page
      images (wraps nougat.dataset.rasterize.rasterize_paper + PIL.Image.open,
      the same sequence predict.py's LazyDataset/ImageDataset drive).
    - model.encoder.prepare_input(page_image, random_padding=False) turns one
      page image into an input tensor.
    - model.inference(image_tensors=batch, early_stopping=True) returns a dict
      with a "predictions" list of one markdown string per page - this is the
      real NougatModel.inference() return shape (nougat/model.py).
    """
    import torch

    import researchbridge.benchmark.fulltext as ft

    class FakeEncoder:
        def prepare_input(self, page_image, random_padding=False):
            return torch.zeros(3, 4, 4)

    class FakeModel:
        def __init__(self):
            self.encoder = FakeEncoder()

        def inference(self, image_tensors, early_stopping=True):
            assert image_tensors.shape[0] == 1  # one page in this fake PDF
            return {
                "predictions": ["# Title\n\nSome text with $\\alpha_i$ math."],
                "repeats": [None],
            }

    monkeypatch.setattr(ft, "_load_nougat_model", lambda: FakeModel())
    monkeypatch.setattr(ft, "_rasterize_pdf_pages", lambda pdf_bytes: ["fake-page-image"])

    result = ft._extract_nougat(b"fake-pdf-bytes")

    assert "alpha_i" in result or "\\alpha_i" in result
    assert "Title" in result
