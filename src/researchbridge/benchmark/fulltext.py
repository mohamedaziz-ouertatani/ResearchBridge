"""Fetches and caches full text for the benchmark papers only.

The corpus stores abstracts (Sec 8), which is enough for embeddings but not
for annotation: Sec 25 asks for explicitly-stated limitations, remaining
gaps, and supporting passages with sections, and those live in a paper's
Discussion and Future Work - not in a 240-word abstract.

This is the narrow, justified case Sec 46 allows for PDF processing: 40
benchmark papers, fetched once and cached to disk. It is deliberately NOT
wired into the ingestion pipeline.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import requests

from researchbridge.connectors.arxiv import MIN_REQUEST_INTERVAL_SECONDS

logger = logging.getLogger(__name__)

ARXIV_PDF_URL = "https://arxiv.org/pdf/{source_id}"
REQUEST_TIMEOUT = 60


_EXTRACTOR_FILENAMES = {
    "pymupdf": "{source_id}.txt",
    "nougat": "{source_id}.nougat.md",
}


def fulltext_path(output_dir: Path, source_id: str, extractor: str = "pymupdf") -> Path:
    if extractor not in _EXTRACTOR_FILENAMES:
        raise ValueError(f"extractor must be one of {sorted(_EXTRACTOR_FILENAMES)}, got {extractor!r}")
    return output_dir / _EXTRACTOR_FILENAMES[extractor].format(source_id=source_id)


def fetch_fulltext(source_id: str, output_dir: Path, session: requests.Session | None = None) -> str:
    """Download one paper's PDF and cache its extracted text. Returns the text."""
    path = fulltext_path(output_dir, source_id)
    if path.exists():
        return path.read_text(encoding="utf-8")

    http = session or requests
    response = http.get(
        ARXIV_PDF_URL.format(source_id=source_id),
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": "ResearchBridge/0.1 (benchmark annotation; solo research project)"},
    )
    response.raise_for_status()

    text = extract_text(response.content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def extract_text(pdf_bytes: bytes, extractor: str = "pymupdf") -> str:
    """Extract readable text from a PDF using the given extractor.

    "pymupdf": fast, always available, loses math notation (see module
    docstring). "nougat": math-aware, much slower, a real ML model - see
    _extract_nougat's docstring for what its output actually looks like.
    """
    if extractor == "pymupdf":
        return _extract_pymupdf(pdf_bytes)
    if extractor == "nougat":
        return _extract_nougat(pdf_bytes)
    raise ValueError(f"extractor must be one of ['nougat', 'pymupdf'], got {extractor!r}")


def _extract_pymupdf(pdf_bytes: bytes) -> str:
    """Extract readable text from a PDF, page by page.

    Imported lazily so the rest of the benchmark tooling doesn't pay for
    PyMuPDF, which is only needed for this one step.
    """
    import pymupdf

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pages = [page.get_text() for page in doc]

    return _tidy("\n\n".join(pages))


def _extract_nougat(pdf_bytes: bytes) -> str:
    """Math-aware extraction via Nougat (nougat-ocr==0.1.17, model tag "0.1.0-small").

    Verified live against the installed package's source (nougat/model.py,
    nougat/dataset/rasterize.py, nougat/utils/checkpoint.py, and the
    reference predict.py CLI, all under .venv/Lib/site-packages/) - do not
    re-derive this from memory if the package is ever reinstalled/upgraded,
    re-check the source, since nougat-ocr has not been released since 2023
    and its pinned API has drifted from its own declared dependencies:

    - PDF bytes go in directly: nougat.dataset.rasterize.rasterize_paper
      accepts `Union[Path, bytes]` and (via pypdfium2) rasterizes each page
      to an in-memory BMP image, one page per list entry. There is no
      intermediate file. _rasterize_pdf_pages wraps that call and
      PIL.Image.open on each result, mirroring what predict.py's
      LazyDataset/ImageDataset do internally.
    - Each page image is converted to a tensor via
      model.encoder.prepare_input(image, random_padding=False), then all
      pages for one PDF are stacked into a single (num_pages, C, H, W)
      batch and passed to model.inference(image_tensors=batch,
      early_stopping=True) - exactly the sequence predict.py's main() uses.
      inference() returns a dict; output["predictions"] is a list of one
      markdown string per page, in page order.
    - Dependency conflict found and worked around here, not upstream:
      nougat-ocr declares `transformers>=4.25.1` (unbounded above) and
      `albumentations>=1.0.0` (unbounded above), so a fresh install pulls in
      whatever is newest today. transformers' v5 rewrite renamed
      PretrainedConfig -> PreTrainedConfig and dropped the old name from
      transformers.modeling_utils (though transformers.PretrainedConfig
      still works as a back-compat alias at the top level) - nougat/model.py
      imports the old name from the submodule, so `import nougat` raises
      ImportError on any transformers 5.x unless that alias is restored
      first; _patch_transformers_for_nougat does that restoration and runs
      before every `import nougat`-family statement in this module.
      Separately, albumentations 2.x tightened parameter validation
      (pydantic-backed) and rejects nougat/transforms.py's
      `alb.ImageCompression(95, ...)` positional-int call at import time
      (compression_type must now be 'jpeg'/'webp', not a quality int) - that
      one has no in-code workaround, so pyproject.toml pins
      `albumentations<2.0.0` to keep the 1.x API nougat was written against.
    - Checkpoint: get_checkpoint(model_tag="0.1.0-small") is nougat's own
      default (both nougat/utils/checkpoint.py's MODEL_TAG and predict.py's
      --model default agree). It downloads ~1.3GB of checkpoint files from
      facebookresearch/nougat's GitHub releases into the torch hub cache
      directory on first use and reuses that cache afterward; not exercised
      by this module's tests (see _load_nougat_model's docstring).
    - bf16=True, cuda=torch.cuda.is_available() mirrors predict.py's default
      (`move_to_device(model, bf16=not args.full_precision, cuda=args.batchsize > 0)`
      with args.full_precision defaulting to False).

    Real output quirks against an actual benchmark PDF have not been
    observed yet - see Task 4, which runs this against real papers and may
    add notes here about [MISSING_PAGE_*] markers or repetition artifacts
    predict.py itself watches for.
    """
    model = _load_nougat_model()
    pages = _rasterize_pdf_pages(pdf_bytes)
    if not pages:
        return ""

    import torch

    _patch_transformers_for_nougat()
    from nougat.postprocessing import markdown_compatible

    image_tensors = torch.stack(
        [model.encoder.prepare_input(page, random_padding=False) for page in pages]
    )
    output = model.inference(image_tensors=image_tensors, early_stopping=True)
    markdown = "\n\n".join(markdown_compatible(page) for page in output["predictions"])
    return _tidy(markdown)


def _patch_transformers_for_nougat() -> None:
    """Restore transformers.modeling_utils.PretrainedConfig, the name
    nougat/model.py imports at `import nougat` time. transformers' v5
    rewrite renamed the class to PreTrainedConfig and moved it to
    transformers.configuration_utils, dropping the old name from the
    modeling_utils submodule - but transformers.PretrainedConfig (top
    level) still works as a back-compat alias, so this just re-exposes it
    under the submodule path nougat expects. Idempotent and cheap (no
    torch/nougat import): safe to call before every `import nougat`-family
    statement in this module, since any of them can be the first to trigger
    nougat/__init__.py in a given process.
    """
    import transformers

    transformers.modeling_utils.PretrainedConfig = transformers.PretrainedConfig


def _rasterize_pdf_pages(pdf_bytes: bytes) -> list:
    """Turn PDF bytes into a list of PIL page images via Nougat's own
    rasterizer (nougat.dataset.rasterize.rasterize_paper, which accepts
    bytes directly - verified against its source and against pypdfium2
    accepting a bytes buffer for PdfDocument()). Kept as its own function
    so tests can monkeypatch rasterization without needing a real PDF."""
    _patch_transformers_for_nougat()
    from PIL import Image
    from nougat.dataset.rasterize import rasterize_paper

    page_buffers = rasterize_paper(pdf_bytes, outpath=None)
    return [Image.open(buf) for buf in page_buffers]


def _load_nougat_model():
    """Lazily imports and loads the Nougat model - kept as its own function
    so tests can monkeypatch model loading without a real multi-GB download.

    See _patch_transformers_for_nougat's docstring and _extract_nougat's
    docstring for why the compatibility patch below is necessary.
    """
    _patch_transformers_for_nougat()

    from nougat import NougatModel
    from nougat.utils.checkpoint import get_checkpoint
    from nougat.utils.device import move_to_device
    import torch

    checkpoint = get_checkpoint(model_tag="0.1.0-small")
    model = NougatModel.from_pretrained(checkpoint)
    model = move_to_device(model, bf16=True, cuda=torch.cuda.is_available())
    model.eval()
    return model


# Math-heavy PDFs often set equations in fonts with custom glyph-to-code
# mappings (e.g. a symbol font where code point 0x11 is drawn as an
# operator glyph). A text extractor has no way to recover the intended
# Unicode for those, so it emits raw control characters or Private Use
# Area code points instead - not a rendering bug, the correct character
# genuinely isn't recoverable this way. Stripping them turns garbled runs
# like "x(tensor)k<PUA>" into "x(tensor)k": the surrounding prose reads
# cleanly, and the lost symbol becomes a gap rather than a broken glyph.
# Recovering the equation itself would need math-aware OCR (e.g. Mathpix,
# Nougat), well beyond what plain text extraction can do.
_UNRENDERABLE_RANGES = [
    (0x00, 0x08),  # C0 controls, excluding \t (0x09) \n (0x0A)
    (0x0B, 0x0C),
    (0x0E, 0x1F),
    (0x7F, 0x9F),  # DEL and C1 controls
    (0xE000, 0xF8FF),  # Private Use Area
    (0xF0000, 0xFFFFD),  # Supplementary Private Use Area-A
    (0x100000, 0x10FFFD),  # Supplementary Private Use Area-B
    (0xFFFD, 0xFFFD),  # replacement character
]
_UNRENDERABLE = re.compile("[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in _UNRENDERABLE_RANGES) + "]")


def _tidy(text: str) -> str:
    """Collapse the ragged whitespace PDF extraction leaves behind, and drop
    characters a text extractor cannot have gotten right (see _UNRENDERABLE_RANGES).

    Line structure is kept (annotators read this, and section headings
    depend on line breaks); only runs of blank lines and trailing spaces
    on each line are normalized.
    """
    text = _UNRENDERABLE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def throttle() -> None:
    """Space PDF downloads the same way the API connector spaces its requests."""
    time.sleep(MIN_REQUEST_INTERVAL_SECONDS)
