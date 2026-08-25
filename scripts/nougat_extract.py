"""Standalone Nougat extraction script.

Runs ONLY under the isolated `.nougat-venv/` interpreter (see
scripts/setup_nougat_env.ps1) - it is deliberately NOT part of the
`researchbridge` package/import path, since `nougat`/`torch` must never be
imported into the main project's environment (see the "Amendment" section
of docs/superpowers/specs/2026-08-22-nougat-math-extraction-design.md for
why: six independent, unrelated dependency-version incompatibilities were
found attempting that).

Usage (from the isolated interpreter only):

    .nougat-venv/Scripts/python.exe scripts/nougat_extract.py <path-to.pdf>

Writes the extracted Markdown to stdout. Any failure (bad checkpoint,
corrupt PDF, model crash) raises and exits non-zero rather than printing
partial/empty output - the caller (researchbridge.benchmark.fulltext.
_extract_nougat) depends on a non-zero exit to detect failure instead of
silently treating empty stdout as a successful extraction.

Based on nougat-ocr==0.1.17's own reference CLI (predict.py, from the
facebookresearch/nougat repository at the exact commit that released
0.1.17): load checkpoint -> NougatModel.from_pretrained -> move_to_device
-> model.eval() -> build a LazyDataset over the PDF (this rasterizes pages
internally and prepares image tensors) -> model.inference(...) per batch
-> nougat.postprocessing.markdown_compatible(...) per page -> join pages.
"""

from __future__ import annotations

import logging
import re
import sys
from contextlib import redirect_stdout
from functools import partial
from pathlib import Path

import torch
from nougat import NougatModel
from nougat.postprocessing import markdown_compatible
from nougat.utils.checkpoint import get_checkpoint
from nougat.utils.dataset import LazyDataset
from nougat.utils.device import move_to_device

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger(__name__)


def extract_markdown(pdf_path: Path) -> str:
    # 0.1.0-base (350M) rather than 0.1.0-small (247M): the small model
    # garbles math and, more damagingly, repeats itself often enough that
    # nougat's own repetition detector rejected 66 pages across the 40
    # benchmark papers - whole pages of content silently replaced by a
    # placeholder. The base model is the accuracy the corpus needs.
    checkpoint = get_checkpoint(None, model_tag="0.1.0-base")
    model = NougatModel.from_pretrained(checkpoint)
    model = move_to_device(model, bf16=False, cuda=torch.cuda.is_available())
    model.eval()

    dataset = LazyDataset(
        pdf_path,
        partial(model.encoder.prepare_input, random_padding=False),
        None,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Nougat could not rasterize any pages from {pdf_path}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=LazyDataset.ignore_none_collate,
    )

    pages: list[str] = []
    # 0-indexed, incremented once per page in the same order LazyDataset
    # rasterized them - the same order PyMuPDF's own page iteration uses, so
    # fulltext.py's separate image-extraction pass can match a figure back to
    # the marker for the page it came from.
    page_index = 0
    for sample, _is_last_page in dataloader:
        if sample is None:
            continue
        # early_stopping=False is what nougat's CLI exposes as --no-skipping.
        # With it on, a page that trips the repetition detector is discarded
        # whole; the text that was read correctly goes in the bin along with
        # the repetition. A page containing the real paper plus some repeated
        # tokens is still the paper - a placeholder is nothing at all - so
        # keep the content and let the reader see what was actually read.
        model_output = model.inference(image_tensors=sample, early_stopping=False)
        for output in model_output["predictions"]:
            marker = f"<!--PAGE:{page_index}-->\n\n"
            if not output.strip():
                # Genuinely empty: nougat produced nothing for this page, so
                # record the gap rather than inventing content.
                pages.append(f"\n\n{marker}[MISSING_PAGE_FAIL]\n\n")
            else:
                pages.append(f"\n\n{marker}" + markdown_compatible(output))
            page_index += 1

    markdown = "".join(pages).strip()
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) != 2:
        print("usage: nougat_extract.py <path-to.pdf>", file=sys.stderr)
        sys.exit(2)

    pdf_path = Path(sys.argv[1])
    if not pdf_path.exists():
        raise FileNotFoundError(f"No such PDF: {pdf_path}")

    # Nougat prints diagnostics ("INFO: likely hallucinated title...") on
    # stdout, which is this script's data channel - they were landing inside
    # the extracted papers. Give the extraction a stdout that points at
    # stderr, so only the Markdown below reaches the real one.
    with redirect_stdout(sys.stderr):
        markdown = extract_markdown(pdf_path)

    sys.stdout.write(markdown)


if __name__ == "__main__":
    main()
