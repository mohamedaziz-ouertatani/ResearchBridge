"""Small on-brand charts shared by the PDF and DOCX exporters (export.py).

Drawn once with Pillow (already a dependency - see the fonts/ note below)
and returned as PNG bytes, reused as-is by both formats: reportlab's
Image flowable embeds them in the PDF, and python-docx's add_picture()
embeds them in the .docx. A vector implementation (reportlab.graphics)
was tried first, but rasterizing it for docx needs reportlab's renderPM,
which in turn needs the optional rlPyCairo/Cairo backend - not installed,
and not something to newly require just for two small charts. Pillow
avoids that: it's already pulled in transitively (uv.lock), so this adds
no new dependency.

Rendered at _SCALE (3x) the point size actually used in the document,
then placed at the smaller point size - the same "render at 2-3x, display
smaller" trick used for any raster asset that needs to look sharp next to
vector text.
"""

from __future__ import annotations

import io
from pathlib import Path

FONTS_DIR = Path(__file__).parent / "fonts"

# Same ink-gray palette as the rest of the export (export.py) - kept here
# since export.py imports it from this module rather than duplicating the
# hex values in two places.
INK = "#14181d"
INK_SOFT = "#4a545f"
INK_FAINT = "#78838f"
RULE = "#c6cbd2"

_LEVEL_RANK = {"low": 1, "medium": 2, "high": 3}
_SCALE = 3


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _font(name: str, size: int):
    from PIL import ImageFont

    return ImageFont.truetype(str(FONTS_DIR / name), size)


def _png_bytes(image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def level_gauge_png(level: str | None, *, width_pt: float = 140, height_pt: float = 10) -> bytes:
    """A 3-segment horizontal bar, filled up to the rank of `level`
    (low/medium/high) - the graphical counterpart to the existing •••/···
    dots readout, for a novelty/feasibility heading."""
    from PIL import Image, ImageDraw

    width, height = int(width_pt * _SCALE), int(height_pt * _SCALE)
    rank = _LEVEL_RANK.get(level or "", 0)
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    gap = 2 * _SCALE
    segment_width = (width - 2 * gap) / 3
    for i in range(3):
        x0 = i * (segment_width + gap)
        fill = _rgb(INK) if i < rank else _rgb(RULE)
        draw.rectangle([x0, 0, x0 + segment_width, height], fill=fill)

    return _png_bytes(image)


def evidence_bar_chart_png(
    section_counts: list[tuple[str, int]], *, width_pt: float = 460, row_height_pt: float = 16
) -> tuple[bytes, float] | None:
    """One horizontal bar per (section label, evidence-quote count), length
    proportional to count - the report's evidence distribution across
    sections. Sections with no evidence are left out rather than drawn as
    an empty row. Returns None (nothing to draw) if every count is zero.

    Returns (png_bytes, total_height_pt) - the caller needs the height to
    place the image (row count, and so total height, isn't known upfront)."""
    from PIL import Image, ImageDraw

    rows = [(label, count) for label, count in section_counts if count > 0]
    if not rows:
        return None

    row_height = row_height_pt * _SCALE
    width = int(width_pt * _SCALE)
    height = int(row_height * len(rows))
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    label_font = _font("SpaceGrotesk-Bold.ttf", int(7 * _SCALE))
    count_font = _font("SourceSerif4-Regular.ttf", int(7.5 * _SCALE))
    label_width = 160 * _SCALE
    count_width = 20 * _SCALE
    bar_area = width - label_width - count_width
    bar_height = row_height - 6 * _SCALE
    max_count = max(count for _, count in rows)
    ink_soft, ink_faint = _rgb(INK_SOFT), _rgb(INK_FAINT)

    for i, (label, count) in enumerate(rows):
        row_top = i * row_height
        mid_y = row_top + row_height / 2
        draw.text((0, mid_y), label, font=label_font, fill=ink_faint, anchor="lm")

        bar_width = max((count / max_count) * bar_area, 2 * _SCALE) if max_count else 0
        bar_top = row_top + (row_height - bar_height) / 2
        draw.rectangle([label_width, bar_top, label_width + bar_width, bar_top + bar_height], fill=ink_soft)

        draw.text((label_width + bar_area + 4 * _SCALE, mid_y), str(count), font=count_font, fill=ink_faint, anchor="lm")

    return _png_bytes(image), height / _SCALE
