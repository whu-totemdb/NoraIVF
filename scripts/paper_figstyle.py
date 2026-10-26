#!/usr/bin/env python3
"""Paper-wide figure typography for the ACM sigconf manuscript.

The manuscript is two-column ACM sigconf:

  * single column  ``\\linewidth`` = 3.34 in
  * double column  ``\\textwidth`` = 7.00 in

Figures are laid out on whatever design canvas the panel grid needs (several
of them use a 7.4 in canvas even though they are inserted single-column).
LaTeX scales the *cropped* PDF to the insertion width, so a text element drawn
at ``d`` points on a canvas that measures ``w`` inches ends up at

    effective = d * insert_width / w

points on the page.  To keep every figure at the same effective type size we
render, measure the real cropped PDF width, and scale every point-based text
size by ``scale = w / insert_width`` (the iteration is needed because growing
the fonts grows the tight bounding box).  At the fixed point

    design = target * scale,    effective = design * insert_width / w = target.

The targets are the ACM-consistent effective sizes:

    tick / axis label   7.25 pt   (allowed band 7.0-7.5)
    panel title         8.0  pt
    legend              7.0  pt
    annotation/footnote 7.0  pt
    hard floor          6.5  pt   (dense 14-panel figures may use it)

Everything point-based (line widths, marker sizes, tick lengths, paddings) is
scaled by the same factor so the rendered figure is geometrically identical to
a figure laid out natively at the insertion width; legend spacing, tight_layout
pads and font-relative knobs already scale with the font size and are left
alone.
"""

from __future__ import annotations

import re
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager


def _register_paper_fonts() -> str:
    """Use the official VLDB template's Linux Libertine O family in every plot.

    Set NORA_PAPER_FONT_DIR on a machine without a discoverable TeX Live
    installation. Register all four faces explicitly so labels, math, and bold
    or italic annotations cannot silently fall back to DejaVu.
    """
    configured = os.environ.get("NORA_PAPER_FONT_DIR")
    if configured:
        font_dir = Path(configured)
    else:
        try:
            regular = subprocess.check_output(
                ["kpsewhich", "LinLibertine_R.otf"], text=True
            ).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            regular = ""
        if not regular:
            raise RuntimeError(
                "Linux Libertine O fonts not found; set NORA_PAPER_FONT_DIR "
                "to the directory containing LinLibertine_*.otf."
            )
        font_dir = Path(regular).parent
    for filename in ("LinLibertine_R.otf", "LinLibertine_RB.otf",
                     "LinLibertine_RI.otf", "LinLibertine_RBI.otf"):
        path = font_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        font_manager.fontManager.addfont(str(path))
    return font_manager.FontProperties(
        fname=str(font_dir / "LinLibertine_R.otf")
    ).get_name()


PAPER_FONT_FAMILY = _register_paper_fonts()

# --------------------------------------------------------------------------- #
# insertion widths (measured from the manuscript)
# --------------------------------------------------------------------------- #
SINGLE_COL_IN = 3.34
DOUBLE_COL_IN = 7.00

# --------------------------------------------------------------------------- #
# target effective (typeset) sizes, in points
# --------------------------------------------------------------------------- #
TARGETS = {
    "label": 7.25,   # axis labels
    "tick": 7.25,    # tick labels
    "title": 8.0,    # panel / axes titles
    "legend": 7.0,   # legends
    "text": 7.0,     # annotations, footnotes
}
TARGET_MIN = 6.5

# Paper-wide subplot shape: every axes *drawing* area (spines box, excluding
# titles / tick labels / axis labels) is 1.2 wide for 1.0 high.  Locked with
# ``lock_aspect`` and verified by ``scripts/verify_subplot_aspects.py``.
SUBPLOT_ASPECT = 1.2


def lock_aspect(axes, aspect: float = SUBPLOT_ASPECT):
    """Force every axes in ``axes`` to the paper subplot aspect (w/h).

    ``matplotlib.axes.Axes.set_box_aspect`` takes height / width, so the
    reciprocal is passed.  Accepts a single axes or any iterable of axes.
    """
    try:
        iterable = list(axes)
    except TypeError:
        iterable = [axes]
    for ax in iterable:
        ax.set_box_aspect(1.0 / aspect)


def scaled_rc(scale: float, targets: dict | None = None) -> dict:
    """rcParams with every absolute font size = target * scale."""
    t = {**TARGETS, **(targets or {})}
    return {
        "font.family": PAPER_FONT_FAMILY,
        "font.serif": [PAPER_FONT_FAMILY],
        "mathtext.fontset": "custom",
        "mathtext.rm": PAPER_FONT_FAMILY,
        "mathtext.it": f"{PAPER_FONT_FAMILY}:italic",
        "mathtext.bf": f"{PAPER_FONT_FAMILY}:bold",
        "mathtext.bfit": f"{PAPER_FONT_FAMILY}:bold:italic",
        "mathtext.sf": PAPER_FONT_FAMILY,
        "mathtext.tt": PAPER_FONT_FAMILY,
        "mathtext.fallback": "stix",
        "font.size": t["label"] * scale,
        "axes.labelsize": t["label"] * scale,
        "axes.titlesize": t["title"] * scale,
        "legend.fontsize": t["legend"] * scale,
        "xtick.labelsize": t["tick"] * scale,
        "ytick.labelsize": t["tick"] * scale,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def scaled(targets: dict, scale: float) -> dict:
    """Effective targets multiplied by the render scale (design sizes)."""
    t = {**TARGETS, **(targets or {})}
    return {k: v * scale for k, v in t.items()}


def pdf_width_in(path: str | Path) -> float:
    """Width of a (cropped) PDF page in inches."""
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True,
                         text=True, check=True).stdout
    m = re.search(r"Page size:\s+([0-9.]+)\s+x\s+([0-9.]+)", out)
    if not m:
        raise RuntimeError(f"cannot parse Page size from {path}")
    return float(m.group(1)) / 72.0


def pdf_font_sizes(path: str | Path) -> list[float]:
    """Every font size (pt) written with a ``Tf`` operator in the PDF.

    FlateDecode content streams are inflated; object streams (rare for
    matplotlib output) are ignored.  Sizes include mathtext script sizes
    (0.7x a parent size), which callers can filter with :func:`drop_scripts`.
    """
    import zlib

    data = Path(path).read_bytes()
    sizes: list[float] = []
    for m in re.finditer(rb"stream\r?\n", data):
        start = m.end()
        end = data.find(b"endstream", start)
        if end == -1:
            continue
        # PDF inserts a single EOL before ``endstream``; the compressed data
        # itself may legitimately end in 0x0A/0x0D, so strip only ONE EOL
        # (rstrip would corrupt the trailing binary byte and break zlib).
        raw = data[start:end]
        if raw.endswith(b"\r\n"):
            raw = raw[:-2]
        elif raw.endswith((b"\n", b"\r")):
            raw = raw[:-1]
        try:
            content = zlib.decompress(raw)
        except Exception:
            # A compressed stream may contain the literal bytes ``endstream``
            # (so the slice can be truncated) or trailing padding;  the
            # decompressobj tolerates both and returns everything decoded so
            # far, which still holds the ``Tf`` operators we need.
            try:
                content = zlib.decompressobj().decompress(raw)
            except Exception:
                content = raw
        for tf in re.finditer(rb"([0-9]*\.?[0-9]+)\s+Tf", content):
            sizes.append(float(tf.group(1)))
    return sizes


def drop_scripts(sizes: list[float], tol: float = 0.01) -> list[float]:
    """Remove mathtext script (superscript/subscript) sizes.

    Matplotlib renders scripts at 0.7x the parent size, so a size is a script
    size when another rendered size sits within ``tol`` of ``size / 0.7``.
    """
    rounded = sorted(round(s, 3) for s in sizes)
    parents = set(rounded)
    return [s for s in rounded
            if not any(abs(s - p * 0.7) <= tol for p in parents)]


def save_native(build, pdf_path: str | Path, insert_w: float,
                png_path: str | Path | None = None, png_dpi: int = 240,
                targets: dict | None = None, bbox=None):
    """Render ``build(1.0)`` on a canvas whose width *is* the insertion width.

    This is the size-correct alternative to :func:`save_with_target_fonts`.
    ``build`` must lay the figure out at ``insert_w`` inches (e.g.
    ``figsize=(insert_w, H)``) and use the effective sizes from
    ``scaled(targets, 1.0)`` / ``scaled_rc(1.0, targets)``.  The page is not
    tight-cropped (``bbox_inches=None`` by default), so the PDF page width is
    exactly the design width and every point-based size is already the
    effective on-page size -- no font-compensation loop, hence no runaway.
    """
    pdf_path = Path(pdf_path)
    fig = build(1.0)
    fig.savefig(pdf_path, bbox_inches=bbox)
    if png_path is not None:
        fig.savefig(png_path, dpi=png_dpi, bbox_inches=bbox)
    plt.close(fig)

    canvas_in = pdf_width_in(pdf_path)
    design = scaled(targets, 1.0)
    effective = {k: v * insert_w / canvas_in for k, v in design.items()}
    return {
        "pdf": pdf_path.name,
        "canvas_in": canvas_in,
        "insert_in": insert_w,
        "scale": 1.0,
        "design": design,
        "effective": effective,
    }


def save_with_target_fonts(build, pdf_path: str | Path, insert_w: float,
                           png_path: str | Path | None = None,
                           png_dpi: int = 240, targets: dict | None = None,
                           tight: bool = True, max_iter: int = 12):
    """Render ``build(scale)`` until the effective type size hits the targets.

    ``build(scale)`` must (re)create the whole figure using
    ``scaled_rc(scale, targets)`` and multiplying every explicit point-based
    size by ``scale``.  Returns a report dict for the figure.
    """
    pdf_path = Path(pdf_path)
    probe = pdf_path.with_name(pdf_path.stem + "__probe.pdf")
    bbox = "tight" if tight else None
    scale = 1.0
    width = None
    for _ in range(max_iter):
        fig = build(scale)
        fig.savefig(probe, bbox_inches=bbox)
        plt.close(fig)
        width = pdf_width_in(probe)
        new_scale = width / insert_w
        if abs(new_scale - scale) <= 0.002 * new_scale:
            scale = new_scale
            break
        # damped fixed-point update for stability
        scale = 0.5 * scale + 0.5 * new_scale
    probe.unlink(missing_ok=True)

    fig = build(scale)
    fig.savefig(pdf_path, bbox_inches=bbox)
    if png_path is not None:
        fig.savefig(png_path, dpi=png_dpi, bbox_inches=bbox)
    plt.close(fig)

    canvas_in = pdf_width_in(pdf_path)
    design = scaled(targets, scale)
    effective = {k: v * insert_w / canvas_in for k, v in design.items()}
    return {
        "pdf": pdf_path.name,
        "canvas_in": canvas_in,
        "insert_in": insert_w,
        "scale": scale,
        "design": design,
        "effective": effective,
    }


def report(info: dict, label: str) -> None:
    """Print one figure's design/effective font table."""
    d, e = info["design"], info["effective"]
    print(f"[{label}] canvas={info['canvas_in']:.2f}in  "
          f"insert={info['insert_in']:.2f}in  scale={info['scale']:.3f}")
    for key in ("tick", "label", "title", "legend", "text"):
        print(f"    {key:6s} design={d[key]:5.2f}pt  "
              f"effective={e[key]:5.2f}pt")
