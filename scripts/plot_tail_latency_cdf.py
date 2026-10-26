#!/usr/bin/env python3
"""Tail-latency CDFs from the raw mixed-workload latency dumps.

Replaces the old ``figures/exp7_tail_latency`` figure.  Data source is the
seven ``results/latency_scale/*_raw.json`` files, whose
``methods[*].mixed.raw`` holds the per-operation latency arrays
(``insert_us`` / ``delete_us`` / ``query_us``).

Layout: 2 rows x 7 columns, one column per dataset.
  * top row    -> delete-latency CDFs (the paper's core argument: real
                  deletes vs. lazy/tombstone deletes span >3 decades)
  * bottom row -> query-latency CDFs (supporting: query service stays flat)

A single 2x7 grid (rather than a 2x4 + 2x3 split) keeps every panel aligned;
at the 7.00 in double-column insertion width the 14 panels are the densest
figure in the paper, so its tick/label effective size uses the 6.5 pt floor
(panel titles 8 pt, legend 7 pt).  Type sizes are driven by
``scripts/paper_figstyle.py`` and verified by the existing tick-overlap check.

Each panel draws one CDF per method (Frozen / DeDrift / SPFresh / NoraIVF /
AdaIVF) using the shared method colours, on an independent log-x (microseconds)
axis.  The x-axis upper limit is truncated at the pooled 99.9th percentile
so the several-order-of-magnitude tail cannot flatten the visible bulk; the
99.9th-percentile truncation is described in the manuscript text/caption
rather than annotated inside the figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

# method colours/markers/linestyles shared with the other rebuttal figures
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_claim_figures import METHOD_STYLE, display_name  # noqa: E402
from paper_figstyle import (  # noqa: E402
    DOUBLE_COL_IN,
    lock_aspect,
    report,
    save_native,
    scaled,
    scaled_rc,
)

_ROOT = Path(__file__).resolve().parent.parent

METHODS = ["Frozen", "DeDrift", "SPFresh", "NovaIVF", "AdaIVF"]

# The 14-panel density requires the 6.5 pt floor for ticks/axis labels and
# footnotes; titles and the legend keep the paper targets.
TAIL_TARGETS = {"tick": 6.5, "label": 6.5, "text": 6.5}

# (friendly title, raw-file stem without .json)
DATASETS = [
    ("MNIST", "mnist_raw"),
    ("Fashion-MNIST", "fashion_raw"),
    ("SIFT-1M", "sift-128-euclidean_raw"),
    ("GIST-960", "gist_raw"),
    ("GloVe-1.2M", "glove_raw"),
    ("AudioSet-2M", "audioset-unbal-128_raw"),
    (r"DEEP-9.9M$^{*}$", "deep_raw"),
]

ROWS = [
    ("delete_us", "Delete"),
    ("query_us", "Query"),
]

# the DeDrift latency run lives in a sibling directory and uses the full dataset
# name rather than the shorter ``latency_scale`` raw stem.
DEDRIFT_DIR = _ROOT / "results" / "latency_scale_dedrift"
DEDRIFT_STEM = {
    "mnist_raw": "mnist-784-euclidean",
    "fashion_raw": "fashion-mnist-784-euclidean",
    "sift-128-euclidean_raw": "sift-128-euclidean",
    "gist_raw": "gist-960-euclidean",
    "glove_raw": "glove-100-angular",
    "audioset-unbal-128_raw": "audioset-unbal-128",
    "deep_raw": "deep-image-96-angular",
}

# DEEP-9.9M was measured over a bounded window; the measurement-window and
# truncation notes now live in the manuscript body / caption, not in the
# figure itself (the in-figure footnote was removed in the layout pass).

CAPTION = (
    "Tail-latency CDFs of the mixed streaming workload for seven datasets "
    "(top: delete, bottom: query; one CDF per method: Frozen, DeDrift, "
    "SPFresh, " + display_name("NovaIVF") + ", " + display_name("AdaIVF") + "). "
    "Each panel uses an independent log-x axis truncated "
    "at its pooled 99.9th percentile, so the CDF bulk stays legible while "
    "the multi-decade tail is clipped at the right edge. DEEP-9.9M*: "
    "latencies were measured over a 200k-operation measurable window with "
    "the mixed phase complete (per-operation metering of all 4.99M stream "
    "operations would take 70-100 h for Frozen at the 5M live-vector "
    "scale); the other six datasets use the full stream."
)

# cap the number of vertices per curve (CDFs are smooth); the tail is kept
MAX_CURVE_POINTS = 3000
TRUNC_PCT = 99.9


def _load(name: str) -> dict:
    """Load a raw latency file, merged with its DeDrift companion run."""
    path = _ROOT / "results" / "latency_scale" / f"{name}.json"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    dedrift_stem = DEDRIFT_STEM.get(name)
    if dedrift_stem is not None:
        dedrift = DEDRIFT_DIR / f"{dedrift_stem}_dedrift.json"
        if dedrift.exists():
            with dedrift.open(encoding="utf-8") as fh:
                data["methods"].update(json.load(fh)["methods"])
    return data


def _raw_array(data: dict, method: str, op: str) -> np.ndarray:
    return np.asarray(data["methods"][method]["mixed"]["raw"][op], dtype=np.float64)


def _pooled(data: dict, op: str) -> np.ndarray:
    return np.concatenate([_raw_array(data, m, op) for m in METHODS])


def _cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(values)
    ys = np.arange(1, xs.size + 1, dtype=np.float64) / xs.size
    if xs.size > MAX_CURVE_POINTS:
        # evenly spaced in probability => tail endpoints preserved
        idx = np.unique(np.linspace(0, xs.size - 1, MAX_CURVE_POINTS).astype(int))
        xs, ys = xs[idx], ys[idx]
    return xs, ys


def _xlim(pooled: np.ndarray) -> tuple[float, float]:
    """Truncated log-x range: keep the bulk, clip the 99.9% tail."""
    lo = float(np.min(pooled)) / 1.4
    hi = float(np.percentile(pooled, TRUNC_PCT)) * 1.12
    return lo, hi


def _power_ticks(xlo: float, xhi: float, max_ticks: int = 3) -> list[float]:
    """Integer powers of ten inside ``[xlo, xhi]``, capped at ``max_ticks``.

    Using only 10**k keeps every label a single compact token (no mantissa
    pairs such as ``2x10^4 3x10^4`` crowding the narrow query panels).  The
    narrowest 7-panel columns span only ~4 decades, so labelling *adjacent*
    decades would leave < 4 pt between 10^1 and 10^2; ticks are therefore
    selected with at least a 2-decade gap where possible (falling back to
    fewer labels rather than violating the gap floor).
    """
    k_lo = int(np.floor(np.log10(xlo)))
    k_hi = int(np.ceil(np.log10(xhi)))
    powers = [k for k in range(k_lo, k_hi + 1)
              if xlo <= 10.0 ** k <= xhi]
    if len(powers) <= max_ticks:
        return [10.0 ** k for k in powers]
    chosen = [powers[-1]]
    for k in reversed(powers[:-1]):
        if chosen[-1] - k >= 2:
            chosen.append(k)
        if len(chosen) == max_ticks:
            break
    chosen.reverse()
    return [10.0 ** k for k in chosen]


def _pow_label(value: float, _pos: int | None = None) -> str:
    """Render a 10**k tick as ``$10^{k}$`` (single-line, no mantissa)."""
    return rf"$10^{{{int(np.round(np.log10(value)))}}}$"


def _plot(scale: float = 1.0) -> tuple[plt.Figure, np.ndarray]:
    d = scaled(TAIL_TARGETS, scale)
    plt.rcParams.update(scaled_rc(scale, TAIL_TARGETS))

    data = {stem: _load(stem) for _, stem in DATASETS}

    # Native double-column canvas: page width == 7.00 in insertion width, so
    # the point sizes below are the effective on-page sizes (no compensation).
    fig, axes = plt.subplots(len(ROWS), len(DATASETS),
                             figsize=(DOUBLE_COL_IN, 2.42), squeeze=False)
    # every panel keeps the paper subplot aspect (1.2:1); the saved PDF is
    # measured by scripts/verify_subplot_aspects.py
    lock_aspect(axes.ravel())

    method_handles, method_labels, method_keys = [], [], []
    for c, (title, stem) in enumerate(DATASETS):
        ds = data[stem]
        for r, (op, row_label) in enumerate(ROWS):
            ax = axes[r, c]
            pooled = _pooled(ds, op)
            xlo, xhi = _xlim(pooled)
            # integer-power ticks; if the range fits <2 powers (narrow query
            # panels), snap the left edge to the next full decade so the panel
            # still carries 2 labels (e.g. 1ms and 10ms).
            ticks = _power_ticks(xlo, xhi)
            if len(ticks) < 2:
                # Narrow query panels (e.g. the small datasets) may bracket no
                # full power of ten: snap the left edge out to a full decade
                # (original behaviour, keeps the 99.9% right-edge truncation).
                xlo = 10.0 ** int(np.floor(np.log10(xlo)))
                ticks = _power_ticks(xlo, xhi)
            if len(ticks) < 2:
                # DEEP-9.9M's query latencies all sit inside one decade, so
                # even after the left snap no second label fits: extend the
                # right edge to the next full decade for legibility.
                xhi = 10.0 ** int(np.ceil(np.log10(xhi)))
                ticks = _power_ticks(xlo, xhi)
            for method in METHODS:
                xs, ys = _cdf(_raw_array(ds, method, op))
                color, marker, ls = METHOD_STYLE[method]
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.0 * scale,
                        label=display_name(method))
                if method not in method_keys:
                    method_handles.append(Line2D([0], [0], color=color,
                                                 linestyle=ls,
                                                 linewidth=1.2 * scale))
                    method_labels.append(display_name(method))
                    method_keys.append(method)
            ax.set_xscale("log")
            ax.set_xlim(xlo, xhi)
            # sparse integer-power ticks (2-3) + no minor labels
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FuncFormatter(_pow_label))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_ylim(0.0, 1.02)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.grid(True, linewidth=0.25 * scale, alpha=0.3)
            ax.tick_params(axis="both", which="major", pad=1.0 * scale,
                           length=2.2 * scale)
            # 1 pt smaller x labels: guarantee single-line, non-overlapping
            ax.tick_params(axis="x", which="major", labelsize=d["tick"])
            if r == 0:
                ax.set_title(title, pad=1.5 * scale, fontsize=d["title"])
            if c == 0:
                # distinguish the two rows by axis name: the top row is the
                # delete-latency CDF, the bottom row the query-latency CDF.
                ax.set_ylabel("Delete latency CDF" if r == 0
                              else "Query latency CDF", labelpad=1.0 * scale)
            else:
                # all panels share the same y range (0/0.5/1): drop the
                # redundant interior y labels so the 7 narrow panels get
                # enough width for their x-tick labels to stay separate.
                ax.tick_params(axis="y", labelleft=False)

    # one shared method legend
    order = [method_keys.index(m) for m in METHODS if m in method_keys]
    method_handles = [method_handles[i] for i in order]
    method_labels = [method_labels[i] for i in order]
    fig.legend(method_handles, method_labels, loc="upper center",
               ncol=len(method_labels), frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.2,
               handlelength=1.7, handletextpad=0.4)
    # No in-figure footnote any more: the measurement-window and truncation
    # notes moved to the manuscript body.  The freed bottom band is given back
    # to the panels and the canvas height is reduced accordingly.
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 0.935), w_pad=0.18, h_pad=0.4)
    fig.supxlabel("Latency ($\\mu$s)", x=0.52, y=0.02,
                  fontsize=d["label"])
    return fig, axes


def verify(fig: plt.Figure, axes: np.ndarray) -> None:
    """Overlap / curve-count verification on the finished figure."""
    counts = [[len(axes[r, c].get_lines()) for c in range(len(DATASETS))]
              for r in range(len(ROWS))]
    n_ticks = [[len(axes[r, c].get_xticks()) for c in range(len(DATASETS))]
               for r in range(len(ROWS))]
    # overlap check: no two x tick labels on a row may intersect (the whole
    # point of this fix, since the query panels used to smear together).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overlaps = 0
    for r in range(len(ROWS)):
        boxes = []
        for c in range(len(DATASETS)):
            for lbl in axes[r, c].get_xticklabels():
                if lbl.get_text():
                    boxes.append((c, lbl.get_text(), lbl.get_window_extent(renderer)))
        boxes.sort(key=lambda b: b[2].x0)
        for (_, ta, ba), (_, tb, bb) in zip(boxes, boxes[1:]):
            if ba.x1 > bb.x0 + 0.5:  # 0.5 px tolerance
                print(f"  OVERLAP row {r}: {ta!r} vs {tb!r}")
                overlaps += 1

    print("per-panel curve counts (real lines drawn):")
    for r, (_, row_label) in enumerate(ROWS):
        print(f"  {row_label:7s}: {counts[r]}")
    assert all(cnt == len(METHODS) for row in counts for cnt in row), counts
    assert len(ROWS) * len(DATASETS) == 14, (len(ROWS), len(DATASETS))
    print("per-panel x major-tick counts:")
    for r, (_, row_label) in enumerate(ROWS):
        print(f"  {row_label:7s}: {n_ticks[r]}")
    assert all(2 <= t <= 4 for row in n_ticks for t in row), n_ticks
    assert overlaps == 0, f"{overlaps} overlapping x-tick label pairs"
    print("no overlapping x-tick labels")
    print(f"all {len(ROWS) * len(DATASETS)} panels carry exactly "
          f"{len(METHODS)} method curves")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "figures")
    args = parser.parse_args()

    def build(scale: float) -> plt.Figure:
        fig, _ = _plot(scale)
        return fig

    info = save_native(
        build, args.out_dir / "tail_latency_CDF.pdf", DOUBLE_COL_IN,
        args.out_dir / "tail_latency_CDF.png", png_dpi=600, targets=TAIL_TARGETS)
    report(info, "tail_latency_CDF")

    # rebuild at the converged scale to run the verification
    fig, axes = _plot(info["scale"])
    verify(fig, axes)
    plt.close(fig)

    (args.out_dir / "tail_latency_CDF_caption.txt").write_text(
        CAPTION + "\n", encoding="utf-8")

    # every CDF must be positive (log-x), finite and normalised to 1.0
    data = {stem: _load(stem) for _, stem in DATASETS}
    for _, stem in DATASETS:
        for op, _ in ROWS:
            for method in METHODS:
                xs, ys = _cdf(_raw_array(data[stem], method, op))
                assert np.all(np.isfinite(xs)) and np.all(xs > 0.0)
                assert ys.size > 1 and ys[0] > 0.0 and ys[-1] == 1.0
                assert np.all(np.diff(ys) >= 0.0)
    print("all CDFs finite, positive, monotonic and ending at 1.0")

    for name in ("tail_latency_CDF.pdf", "tail_latency_CDF.png",
                 "tail_latency_CDF_caption.txt"):
        path = args.out_dir / name
        print(f"Wrote {path}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
