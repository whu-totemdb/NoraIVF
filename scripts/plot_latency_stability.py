#!/usr/bin/env python3
"""Figure 8: P99 stability vs index size (3 rows x 2 cols).

Rows = operations (Query P99 / Insert P99 / Delete P99, top to bottom);
columns = datasets (SIFT-1M / AudioSet-2M).  Each dataset has a single
continuous index-size range, so a dataset column has no cross-dataset
connecting segment; the per-operation y-range is shared across the two
dataset columns.

This script is the authoritative writer of
``figures/latency_stability_1M.pdf``.  ``scripts/plot_rebuttal_figures.py``
holds a superseded 4-method variant of the same figure and writes it as
``latency_stability_legacy_4method.pdf`` so the two cannot collide.

Two rows (datasets) x three columns (operations), five methods per panel
(Frozen / DeDrift / SPFresh / NoraIVF / AdaIVF).  The
base latency files (``results/latency_scale/<name>.json``) carry Frozen,
NovaIVF and SPFresh, the ``<name>_ada.json`` companion adds AdaIVF and the
``results/latency_scale_dedrift/<dataset>_dedrift.json`` run adds DeDrift.

Type sizes come from ``scripts/paper_figstyle.py``: the design canvas width
*is* the insertion width (single column 3.34 in) and the PDF is not
tight-cropped, so every point-based size is the effective on-page size
(tick/label 7.25 pt, title 8 pt, legend 7 pt, floor 6.5 pt).  Every axes is
locked to the paper subplot aspect (1.2:1) via ``paper_figstyle.lock_aspect``.
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
from matplotlib.ticker import FuncFormatter, NullLocator

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_claim_figures import METHOD_ORDER, style_for  # noqa: E402
from paper_figstyle import (  # noqa: E402
    SINGLE_COL_IN,
    lock_aspect,
    report,
    save_native,
    scaled,
    scaled_rc,
)

_ROOT = _SCRIPT_DIR.parent

# Result JSONs still key the method as "NovaIVF" (historical name); every
# rendered legend/label must use the paper spelling "NoraIVF".
METHOD_DISPLAY = {"NovaIVF": "NoraIVF", "AdaIVF": "Ada-IVF"}

# (friendly title, latency_scale file stem)
# One column per operation and one row per dataset: the 2x3 block costs two
# panel rows instead of three, which is what keeps the canvas short.
LAT_OPS = [
    ("query_p99_ms", "Query P99", "ms", False),
    ("insert_p99_us", "Insert P99", "$\\mu$s", True),
    ("delete_p99_us", "Delete P99", "$\\mu$s", True),
]
# rows = datasets
LAT_DATASETS = [
    ("SIFT-1M", "sift-128-euclidean"),
    ("AudioSet-2M", "audioset-unbal-128"),
]

# the DeDrift latency run lives in a sibling directory and uses the full dataset
# name rather than the shorter ``latency_scale`` stem.
# Transposed 2x3 block: two panel rows plus the column titles, the
# one-row legend and the shared x label.
LAT_H_IN = 2.06

DEDRIFT_DIR = _ROOT / "results" / "latency_scale_dedrift"
DEDRIFT_STEM = {
    "sift-128-euclidean": "sift-128-euclidean",
    "audioset-unbal-128": "audioset-unbal-128",
}


def display_name(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


def _load_latency(name: str) -> dict:
    path = _ROOT / "results" / "latency_scale" / f"{name}.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_latency_methods(name: str) -> dict:
    """Base 3-method file merged with AdaIVF and DeDrift companion runs."""
    result = _load_latency(name)
    extra = _ROOT / "results" / "latency_scale" / f"{name}_ada.json"
    if extra.exists():
        with extra.open(encoding="utf-8") as fh:
            result["methods"].update(json.load(fh)["methods"])
    dedrift_stem = DEDRIFT_STEM.get(name)
    if dedrift_stem is not None:
        dedrift = DEDRIFT_DIR / f"{dedrift_stem}_dedrift.json"
        if dedrift.exists():
            with dedrift.open(encoding="utf-8") as fh:
                result["methods"].update(json.load(fh)["methods"])
    return result["methods"]


def build_latency_stability_1M(scale: float) -> plt.Figure:
    d = scaled(None, scale)
    plt.rcParams.update(scaled_rc(scale))
    # Native single-column canvas, 2 rows (datasets) x 3 columns (operations).
    # Transposing the earlier 3x2 block halves the panel rows, so the canvas is
    # LAT_H_IN tall instead of 4.55 in, and every axes keeps the paper subplot
    # aspect (1.2:1).  One latency range is shared down each column (one
    # operation) and one index-size range along each row (one dataset).
    fig, axes = plt.subplots(len(LAT_DATASETS), len(LAT_OPS),
                             figsize=(SINGLE_COL_IN, LAT_H_IN))
    lock_aspect(axes.ravel())

    data = {name: _load_latency_methods(name) for _, name in LAT_DATASETS}

    # one shared y-range per operation column, across both dataset rows
    ylims: dict[str, tuple[float, float]] = {}
    for key, _, _, logy in LAT_OPS:
        vals = []
        for _, name in LAT_DATASETS:
            for method in METHOD_ORDER:
                mr = data[name].get(method)
                if mr and "checkpoints" in mr:
                    vals.extend(np.asarray(mr["checkpoints"][key], dtype=float))
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        if logy:
            ylims[key] = (vmin / 1.6, vmax * 1.5)
        else:
            pad = 0.06 * (vmax - vmin)
            ylims[key] = (vmin - pad, vmax + pad)

    method_handles, method_labels = [], []
    for r, (_row_title, name) in enumerate(LAT_DATASETS):
        for method in METHOD_ORDER:
            mr = data[name].get(method)
            if not mr or "checkpoints" not in mr:
                continue
            cp = mr["checkpoints"]
            x = np.asarray(cp["index_size"], dtype=float) / 1e6
            color, marker, mls = style_for(method)
            for c, (key, _, _, _) in enumerate(LAT_OPS):
                axes[r, c].plot(x, cp[key], color=color, marker=marker,
                                linestyle=mls, linewidth=1.0 * scale,
                                markersize=1.8 * scale, markevery=8)
            if method not in method_labels:
                method_handles.append(
                    plt.Line2D([0], [0], color=color, marker=marker,
                               linestyle=mls, linewidth=1.1 * scale,
                               markersize=3.0 * scale))
                method_labels.append(display_name(method))

    # sparse x ticks: the index-size range differs per dataset row
    # (SIFT ~0.5-0.75M, AudioSet ~1.0-1.53M), three labelled points per panel
    xticks = ([0.5, 0.6, 0.7], [1.1, 1.3, 1.5])
    for r, (_row_title, _name) in enumerate(LAT_DATASETS):
        for c, (key, col_title, unit, logy) in enumerate(LAT_OPS):
            ax = axes[r, c]
            if r == 0:
                # operation + unit on the top edge of each column; every column
                # keeps its own y ticks because the units differ
                ax.set_title(f"{col_title} ({unit})", pad=1.5 * scale,
                             fontsize=d["title"])
            if c == 0:
                # dataset on the left edge of each row
                ax.set_ylabel(_row_title, labelpad=1.0 * scale)
            # the row shares one index-size range, so only the bottom row
            # carries the tick labels
            ax.set_xticks(xticks[r])
            if r != len(LAT_DATASETS) - 1:
                ax.tick_params(axis="x", labelbottom=False)
            if logy:
                ax.set_yscale("log")
                # three labelled decades (was the full 10^0..10^7 stack)
                if key == "insert_p99_us":
                    ax.set_yticks([1e3, 1e4, 1e5])
                else:                       # delete_p99_us
                    ax.set_yticks([1e1, 1e3, 1e5])
                ax.yaxis.set_minor_locator(NullLocator())
            else:
                # query P99 is linear; three ticks (was nine)
                ax.set_yticks([10, 20, 30])
            ax.set_ylim(*ylims[key])
            ax.grid(True, linewidth=0.28 * scale, alpha=0.3)
            ax.tick_params(axis="both", which="major", pad=1.2 * scale,
                           length=2.3 * scale)

    display_order = [display_name(m) for m in METHOD_ORDER]
    order = [method_labels.index(m) for m in display_order
             if m in method_labels]
    method_handles = [method_handles[i] for i in order]
    method_labels = [method_labels[i] for i in order]
    fig.legend(method_handles, method_labels, loc="upper center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 1.00),
               columnspacing=0.8, handlelength=1.4, handletextpad=0.35)
    # shared x label: the axis is identical for the three panels of a row
    fig.supxlabel("Index size (M)", y=0.045, fontsize=d["label"])
    # Reserve enough row separation for the bottom/top log tick labels. Their
    # superscripts otherwise collide at the shared boundary after typesetting.
    fig.tight_layout(rect=(0, 0.075, 1, 0.905), w_pad=0.35, h_pad=1.0,
                     pad=0.4)
    return fig


def verify(fig: plt.Figure):
    """Per-panel curve count + method-name check on the finished figure."""
    axes = [ax for ax in fig.axes]
    counts = [len(ax.get_lines()) for ax in axes]
    print(f"per-panel curve counts: {counts}")
    assert len(counts) == len(LAT_OPS) * len(LAT_DATASETS), counts
    assert all(c == 5 for c in counts), counts
    names = [t.get_text() for t in fig.legends[0].get_texts()]
    print(f"legend: {names}")
    assert names == ["Frozen", "DeDrift", "SPFresh", "NoraIVF", "Ada-IVF"], names
    assert "NovaIVF" not in names
    print("all 6 panels carry exactly 5 method curves; legend uses 'NoraIVF'")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=_ROOT / "figures")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    info = save_native(
        build_latency_stability_1M,
        args.out_dir / "latency_stability_1M.pdf", SINGLE_COL_IN,
        args.out_dir / "latency_stability_1M.png", png_dpi=600)
    report(info, "latency_stability_1M")
    assert abs(info["canvas_in"] - SINGLE_COL_IN) < 1e-6, info

    fig = build_latency_stability_1M(info["scale"])
    verify(fig)
    plt.close(fig)

    for name in ("latency_stability_1M.pdf", "latency_stability_1M.png"):
        path = args.out_dir / name
        print(f"Wrote {path}  ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
