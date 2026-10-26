#!/usr/bin/env python3
"""Rebuttal multi-panel figures (old-figure full-replacement, batch 1).

Outputs three composite figures sharing the exact method styling of
``scripts/plot_claim_figures.py`` (fixed color+marker per method, AdaIVF
crimson dashed):

  * figures/frontier_all.{pdf,png}          — single-row (1x7) QPS-recall
    frontiers, one panel per dataset (mnist / fashion / sift / gist / glove /
    audioset / deep), 5-6 method curves each, replacing the per-dataset
    ``*_qps_recall`` layout.
  * figures/lire_l_sweep_1M.{pdf,png}       — single-row L-sweep: panels
    [boundary reassignments, cumulative maintenance time, NPA violation
    rate, R@100] vs L, one curve per dataset (7 datasets per panel).
  * figures/latency_stability_legacy_4method.{pdf,png}  — SUPERSEDED.
    An earlier 2x3 P99-vs-index-size layout with 4 methods (Frozen /
    NoraIVF / SPFresh / AdaIVF), same idea as the panel layout above.
    The paper's Figure 8 is instead produced by
    ``scripts/plot_latency_stability.py`` (6 panels, 5 methods, DeDrift
    included).  This layout is kept only for reference and is written
    under a distinct name: it used to be written as
    ``latency_stability_1M.pdf``, which silently replaced the paper's
    figure whenever this script ran after ``plot_latency_stability.py``.

Type sizes are unified through ``scripts/paper_figstyle.py``: the design
canvas width *is* the insertion width (single column 3.34 in, double column
7.00 in) and the PDF is not tight-cropped, so every point-based size is the
effective on-page size (tick/label 7.25 pt, title 8 pt, legend 7 pt, floor
6.5 pt).  Every axes is locked to the paper subplot aspect (1.2:1) via
``paper_figstyle.lock_aspect``.

Style/layout follow plot_claim_figures.py; method colours are imported from
there so the three figures stay consistent with the rest of the paper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from plot_claim_figures import (  # noqa: E402
    METHOD_ORDER,
    present_methods,
    style_for,
    value_at_nprobe,
)
from paper_figstyle import (  # noqa: E402
    DOUBLE_COL_IN,
    SINGLE_COL_IN,
    lock_aspect,
    report,
    save_native,
    scaled,
    scaled_rc,
)

_ROOT = _SCRIPT_DIR.parent

# --------------------------------------------------------------------------- #
# figure 1: all-dataset QPS-recall frontiers
# --------------------------------------------------------------------------- #
# (friendly title, claim-evidence file stem, dataset key)
FRONTIER_PANELS = [
    ("MNIST", "small", "mnist-784-euclidean"),
    ("Fashion-MNIST", "small", "fashion-mnist-784-euclidean"),
    ("SIFT-1M", "sift", "sift-128-euclidean"),
    ("GIST-960", "gist500k", "gist-960-euclidean"),
    ("GloVe-1.2M", "glove", "glove-100-angular"),
    ("AudioSet-2M", "audioset", "audioset-unbal-128"),
    ("DEEP-9.9M", "deep", "deep-image-96-angular"),
]
NROWS, NCOLS = 1, 7
# Single-row strip canvas height.  With the shared 7.00 in width this
# puts every panel at the paper subplot aspect (1.2:1) while keeping
# room for the titles, the one-row legend and the shared QPS label.
# Strip canvas height.  Panels are width-limited, so this is set from the
# panel height the gutter needs: 41.7 pt of panel + 24.9 pt above (one-row
# method legend, dataset titles) + 20.4 pt below (x ticks, shared x label).
FRONTIER_H_IN = 1.19
# Single-row 4-panel L-sweep strip: panels + two-line titles + the
# two-row dataset legend + the x ticks.
LIRE_H_IN = 1.42
# Panel titles here are one step below TARGETS["title"] (8.0): these are the
# narrowest panels in the paper (0.587 in), and at 8.0 pt even the abbreviated
# names dominated the panel.
LIRE_TITLE_PT = 7.0

# Result JSONs still key the method as "NovaIVF" (historical name); every
# rendered legend/annotation must use the paper spelling "NoraIVF".  Look-ups
# keep using the JSON key, only the displayed label goes through this map.
METHOD_DISPLAY = {"NovaIVF": "NoraIVF", "AdaIVF": "Ada-IVF"}


def display_name(method: str) -> str:
    return METHOD_DISPLAY.get(method, method)


FRONTIER_CAPTION = (
    "QPS-recall frontiers on seven streaming datasets (4+3 panel grid, one "
    "panel per dataset; the three bottom panels are horizontally centred). "
    "DEEP-9.9M uses the full-coverage rerun (coverage_fraction=1.0, full GT "
    "scope, nlist=4096): its QPS is markedly lower than the earlier "
    "bounded-window measurement (e.g. Frozen QPS@8=18.2, NoraIVF QPS@8=27.9), "
    "so every panel keeps an independent y-axis range instead of a shared QPS "
    "scale. Rebuild is absent on DEEP-9.9M (its full-coverage cost would "
    "extrapolate to approximately 77 days, so the run was omitted), leaving the "
    "DEEP panel with five curves; the shared legend still lists the six methods "
    "used by the other six datasets (Frozen, Rebuild, DeDrift, SPFresh, NoraIVF, "
    "Ada-IVF)."
)


def _load_claim(stem: str) -> dict:
    """Load a claim-evidence file, preferring its full-coverage variant.

    DEEP-9.9M was originally measured with a bounded query window; the
    full-coverage rerun (``claim_evidence_deep_fullcov.json``) supersedes it
    for both the frontier and the L-sweep, so any ``*_fullcov.json`` sibling
    takes precedence over the base file.  The other six datasets have no
    fullcov sibling and are unaffected.
    """
    fullcov = _ROOT / "results" / f"claim_evidence_{stem}_fullcov.json"
    path = fullcov if fullcov.exists() else (
        _ROOT / "results" / f"claim_evidence_{stem}.json")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_frontier_all(scale: float) -> plt.Figure:
    d = scaled(None, scale)
    plt.rcParams.update(scaled_rc(scale))
    # Native double-column canvas: the page width is exactly the 7.00 in
    # insertion width, so every point-based size below is the effective on-page
    # size.  All seven datasets sit in ONE row as a compact strip (1.28 in tall
    # instead of the earlier 2x4 block at 3.40 in).  Each axes carries its own
    # QPS ticks, because the datasets differ by an order of magnitude in
    # throughput, but the two axis labels are printed once for the whole strip.
    fig = plt.figure(figsize=(DOUBLE_COL_IN, FRONTIER_H_IN))
    # left 0.070 rather than 0.056: the shared "QPS" label (8.5 pt wide) plus
    # panel 1's y-tick labels (13.8 pt) plus the tick (2.6 pt) and a gap need
    # 27.4 pt, and 0.056 left only 28.2 pt with the label at x=0.021 sitting on
    # top of the "800"/"400" digits.
    # wspace 0.407 = 19.8 pt gutter / 48.7 pt panel.  The gutter holds a
    # three-digit y-tick label (13.8 pt) plus the tick length and pad (2.6 pt)
    # plus 3.0 pt of clearance; at the earlier 12.6 pt gutter those labels ran
    # 3.8 pt into the panel on their left.
    gs = fig.add_gridspec(NROWS, NCOLS,
                          left=0.070, right=0.982, top=0.710, bottom=0.2375,
                          wspace=0.407)
    axes = [fig.add_subplot(gs[0, i]) for i in range(NCOLS)]
    lock_aspect(axes)
    handles, labels = [], []
    panel_methods: dict[str, list[str]] = {}
    for ax, (title, stem, ds_key) in zip(axes, FRONTIER_PANELS):
        ds_result = _load_claim(stem)[ds_key]
        methods = present_methods(ds_result)
        panel_methods[title] = list(methods)
        for method in methods:
            frontier = ds_result["methods"][method]["frontier"]
            color, marker, ls = style_for(method)
            label = display_name(method)
            line, = ax.plot(frontier["recall"], frontier["qps"],
                            color=color, marker=marker, linestyle=ls,
                            linewidth=1.0 * scale, markersize=2.2 * scale,
                            label=label)
            if label not in labels:
                handles.append(line)
                labels.append(label)
        ax.set_title(title, pad=1.2 * scale, fontsize=d["title"])
        ax.grid(True, linewidth=0.28 * scale, alpha=0.3)
        ax.set_xlim(left=0.0, right=1.01)
        # three y ticks and five x ticks are as many as a 0.79 in panel can
        # label without the neighbouring panel's labels colliding
        ax.yaxis.set_major_locator(MaxNLocator(3))
        # explicit 0 / 0.5 / 1 with trailing zeros dropped: three short labels
        # are the most a 0.79 in panel can carry without the neighbouring
        # panel's label running into them
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(axis="both", which="major", pad=0.8 * scale,
                       length=1.8 * scale)
    # one shared pair of axis labels instead of fourteen repeated ones
    fig.text(0.5, 0.050, "Recall@100", ha="center", va="center",
             fontsize=d["label"])
    fig.text(0.021, 0.475, "QPS", ha="center", va="center", rotation=90,
             fontsize=d["label"])
    # order the shared legend by METHOD_ORDER
    display_order = [display_name(m) for m in METHOD_ORDER]
    order = [labels.index(m) for m in display_order if m in labels]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    # DEEP-9.9M has no Rebuild run: that panel shows the five remaining methods
    # while the shared legend still advertises all six methods used elsewhere.
    assert "Rebuild" not in panel_methods["DEEP-9.9M"], panel_methods["DEEP-9.9M"]
    assert len(panel_methods["DEEP-9.9M"]) == 5, panel_methods["DEEP-9.9M"]
    assert set(labels) == set(display_order), labels
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 1.0), columnspacing=0.9,
               handlelength=1.5, handletextpad=0.4)
    return fig


# --------------------------------------------------------------------------- #
# figure 2: LIRE L-sweep (2x2 panels, 7 datasets per panel)
# --------------------------------------------------------------------------- #
# (legend label, claim-evidence file stem, dataset key)
LIRE_DATASETS = [
    ("MNIST", "small", "mnist-784-euclidean"),
    ("Fashion-MNIST", "small", "fashion-mnist-784-euclidean"),
    ("SIFT-1M", "sift", "sift-128-euclidean"),
    ("GloVe-1.2M", "glove", "glove-100-angular"),
    ("GIST-960", "gist500k", "gist-960-euclidean"),
    ("AudioSet-2M", "audioset", "audioset-unbal-128"),
    ("DEEP-9.9M", "deep", "deep-image-96-angular"),
]

# panel: (title, result key, y label, log y-scale)
# Panel titles are kept to about ten characters per line: on a 0.59 in panel
# at 8 pt a longer line runs into the neighbouring panel's title.  The caption
# spells the four panels out in full, so the short forms lose no information.
# One line per panel, kept short: on a 0.587 in panel a two-line title needed
# 19 pt of height and collided with the dataset legend above it.  "Maint."
# follows the abbreviation used in Tables 5 and 10; the caption spells the four
# panels out in full.
LIRE_PANELS = [
    ("Reassignments", "n_reassigns", "Vectors", True),
    ("Maint. (s)", "maintenance_time_s", "Time (s)", True),
    ("NPA rate", "npa_violation_rate", "Rate", True),
    ("Recall@100", "recall", "Recall", False),
]

# Sparse labelled x ticks: the six L settings keep their categorical slots, but
# only four are labelled so the single-column panels breathe (L=1 and L=8 stay
# on the curves, they just do not get a tick label).
L_MAJOR_TICKS = [0, 2, 5, 10]

# high-contrast colour + marker per dataset; DEEP gets a special dashed style
LIRE_DATASET_STYLE = {
    "MNIST": ("#1f77b4", "o", "-"),
    "Fashion-MNIST": ("#ff7f0e", "s", "-"),
    "SIFT-1M": ("#2ca02c", "^", "-"),
    "GloVe-1.2M": ("#d62728", "v", "-"),
    "GIST-960": ("#9467bd", "D", "-"),
    "AudioSet-2M": ("#8c564b", "P", "-"),
    "DEEP-9.9M": ("#000000", "X", (0, (5, 1))),
}

# L is categorical: 0 cannot live on a log x-axis, so use equidistant slots.
# L=5 is the SPFresh operating point used in our main experiments, but the
# strip does not mark it: the dashed guide line and the caption sentence that
# explained it were dropped together, so no unexplained element is left (an
# earlier rotated in-panel label had already been removed for colliding with
# the axis frame).  DEEP-9.9M has no L=8 run yet, so its curve simply carries
# one fewer marker (see per-curve assertions below).
L_TICKS = [0, 1, 2, 5, 8, 10]
_L_POS = {l: i for i, l in enumerate(L_TICKS)}

LIRE_CAPTION = (
    "LIRE L-sweep on seven streaming datasets, one curve per dataset in each "
    "of the four panels (boundary reassignments, cumulative maintenance time, "
    "NPA violation rate, and Recall@100 vs L; log y except for recall). "
    "DEEP-9.9M uses the full-coverage rerun (coverage_fraction=1.0, full GT "
    "scope, nlist=4096); its SPFresh scan points are L in {0,1,2,5,10} (no "
    "L=8 run), so its curve carries five markers, while the other six datasets "
    "scan the full set L in {0,1,2,5,8,10} and carry six markers."
)


def _parse_l(method: str) -> int:
    return int(method.rsplit("L", 1)[1])


def _reassign_ytick(value, _pos=None) -> str:
    """Sparse symlog y-tick labels for the boundary-reassignment panel.

    The panel spans 0..~1e6 vectors.  Labelling every decade (10^0..10^6
    plus 0) crowds the single-column panel until the labels overlap, so only
    0 and the even decades (10^2 / 10^4 / 10^6) are labelled; minor tick
    marks/labels are suppressed separately.
    """
    if value == 0:
        return "0"
    return rf"$10^{{{int(round(np.log10(value)))}}}$"


def _lire_value(result: dict, key: str):
    """Panel value for one L setting (recall = final frontier point)."""
    if key == "npa_violation_rate":
        npa = result.get("npa") or {}
        return npa.get("violation_rate")
    if key == "recall":
        return result["frontier"]["recall"][-1]
    return result.get(key)


def build_lire_l_sweep_1M(scale: float) -> plt.Figure:
    d = scaled({"title": LIRE_TITLE_PT}, scale)
    plt.rcParams.update(scaled_rc(scale, {"title": LIRE_TITLE_PT}))
    # Native single-column canvas: the page width is exactly the 3.34 in
    # insertion width, all four panels sit in ONE row (a LIRE_H_IN strip
    # instead of the earlier 2x2 block at 3.72 in), and every axes keeps the
    # paper subplot aspect (1.2:1).
    fig, axes = plt.subplots(1, len(LIRE_PANELS),
                             figsize=(SINGLE_COL_IN, LIRE_H_IN))
    lock_aspect(axes)
    f_label = d["label"]
    f_title = d["title"]
    f_tick = d["tick"]
    f_legend = d["legend"]

    sweeps = {}
    for label, stem, ds_key in LIRE_DATASETS:
        sweep = _load_claim(stem)[ds_key].get("lire_l_sweep", {})
        sweeps[label] = {m: res for m, res in sweep.items() if "frontier" in res}

    # DEEP-9.9M has no L=8 run: assert the documented {0,1,2,5,10} scan set.
    deep_scan = sorted(_parse_l(m) for m in sweeps["DEEP-9.9M"])
    assert deep_scan == [0, 1, 2, 5, 10], deep_scan

    handles, labels = [], []
    label_set = {lab for lab, _, _ in LIRE_DATASETS}
    expected_pts = sum(len(sweeps[lab]) for lab, _, _ in LIRE_DATASETS)
    for ax, (title, key, ylabel, logy) in zip(axes, LIRE_PANELS):
        for label, _, _ in LIRE_DATASETS:
            sweep = sweeps[label]
            l_values = sorted(_parse_l(m) for m in sweep)
            xs = [_L_POS[l] for l in l_values]
            ys = [_lire_value(sweep[f"SPFresh-L{l}"], key) for l in l_values]
            # every curve starts at the L=0 slot, and L=0 disables
            # reassignment so its reassignment count is exactly 0
            assert xs[0] == _L_POS[0], (label, l_values)
            if key == "n_reassigns":
                assert ys[0] == 0, (label, ys[0])
            color, marker, ls = LIRE_DATASET_STYLE[label]
            ax.plot(xs, ys, color=color, marker=marker, linestyle=ls,
                    linewidth=1.0 * scale, markersize=3.0 * scale, label=label)
            if title == LIRE_PANELS[0][0]:
                handles.append(plt.Line2D([0], [0], color=color, marker=marker,
                                          linestyle=ls, linewidth=1.1 * scale,
                                          markersize=3.0 * scale))
                labels.append(label)
        # pad 3.0 pt rather than 0.6: the log axis puts its topmost tick label
        # ("10^6") above the axes frame, and with single-line titles there is
        # room between the legend and the axes to clear it.
        ax.set_title(title, pad=3.0 * scale, fontsize=f_title)
        # The single-row strip has no room for four rotated y-axis labels; the
        # panel title already names the quantity and its unit, so only the y
        # tick numbers are printed.  ``ylabel`` stays in LIRE_PANELS for
        # reference and is intentionally unused here.
        ax.set_xticks([_L_POS[l] for l in L_MAJOR_TICKS])
        ax.set_xticklabels([str(l) for l in L_MAJOR_TICKS])
        ax.set_xlim(-0.3, len(L_TICKS) - 0.7)
        if logy:
            if key == "n_reassigns":
                # L=0 is a real point (reassignments = 0).  A plain log axis
                # silently masks it, which makes every curve look as if it
                # started at L=1; a symlog axis (linthresh=1) can show 0.
                ax.set_yscale("symlog", linthresh=1.0)
                ax.set_ylim(bottom=0.0)
                # sparse major ticks + no minor labels: 0, 10^2, 10^4, 10^6
                ax.set_yticks([0.0, 1e2, 1e4, 1e6])
                ax.yaxis.set_major_formatter(FuncFormatter(_reassign_ytick))
                ax.yaxis.set_minor_locator(NullLocator())
            elif key == "maintenance_time_s":
                # three decades instead of the crowded seven-decade default
                ax.set_yscale("log")
                ax.set_yticks([1e0, 1e2, 1e4])
                ax.yaxis.set_minor_locator(NullLocator())
            elif key == "npa_violation_rate":
                # data sit in 0.037-0.156: label four plain-decimal values so
                # no mathtext mantissa pairs (6x10^-2 ...) crowd the panel
                ax.set_yscale("log")
                ax.set_ylim(0.02, 0.30)
                ax.set_yticks([0.02, 0.05, 0.1, 0.2])
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}"))
                ax.yaxis.set_minor_locator(NullLocator())
            else:
                ax.set_yscale("log")
                ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.set_ylim(0.5, 1.0)
            ax.set_yticks([0.6, 0.8, 1.0])
        ax.grid(True, linewidth=0.28 * scale, alpha=0.3)
        ax.tick_params(axis="both", which="major", pad=1.2 * scale,
                       length=2.3 * scale, labelsize=f_tick)
        # each panel must carry exactly one curve per dataset (6 L points for
        # six datasets, 5 for DEEP-9.9M which has no L=8 run yet)
        data_lines = [ln for ln in ax.get_lines() if ln.get_label() in label_set]
        assert len(data_lines) == len(LIRE_DATASETS), (
            f"{title}: {len(data_lines)} curves != {len(LIRE_DATASETS)}")
        n_pts = sum(len(line.get_xdata()) for line in data_lines)
        assert n_pts == expected_pts, (
            f"{title}: {n_pts} points != {expected_pts}")

    # 2-row legend (4 entries on the first row, 3 on the second).  Matplotlib
    # fills an ncol legend column-major, so the 7 dataset entries are
    # re-ordered to [L0,L4,L1,L5,L2,L6,L3]: array_split(_, 4) then yields the
    # columns [L0,L4] [L1,L5] [L2,L6] [L3], i.e. row 1 = L0 L1 L2 L3 and
    # row 2 = L4 L5 L6.  Four columns only fit the 3.34 in single-column
    # canvas with the tighter handle/column spacing below.
    _LEGEND_PERM = [0, 4, 1, 5, 2, 6, 3]
    handles = [handles[i] for i in _LEGEND_PERM]
    labels = [labels[i] for i in _LEGEND_PERM]
    fig.legend(handles, labels, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.0), columnspacing=0.55,
               handlelength=1.15, handletextpad=0.3, fontsize=f_legend)
    # rect top 0.93 keeps the 2-row legend tight against the two-line panel
    # titles (was 0.80 for the taller 3-row legend, which left a large gap).
    # Explicit margins instead of tight_layout: with box_aspect locked, the
    # panel width is set by the slot height, and tight_layout left ~0.31 in of
    # dead gutter between panels (0.52 in panels on a 3.34 in canvas).  These
    # numbers put a 0.645 in panel (0.5375 in tall at 1.2:1) in each slot,
    # 0.186 in gutter for the neighbouring y-tick numbers. Reserve an extra
    # 0.10 in below the axes for the shared label, preserving panel size.
    fig.subplots_adjust(left=0.072, right=0.986, top=0.5984, bottom=0.2526,
                        wspace=0.40)
    fig.supxlabel("LIRE repair neighborhood size", x=0.529, y=0.008,
                  fontsize=f_label)
    return fig


# --------------------------------------------------------------------------- #
# figure 3: P99 stability vs index size (3 rows x 2 cols, transposed)
# --------------------------------------------------------------------------- #
# rows = operations (query / insert / delete P99, top to bottom); cols =
# datasets (SIFT-1M / AudioSet-2M).  Each dataset has a single continuous
# index-size range, so a dataset column has no cross-dataset connecting
# segment; the per-operation y-range is shared across the two dataset columns.
LAT_ROWS = [
    ("query_p99_ms", "Query P99", "ms", False),
    ("insert_p99_us", "Insert P99", "$\\mu$s", True),
    ("delete_p99_us", "Delete P99", "$\\mu$s", True),
]
LAT_COLS = [
    ("SIFT-1M", "sift-128-euclidean"),
    ("AudioSet-2M", "audioset-unbal-128"),
]


def _load_latency(name: str) -> dict:
    path = _ROOT / "results" / "latency_scale" / f"{name}.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_latency_methods(name: str) -> dict:
    """Base 3-method file merged with the AdaIVF-only companion file."""
    result = _load_latency(name)
    extra = _ROOT / "results" / "latency_scale" / f"{name}_ada.json"
    if extra.exists():
        with extra.open(encoding="utf-8") as fh:
            result["methods"].update(json.load(fh)["methods"])
    return result["methods"]


def build_latency_stability_1M(scale: float) -> plt.Figure:
    d = scaled(None, scale)
    plt.rcParams.update(scaled_rc(scale))
    # Native single-column canvas, transposed to 3 rows (operations) x 2 cols
    # (datasets); every axes keeps the paper subplot aspect (1.2:1).
    fig, axes = plt.subplots(len(LAT_ROWS), len(LAT_COLS),
                             figsize=(SINGLE_COL_IN, 4.55))
    lock_aspect(axes.ravel())

    data = {name: _load_latency_methods(name) for _, name in LAT_COLS}

    # one shared y-range per operation row, across both dataset columns
    ylims: dict[str, tuple[float, float]] = {}
    for key, _, _, logy in LAT_ROWS:
        vals = []
        for _, name in LAT_COLS:
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
    for c, (_col_title, name) in enumerate(LAT_COLS):
        for method in METHOD_ORDER:
            mr = data[name].get(method)
            if not mr or "checkpoints" not in mr:
                continue
            cp = mr["checkpoints"]
            x = np.asarray(cp["index_size"], dtype=float) / 1e6
            color, marker, mls = style_for(method)
            for r, (key, _, _, _) in enumerate(LAT_ROWS):
                axes[r, c].plot(x, cp[key], color=color, marker=marker,
                                linestyle=mls, linewidth=1.0 * scale,
                                markersize=1.8 * scale, markevery=8)
            if method not in method_labels:
                method_handles.append(
                    plt.Line2D([0], [0], color=color, marker=marker,
                               linestyle=mls, linewidth=1.1 * scale,
                               markersize=3.0 * scale))
                method_labels.append(display_name(method))

    for c, (col_title, _name) in enumerate(LAT_COLS):
        for r, (key, row_title, unit, logy) in enumerate(LAT_ROWS):
            ax = axes[r, c]
            if r == 0:
                # dataset on the top edge of each column
                ax.set_title(col_title, pad=1.5 * scale, fontsize=d["title"])
            if r == len(LAT_ROWS) - 1:
                ax.set_xlabel("Index size (M)", labelpad=1.0 * scale)
            if c == 0:
                # operation + unit on the left edge of each row (single line
                # keeps the left margin narrow); the right column shares the
                # same y-range, so its y-tick labels are dropped to keep the
                # single-column panels wide.
                ax.set_ylabel(f"{row_title} ({unit})", labelpad=1.0 * scale)
            else:
                ax.tick_params(axis="y", labelleft=False)
            # sparse x ticks: the index-size range differs per dataset column
            # (SIFT ~0.5-0.75M, AudioSet ~1.0-1.53M), three labelled points
            # per panel keeps the single-column axes uncluttered.
            ax.set_xticks([0.5, 0.6, 0.7] if c == 0 else [1.1, 1.3, 1.5])
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
    fig.legend(method_handles, method_labels, loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.00),
               columnspacing=0.9, handlelength=1.5, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.945), w_pad=0.3, h_pad=0.4, pad=0.3)
    return fig


def main() -> None:
    out_dir = _ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    infos = []
    infos.append(save_native(
        build_frontier_all, out_dir / "frontier_all.pdf",
        DOUBLE_COL_IN, out_dir / "frontier_all.png", png_dpi=240))
    (out_dir / "frontier_all_caption.txt").write_text(
        FRONTIER_CAPTION + "\n", encoding="utf-8")

    infos.append(save_native(
        build_lire_l_sweep_1M, out_dir / "lire_l_sweep_1M.pdf",
        SINGLE_COL_IN, out_dir / "lire_l_sweep_1M.png", png_dpi=240))
    (out_dir / "lire_l_sweep_1M_caption.txt").write_text(
        LIRE_CAPTION + "\n", encoding="utf-8")

    # Do NOT write figures/latency_stability_1M.pdf here.  That file belongs to
    # scripts/plot_latency_stability.py; this script's older 4-method layout
    # used to overwrite it whenever the two ran in the wrong order.
    infos.append(save_native(
        build_latency_stability_1M,
        out_dir / "latency_stability_legacy_4method.pdf",
        SINGLE_COL_IN, out_dir / "latency_stability_legacy_4method.png",
        png_dpi=240))

    for label, info in zip(
            ("frontier_all", "lire_l_sweep_1M",
             "latency_stability_legacy_4method"), infos):
        report(info, label)
    print(f"Wrote frontier_all / lire_l_sweep_1M / "
          f"latency_stability_legacy_4method -> {out_dir}")
    print("The paper's latency_stability_1M figure is written by "
          "scripts/plot_latency_stability.py.")


if __name__ == "__main__":
    main()
