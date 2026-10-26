# NoraIVF

Code for the paper **"NoraIVF: No-Reassignment Incremental IVF Index Maintenance
for Streaming Vector Search"** (under submission).

NoraIVF is a memory-native dynamic IVF index for streaming vector search. Its
**SCDKM** (Streaming Coordinate Descent K-Means) protocol updates vector
assignments using maintained cluster statistics and a configurable candidate
set. Dense postings and direct identifier lookup support insertion and
swap-and-pop deletion without a separate LIRE-style boundary-repair pass.

The repository also contains the baselines the paper compares against:
`Frozen`, `Rebuild`, `DeDrift`, `SPFresh` (LIRE repair), and `AdaIVF`.

## Repository layout

```
noraivf/                       Core Python package
  core.py                      IVF index implementations (Frozen, Rebuild, DeDrift,
                               SPFresh, NovaIVF, AdaIVF) + the SCDKM protocol
  datasets.py                  Dataset loading, generation, ground truth
  lean.py                      Array-backed dict views for reference protocols
  lean_methods.py              Lean index classes + the candidate-set-width
                               (restricted / top-L / full-k) CDKM variants
  scale_storage.py             Lossless SIFT uint8 mappings, compact postings,
                               reference Nora maintenance and batched LIRE port
  scale_ada.py, scale_dedrift.py         Compact Ada-IVF and DeDrift maintenance
  native_scan.py, uint8_scan.cpp Shared exact single-thread SIFT query scanner
  logging/                     Structured experiment logging
    logger.py                  ExperimentLogger: JSONL events, phases, checkpoints
    summary.py                 summarize_run / write_summary / format_summary
    env.py                     Environment and data-provenance capture

experiments/                   Experiment entry points
  run_claim_evidence_annbench.py         Main multi-method suite (L2 distance,
                                         full-coverage streaming protocol)
  run_experiments.py                     Earlier 30k-scale suite (cosine/IP)
  run_latency_scale.py                   Million-scale P99 stability and
                                         mixed-workload query/insert/delete latency
  run_microbenchmark_four_methods.py     Insert/delete microbenchmark
  run_p99_stability_5datasets.py         P99 query-latency stability table
  measure_scan_width.py                  SCDKM candidate-set-width microbenchmark
  run_cdkm_vs_lloyd_local.py             CDKM vs Lloyd local convergence diagnostic
  run_boundary_sampled_scdkm_diagnostic.py
  run_local_cdkm_repair_diagnostic.py
  fidelity_check.py                      Equivalence gate for the lean substrate
  run_billion_probe.py                   Earlier lean probe; not the new 1B evidence
  build_sift1b_scale.py                  Checkpointed full-corpus IVF assignments
  run_sift1b_scale.py                    Full1B query + bounded maintenance probe
  prepare_sift1b_ada_stats.py             Initial Ada-IVF statistics from shared cache
  run_sift1b_fourway.py                   Matched four-policy full-SIFT1B workload

scripts/                       Data preparation, orchestration, plotting, QA
  prepare_annbench_datasets.py           ann-benchmarks HDF5 -> canonical npy layout
  prepare_audioset.py                    AudioSet TFRecord embeddings -> npy
  probe_audioset_tfrecord.py             AudioSet container inspection
  prepare_sift1b.py                      bigann .bin -> npy (lossless float16)
  prepare_real_vector_datasets.py        Download + encode real image datasets
  prepare_l2_clip_datasets.py            Synthetic CLIP-style datasets
  master_run.sh, master_run2.sh          Orchestration for the full-coverage suite
  run_*.sh                               Per-stage runners (nlist sweep, DeDrift,
                                         multiseed, L8 + CDF, ...)
  merge_method_results.py                Merge single-method JSONs into the main files
  recompute_maintenance_budget.py        Rebuild a results file's budget section
  check_dataset_shapes.py                Dataset shape / load-path self-check
  verify_figure_*.py                     Figure QA (sizes, gaps, subplot aspects)
  log_summary.py                         CLI summary of one logged run
  summarize_sift1b_fourway.py             Recompute the four-policy result table

  Figure generators:
    plot_claim_figures.py                Per-dataset QPS-recall frontiers, L-sweeps
    plot_rebuttal_figures.py             Composite panels (frontier_all, lire_l_sweep_1M)
    plot_robustness.py                   Cluster-count sweep figure
    plot_latency_stability.py            P99 stability vs index size
    plot_tail_latency_cdf.py             Tail-latency CDFs
    plot_split_time_breakdown.py         Split-time 100% stacked bars
    paper_figstyle.py                    Shared typography for the sigconf layout

plots/                         Earlier 30k-scale plotting scripts
tests/                         Unit tests
results/                       Experiment output; a curated subset is committed
figures/                       Generated figures (committed)
data/                          Datasets (not committed)
```

## Installation

```bash
git clone https://github.com/whu-totemdb/NoraIVF.git
cd NoraIVF

pip install -r requirements.txt
```

Run scripts from the repository root. No editable installation is required;
the repository does not contain Python packaging metadata. The optional native
query scanner requires a C++17 compiler; its build command appears below.

## Preparing Data

Every experiment script expects datasets in `data/`, each a directory holding
`train.npy`, `queries.npy`, and `gt.npy` (unit-normalised `float32` vectors with
exact ground-truth neighbours).

### Option 0: convert local ann-benchmarks HDF5 files (recommended)

Given the standard ann-benchmarks HDF5 files, the pipeline converts them into
canonical dataset directories and records the run in the structured logger
(source fingerprints, per-file SHA-256 provenance, metrics):

```bash
python scripts/prepare_annbench_datasets.py \
    --source-dir /path/to/ann_benchmark \
    --out-root data/annbench

# subset, force re-conversion, or disable logging
python scripts/prepare_annbench_datasets.py \
    --datasets sift-128-euclidean mnist-784-euclidean \
    --overwrite --no-log
```

Output layout (loadable by `load_npy_vector_dataset_l2`):

```
data/annbench/<name>/
    train.npy        # float32 base vectors (N, d)
    queries.npy      # float32 queries (Q, d)
    gt.npy           # int32 ground truth (Q, 100)
    distances.npy    # float32 ground-truth distances (Q, 100)
    meta.json        # source fingerprint, shapes, dtype, normalization flag
```

Re-running is idempotent (unchanged datasets are skipped via source
fingerprint; `--overwrite` forces re-conversion). List converted datasets from
Python with `noraivf.list_datasets("data/annbench")`.

### Option 1: AudioSet embeddings

```bash
python scripts/prepare_audioset.py      # TFRecord embeddings -> npy
```

### Option 2: SIFT1B (for the scale probe)

The current scale experiment reads the original `base.bin`, `query.bin` and
`gt.bin` directly. Each file starts with two little-endian uint32 values
`(rows, dimensions)`, followed by row-major uint8 vectors or int32 ground-truth
IDs. The full base has shape `(1_000_000_000, 128)` and is exactly
128,000,000,008 bytes. No float16/float32 corpus copy is needed. See the scale
instructions below.

The older lean probe accepts this optional conversion:

```bash
# expects base.bin / query.bin / gt.bin in the bigann layout
python scripts/prepare_sift1b.py \
    --base base.bin --query query.bin --gt gt.bin --out data/annbench/sift1b
```

`train.npy` is written as **float16**. SIFT components are integers in 0..255 and
float16 represents every integer up to 2048 exactly, so the conversion is
lossless; it halves the on-disk footprint (256 GB instead of 512 GB), and disk
can be a constraint when retaining this optional copy. This conversion is not used by
`build_sift1b_scale.py` or `run_sift1b_scale.py`.

### Option 3: synthetic CLIP-style datasets

```bash
python scripts/prepare_l2_clip_datasets.py \
    --datasets core50 ScanNetCLIP Ego4D \
    --n-vectors 30000 --n-queries 1000 --dim 512 \
    --out-root data
```

### Option 4: download and encode real datasets

Requires PyTorch and `torchvision`. Downloads images and encodes them with
ResNet18.

```bash
python scripts/prepare_real_vector_datasets.py \
    --datasets core50 scannet ego4d \
    --train-count 30000 --query-count 1000 \
    --out-root data
```

## Running Experiments

All scripts run from the repository root. Results go to `results/`, figures to
`figures/`.

### Main suite (L2 distance, full-coverage streams)

```bash
# quick smoke test
python experiments/run_claim_evidence_annbench.py --smoke --download

# the protocol the paper reports: build on data[:n_init], then stream n_stream
# updates, with per-dataset operating points.  See --help for the full matrix.
python experiments/run_claim_evidence_annbench.py \
    --datasets sift-128-euclidean deep-image-96-angular \
    --data-dir data/annbench \
    --n-init 500000 --n-stream 500000 --nlist 4096 --seed 42
```

Output: `results/claim_evidence_<dataset>.json`.

### Million-scale P99 stability and mixed-workload latency

```bash
python experiments/run_latency_scale.py --help
```

Output: `results/latency_scale/<dataset>.json` plus the per-operation arrays in
`<dataset>_raw.json` (these raw files are large and are not committed).

### P99 stability, microbenchmark, and SCDKM diagnostics

```bash
python experiments/run_p99_stability_5datasets.py --max-train 30000 --n-batches 150
python experiments/run_microbenchmark_four_methods.py
python experiments/measure_scan_width.py --nlist 4096 --force-splits 15

python experiments/run_cdkm_vs_lloyd_local.py \
    --datasets MNIST Fashion CoRe50 ScanNet Ego4D --max-train 50000 --local-parts 20
```

### Earlier 30k-scale suite

```bash
python experiments/run_experiments.py     # requires data/ScanNetCLIP, data/Ego4D
```

### LIRE L-sweep diagnostics

```bash
python experiments/run_claim_evidence_annbench.py \
    --only-lire-sweep --l-values 0 1 2 5 10 \
    --datasets mnist-784-euclidean fashion-mnist-784-euclidean \
    --output results/lire_l_sweep_5datasets.json

python scripts/add_nora_to_lire_sweep.py \
    --input results/lire_l_sweep_5datasets.json \
    --output results/lire_l_sweep_5datasets_with_nora.json

python experiments/run_boundary_sampled_scdkm_diagnostic.py \
    --input results/lire_l_sweep_5datasets_with_nora.json \
    --neighbors 2 --samples 4 8 16 32

python experiments/run_local_cdkm_repair_diagnostic.py \
    --input results/lire_l_sweep_5datasets_with_nora.json --l-values 2 5
```

## Results in this repository

`results/` is ignored by default, so files written by a new run stay untracked.
The JSON and CSV that the paper's tables and figures read are committed
explicitly:

| Path | Read by |
|---|---|
| `results/claim_evidence_*.json` | Main maintenance / recall tables; split-time breakdown |
| `results/nlist_sweep_full/`, `results/nlist_sweep/` | Cluster-count sweep table and figure |
| `results/multiseed_large/`, `results/multiseed_dedrift/`, `results/multiseed_deep_fullcov/`, `results/claim_evidence_multiseed60k.json` | Multi-seed stability |
| `results/latency_scale/*.json` (aggregated) | Latency comparison panels |
| `results/p99_stability_5datasets.{json,csv}` | Tail-latency table |
| `results/microbenchmark_four_methods_30k.{json,csv}` | Insert / delete microbenchmark |
| `results/scdkm_scan_width_1m.json` | SCDKM candidate-set-width microbenchmark |
| `results/lean_fidelity.json` | Lean-substrate equivalence gate |
| `results/lire_l8/`, `results/sift1m_*.csv`, `results/claim_evidence_summary_nprobe8.csv` | L-sweep and summary panels |

Three families are deliberately **not** committed, because of their size
(about 236 MB together):

* `results/latency_scale/*_raw.json` (188 MB) — per-operation latency arrays,
  read only by `scripts/plot_tail_latency_cdf.py`. Regenerate with
  `experiments/run_latency_scale.py`.
* `results/latency_scale_dedrift/` (49 MB) — the DeDrift pass over the same
  protocol, read by `scripts/plot_latency_stability.py`. Regenerate with
  `scripts/run_dedrift_latency.sh`.
* `results/logs/`, `results/pi_logs/` — runtime logs, which are not inputs to
  anything.

Those two figures therefore need one experiment run each before they can be
reproduced; every other figure and table works from the committed JSON alone.

## Generating Plots

Figure scripts share one typography module so that the panels land consistently
in the two-column ACM sigconf layout (`\linewidth` = 3.34 in, `\textwidth` =
7.00 in). They use Linux Libertine O, matching the official VLDB 2027 template's
default body text. Install TeX Live's `libertine` package and put `kpsewhich` on
`PATH`; verify discovery with `kpsewhich LinLibertine_R.otf`. Alternatively,
point the helper to
a font directory:

```bash
export NORA_PAPER_FONT_DIR=/path/to/libertine-fonts
```

That directory must contain all four faces:

```text
LinLibertine_R.otf
LinLibertine_RB.otf
LinLibertine_RI.otf
LinLibertine_RBI.otf
```

The plotting helper also needs Poppler's `pdfinfo` on `PATH` to verify PDF
dimensions. Run the figure generators from the repository root:

```bash
python scripts/plot_claim_figures.py    --input 'results/claim_evidence_*.json'
python scripts/plot_rebuttal_figures.py
python scripts/plot_robustness.py
python scripts/plot_latency_stability.py
python scripts/plot_tail_latency_cdf.py
python scripts/plot_split_time_breakdown.py
```

`scripts/plot_latency_stability.py` is the authoritative writer of
`figures/latency_stability_1M.pdf`. `scripts/plot_rebuttal_figures.py` holds a
superseded four-method variant of the same figure and writes it as
`latency_stability_legacy_4method.*`, so the two cannot overwrite each other.

`scripts/restyle_diagram_fonts.py` uses PyMuPDF from `requirements.txt` to
replace text in the three vector diagrams while retaining their artwork.
Pass the original, unmodified diagram PDFs as `--source-dir`, a separate
`--out-dir`. Font discovery uses the same `NORA_PAPER_FONT_DIR` or `kpsewhich`
as the plots; `--font-dir` can override discovery for the diagrams.

The older 30k-scale plots live in `plots/`:

```bash
python plots/plot_stability_summary_5datasets.py --input results/p99_stability_5datasets.json
python plots/plot_stability_traces_5datasets.py  --input results/p99_stability_5datasets.json
python plots/plot_tail_latency_cdf_5datasets.py  --input results/p99_stability_5datasets.json
python plots/plot_latency_stability_combined.py  --microbench results/microbenchmark_four_methods_30k.json
python plots/plot_lire_l_sweep_trends.py         --input results/claim_evidence_annbench.json
python plots/plot_qps_recall_with_l0.py          --input results/claim_evidence_annbench.json
```

## Running Tests

```bash
python tests/test_claim_evidence_l2.py
python tests/test_data_pipeline.py
python tests/test_logging.py
python tests/test_ada_ivf.py
python tests/test_scale_storage.py
python tests/test_scale_dedrift.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python tests/test_scale_ada.py
```

The native-scanner check and optional real-SIFT adapter checks take dataset
paths; see the SIFT1B instructions below.

## Logging

Every experiment run writes a self-contained, disk-persisted log under
`results/logs/<run_id>/`:

```
results/logs/<run_id>/
    meta.json            # env + config + data provenance (git commit, machine, packages)
    run.jsonl            # append-only JSONL event stream, flushed per event
    errors.jsonl         # error events with full stack traces
    events.summary.json  # written by the summary tool
```

Usage from an experiment script (the field schema is documented in
`noraivf/logging/logger.py`):

```python
from noraivf.logging import ExperimentLogger

with ExperimentLogger("exp-scale-10m", config={"nlist": 4096, "seed": 42}) as lg:
    with lg.phase("build"):
        lg.checkpoint("build", batch=1, n=10000)     # progress markers
    with lg.phase("stream"):
        lg.metric("stream", throughput=31250.0, n_splits=128)
    lg.add_data_provenance("data/sift1m/train.npy")  # sha256 fingerprint
    try:
        ...
    except Exception as exc:
        lg.error(exc, phase="stream", batch_idx=42)   # stack trace -> errors.jsonl
```

View a run:

```bash
python scripts/log_summary.py --latest
python scripts/log_summary.py --run-dir results/logs/<run_id>
python scripts/log_summary.py --run-dir results/logs/<run_id> --json
```

Every event is appended and flushed immediately (crash-safe); error events are
fsynced and mirrored to `errors.jsonl`; `meta.json` pins the exact environment
(git commit, python and package versions, machine, GPU: none) and data
fingerprints, so a run can be reproduced and a failure traced.

## Historical two-policy SIFT1B probe and shared construction

The 2026-10-13 two-policy probe below is retained for construction and validation
provenance. Its performance table is superseded by the completed four-policy
comparison in the next section; measurements from the two protocols must not
be combined.

This earlier scale experiment stores and assigns all one billion real SIFT vectors.
It compares **NoraIVF and the repository's SPFresh/LIRE port only**: 100 sampled
official queries, three forced splits of the same original partition IDs,
then 1,000 deterministic delete/reinsert pairs using actual corpus rows.
The insertions can trigger additional splits; these are counted and their
costs remain in the raw insertion times. This is not a full-coverage billion
update stream, an all-baseline suite, or a full C++ implementation.

The corpus remains lossless uint8. A stable uint32 posting array and int32
assignment/maps replace per-vector Python containers. Nora uses the existing
`L2NovaIVF` maintenance methods. SPFresh uses the same two-means/LIRE protocol
with batched frozen-centroid assignment, a near-tie fallback to the reference
float32 distance expression, an added ID-to-partition map, and vectorized
linear-find/shift deletion. Reference-fidelity checks cover ordered postings,
centroids, maps, consecutive splits and a true merge. The shared native query
kernel accumulates exact integer squared L2 distances and uses deterministic
distance/ID tie ordering; it is checked against NumPy distances and results.
These explicit optimizations differ from the earlier small-scale scripts,
so their latency values should not be joined into one scaling curve.

Run from the repository root on a Linux host with sufficient RAM and disk:

```bash
SIFT_BASE=/path/to/bigann/base.bin
SIFT_QUERY=/path/to/bigann/query.bin
SIFT_GT=/path/to/bigann/gt.bin
SIFT_CACHE=/path/to/sift1b_cache

# Train on 1,048,576 sampled rows; assign every one of the 1B rows to 4096 centers.
python experiments/build_sift1b_scale.py \
  --base "$SIFT_BASE" --out "$SIFT_CACHE" --nlist 4096 \
  --train-sample 1048576 --niter 25 --chunk 2000000 \
  --threads 32 --benchmark-threads --seed 42

g++ -O3 -std=c++17 -march=native -shared -fPIC \
  noraivf/uint8_scan.cpp -o "$SIFT_CACHE/uint8_scan.so"

python tests/test_scale_storage.py "$SIFT_BASE"
python tests/test_native_scan.py "$SIFT_BASE" "$SIFT_QUERY" "$SIFT_CACHE/uint8_scan.so"

python experiments/run_sift1b_scale.py \
  --base "$SIFT_BASE" --queries "$SIFT_QUERY" --gt "$SIFT_GT" \
  --cache "$SIFT_CACHE" --out results/sift1b_scale_reproduction.json \
  --methods NoraIVF SPFresh --force-splits 3 --update-pairs 1000 \
  --query-count 100 --nprobes 1 8 32 --threads 32 --seed 42 \
  --audit-sample 20000 --verify-gt-queries 1 \
  --scan-library "$SIFT_CACHE/uint8_scan.so"
```

The builder records its chosen FAISS thread count after benchmarking 8/16/32/64
threads. The measured run trained with 32 and assigned with 64 FAISS threads.
Maintenance configures 32 FAISS threads; Nora's coordinate loop and the shared
native query scan are serial. NumPy/FAISS library worker pools can exist beyond
the configured FAISS count, so this is not a claim of 32 resident OS threads.
All shared corpus, assignment and posting mappings receive an explicit
untimed page touch before either method, followed by three untimed queries
per measured `nprobe`. Query timings include selecting the top 100 results.

The supplied ground truth is checked independently by scanning all 1B rows for
one sampled query and matching every top-100 integer distance rank; all 100
measured query rows are also checked for valid IDs and sorted distances.
Reinserted IDs map back to their original IDs for recall, preserving the vector
multiset. Ties can still have multiple valid neighbor ID sets. An optional
`--reuse-gt-from` reuses only a successful validation phase after checking the
corpus configuration, query IDs and query/GT hashes; it never reuses timings.

The measured host had two Xeon Platinum 8352V CPUs (72 physical cores,
144 logical CPUs), 503 GiB RAM, Python 3.11.16, NumPy 2.4.6, FAISS 1.15.0
and g++ 11.4.0. Measured shared preparation took 133.85 s for sampled training,
2,121.59 s for **all 1B assignments**, 224.80 s for stable posting construction
and 306.09 s for reference-order sums. These are individual measured stages,
not a claim that 133.85 s trains on all 1B rows. The final matched query and
maintenance probe completed in 1,470.05 s.

| Bounded maintenance measurement | NoraIVF | SPFresh port |
|---|---:|---:|
| Three forced split events, total seconds | 208.36 | 237.51 |
| Additional splits during 1,000 update pairs | 5 | 5 |
| All insertion calls, seconds (including those splits) | 211.70 | 366.52 |
| All deletion calls, seconds | 0.0172 | 0.1922 |
| Total bounded maintenance, seconds | 420.07 | 604.22 |
| External LIRE reassignment operations | 0 | 2,319,537 |
| Final live corpus / partitions | 1,000,000,000 / 4,104 | 1,000,000,000 / 4,104 |

Both methods retained Recall@10 0.981 and Recall@100 0.9636 at `nprobe=32`
after all maintenance. Query QPS at that setting was 2.072 for Nora and 1.646
for SPFresh after updates; before updates their measurements were within a
few percent. This is one fixed-order execution, so the later timing difference
is descriptive and does not establish a causal query speedup. See every
operating point and raw per-query value in the results.

Results and provenance for the 2026-10-13 run are under
`results/sift1b_scale_20261013/`: `full1b_final_probe.json` is the final matched
measurement; `build_state.json`, `postings_state.json`,
`build_memory_observed.json`, `independent_gt_validation.json` and
`source_and_validation.json` record construction, checks and source hashes.
`measurement_summary.json` derives compact tables from that complete result
and records its SHA-256. `independent_audit_summary.json` records a separate
full-cache audit: all 1B assignments were counted again, an independent presence
bitmap found zero missing or duplicate posting IDs, and all 1B posting IDs
matched their assigned partitions. It also records full-file SHA-256 hashes
of the raw corpus and cache arrays. This audit ran after the timed process
exited; its original script, full report and operation log remain private.
The result JSON includes each query, each delete/reinsert, every forced split,
additional split/merge counts, audits, configuration and code/library hashes.
The stopped `full1b_probe.json` setup run and `full1b_native_probe.json` preliminary
run are excluded from final performance results, as are all `smoke_20k*` files.
Private operation logs and loaded source snapshots are retained outside this
public repository. Results are ignored by default and must be included
explicitly when preparing an artifact release.

RSS includes resident corpus and cache mappings. Peak RSS is cumulative over
the whole process, including both methods; phase-boundary `VmRSS` is current
memory, not a per-method peak. The builder's original per-chunk peak field
misses the final counting stage: use the separately observed Linux VmHWM in
`build_memory_observed.json` for that build. The uint8 storage advantage is
specific to these lossless SIFT components and does not establish the memory
cost of arbitrary float32 embeddings. Three forced events, one seed and 100
queries support this bounded probe, not a steady-state or production-scale
throughput claim.

## Matched four-policy SIFT1B workload

`experiments/run_sift1b_fourway.py` extends the scale evaluation to SPFresh,
Ada-IVF, DeDrift and NoraIVF. This is a separate protocol from the two-policy
forced-split probe above; their measurements must not be mixed. The four-policy
run completed on 2026-10-14 from 13:03:24 to 15:37:43 Asia/Shanghai
(05:03:24–07:37:43 UTC), in 9,259.43 seconds (about 2 h 34 min), excluding
prior construction and pilots. All four methods completed their workloads and
full final ID audits. These measurements supersede the earlier two-policy
performance table as the main SIFT1B result; pilots remain excluded.

All policies start from the same complete billion-vector index and receive
1,000 identical delete/reinsert pairs from the largest initial partition, plus
one shared workload query after every ten pairs. The largest initial partition
already contains 1,327,882 vectors, compared with a mean of 244,140.625; this
initial imbalance is part of the evaluated workload. Each policy uses its own
maintenance triggers; there are no forced splits. The vector multiset is
unchanged, so this measures a finite localized burst, not distribution drift
or sustained billion-update throughput. The query evaluation uses 100 separate
official queries and never changes policy state. Only workload queries feed
Ada's temperature statistics.

Prepare Ada's initial statistics from the completed cache, preserving the
original whole-partition float32 reduction:

```bash
python experiments/prepare_sift1b_ada_stats.py \
  --base /data/bigann/base.bin --cache /data/sift1b_cache
```

Run from this repository with the existing environment. `--validated-gt-from`
must point to a completed compatible full-corpus record containing the
independent exhaustive ground-truth check. The native scan library is the
same library used by the preceding scale probe.

```bash
OMP_NUM_THREADS=32 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
timeout --signal=TERM --kill-after=60s 28800s \
python -u experiments/run_sift1b_fourway.py \
  --base /data/bigann/base.bin --queries /data/bigann/query.bin \
  --gt /data/bigann/gt.bin --cache /data/sift1b_cache \
  --scan-library /data/sift1b_cache/uint8_scan.so \
  --validated-gt-from results/sift1b_scale_20261013/full1b_final_probe.json \
  --out results/sift1b_fourway/full_fourway.json \
  --methods SPFresh NoraIVF DeDrift AdaIVF \
  --update-pairs 1000 --workload-query-every 10 --query-count 100 \
  --query-repeats 3 --threads 32 --nprobes 1 8 32 --max-seconds 28700
```

The primary metric, `total_update_wall_s`, includes complete delete/insert
calls, policy detection, statistics and triggered maintenance. Structural
event timers are recorded separately and must not replace this metric.
Bootstrap, input gathering, evaluation and consistency audits are outside
update timing. Methods execute sequentially in the declared order; three
final query repeats reuse one final index and are not independent update trials.

The adapters preserve their respective repository policies and use compact
containers. Ada additionally uses stable float64 maintenance distances with
direct recomputation of near ties: the original float32 norm/dot expression
can change rankings with BLAS block shape. Tests distinguish mathematical
distance correctness, equivalence to a stable-distance reference and observed
fidelity to the original reference on real SIFT rows. This is not a claim of
universal bitwise equivalence. If Ada's global-rebuild trigger fires, the
adapter stops explicitly: a billion-row global rebuild is not implemented.

Every completed method receives an untimed exact final ID audit: all posting
IDs, dense partition maps, Nora's position map, deletion tombstones and the
trace-derived reinserted vectors are checked. A byte bitmap proves uniqueness
and complete live-ID coverage. Configuration, code, library, cache and trace
identities are saved. Use ordinary Python; optimized `python -O` is rejected.

Generate the table only from a complete, audited run:

```bash
python scripts/summarize_sift1b_fourway.py \
  --result results/sift1b_fourway/full_fourway.json \
  --summary results/sift1b_fourway/measurement_summary.json \
  --rows results/sift1b_fourway/table10_rows.tex
```

The summarizer rejects pilots and incomplete results, recomputes timings and
recall from individual records, verifies paired query identities, and reports
median QPS across the three repeats. Output rows have one update-time column
and three columns each for Recall@100 and QPS, labelled by probe budget.

### Completed measurements and verification

The complete run is recorded in `results/sift1b_fourway_20261014/`:

- `full_fourway.json`: raw per-operation and per-query records, configuration,
  run identity, completed-method flags, structural events and full ID audits.
- `measurement_summary.json`: recomputed totals and query statistics, bound to
  the raw JSON by SHA-256.
- `table10_rows.tex`: four-method table rows generated from the validated
  summary, with their SHA-256 recorded in that summary.
- `full_source_manifest.json` and `full_run_sources.tgz`: source hashes and the
  source snapshot used by the completed run.

The downloaded raw result and generated table both match the summary's recorded
hashes. Results remain ignored by default; their presence in a working tree does
not mean an artifact release or public repository upload has been made. The
paper and response are being updated from this completed result.

Update time includes complete delete and insert calls. Recall is measured on
100 common evaluation queries, disjoint from the 100 workload queries. QPS is
the median of three repeats on one final index; parentheses give `nprobe`.

| Method | Update time (s) | Recall@100 (1) | Recall@100 (8) | Recall@100 (32) | QPS (1) | QPS (8) | QPS (32) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SPFresh | 558.2 | 0.4045 | 0.8300 | 0.9636 | 47.76 | 6.23 | 1.62 |
| Ada-IVF | 4736.5 | 0.4128 | 0.8281 | 0.9629 | 54.05 | 7.06 | 1.80 |
| DeDrift | 2076.7 | 0.3817 | 0.7778 | 0.9332 | 52.05 | 6.64 | 1.66 |
| NoraIVF | 288.1 | 0.4072 | 0.8307 | 0.9636 | 46.11 | 6.15 | 1.60 |

NoraIVF has the lowest update cost in this run. Ada-IVF has the highest QPS at
all three probe budgets; NoraIVF does not have a query-throughput advantage over
SPFresh in this comparison. SPFresh and NoraIVF perform eight and six splits,
respectively. DeDrift performs 1,010 rebalances. Ada-IVF performs one local
reindex covering 199,634,222 rows and 38 splits, with no global rebuild; the
large pool reflects the initial imbalance and its neighborhood expansion.
These observations describe one finite localized update trial, not a steady
per-update rate or a distribution-drift benchmark.

The full final ID audits verify exactly one billion live vectors for every
method, including ID uniqueness, expected deleted/reinserted sets, partition
maps and NoraIVF position maps. The independent exhaustive ground-truth check
in the reused construction record covers one official query, not all queries.
Peak RSS is 136.5 GiB for the combined process and is not a per-method peak.

## License

MIT — see `LICENSE`.
