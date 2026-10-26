"""Checkpointed exact IVF assignment of the real BIGANN uint8 corpus.

Uses the existing BIGANN binary files directly, without duplicating vectors.
This build is measured, not extrapolated.  A checkpoint records the number of
rows whose assignments were flushed; resume never treats an unfinished array
as a complete index.  Outputs are separate from published experiment results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import struct
import sys
import time

import faiss
import numpy as np


def binary_matrix(path, dtype):
    path = Path(path)
    with path.open("rb") as f:
        n, d = struct.unpack("<II", f.read(8))
    expected = 8 + n * d * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {expected} bytes, got {path.stat().st_size}")
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(n, d))


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def sample_fingerprint(path, blocks=32, block_bytes=65536):
    path = Path(path)
    size = path.stat().st_size
    offsets = np.linspace(0, max(0, size-block_bytes), blocks, dtype=np.int64)
    h = hashlib.sha256()
    with path.open("rb") as f:
        for pos in offsets:
            f.seek(int(pos))
            h.update(struct.pack("<Q", int(pos)))
            h.update(f.read(block_bytes))
    return dict(path=str(path.resolve()), bytes=size, mtime_ns=path.stat().st_mtime_ns,
                sampled_sha256=h.hexdigest(), sampled_blocks=blocks,
                sampled_block_bytes=block_bytes, note="sampled fingerprint, not full-file hash")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--nlist", type=int, default=4096)
    p.add_argument("--niter", type=int, default=25)
    p.add_argument("--train-sample", type=int, default=1048576)
    p.add_argument("--chunk", type=int, default=1000000)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-vectors", type=int, default=0)
    p.add_argument("--benchmark-threads", action="store_true")
    a = p.parse_args()
    faiss.omp_set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = binary_matrix(a.base, np.uint8)
    n = min(len(data), a.max_vectors) if a.max_vectors else len(data)
    config = dict(base=sample_fingerprint(a.base), n=n, dim=data.shape[1],
                  nlist=a.nlist, niter=a.niter, train_sample=a.train_sample,
                  seed=a.seed, storage="uint8, lossless original BIGANN components",
                  assignment="exact IndexFlatL2 over all coarse centroids")
    state_path = out / "build_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["config"] != config:
            raise RuntimeError("Existing cache configuration differs; use a new output directory")
    else:
        free = os.statvfs(out).f_bavail * os.statvfs(out).f_frsize
        if free < n * 4 + 5 * 1024**3:
            raise RuntimeError("Insufficient space for assignment cache plus 5 GiB reserve")
        state = dict(config=config, completed_rows=0, complete=False, train_s=None,
                     assignment_s=0.0, assignment_segments=[], thread_benchmark=[],
                     started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     environment=dict(host=platform.node(), python=sys.version,
                                      numpy=np.__version__, faiss=faiss.__version__,
                                      platform=platform.platform(), command=sys.argv))
        atomic_json(state_path, state)
    if state["complete"]:
        print("Cache already complete", flush=True)
        return
    cpath = out / "centroids.npy"
    if cpath.exists():
        centroids = np.load(cpath)
    else:
        t = time.perf_counter()
        rng = np.random.default_rng(a.seed)
        # With replacement: O(sample) memory, unbiased sampling across all 1B IDs.
        ids = rng.integers(0, n, size=min(a.train_sample, n), dtype=np.int64)
        train = np.asarray(data[ids], dtype=np.float32)
        state["sample_ids_sha256"] = hashlib.sha256(ids.tobytes()).hexdigest()
        state["sample_unique_rows"] = len(np.unique(ids))
        km = faiss.Kmeans(data.shape[1], a.nlist, niter=a.niter, seed=a.seed,
                          verbose=True, gpu=False)
        km.train(train)
        centroids = km.centroids.copy()
        np.save(cpath, centroids)
        state["train_s"] = time.perf_counter() - t
        state["centroids_sha256"] = hashlib.sha256(centroids.tobytes()).hexdigest()
        atomic_json(state_path, state)
        del train, ids, km
        print(f"TRAIN complete {state['train_s']:.3f}s", flush=True)
    quant = faiss.IndexFlatL2(data.shape[1])
    quant.add(centroids)
    if a.benchmark_threads and not state["thread_benchmark"]:
        probe = np.asarray(data[:min(200000, n)], dtype=np.float32)
        for threads in [8, 16, 32, 64]:
            faiss.omp_set_num_threads(threads)
            t = time.perf_counter()
            quant.search(probe, 1)
            elapsed = time.perf_counter() - t
            state["thread_benchmark"].append(dict(threads=threads, seconds=elapsed,
                                                  vectors=len(probe)))
            print(f"BENCH threads={threads} elapsed={elapsed:.3f}s", flush=True)
        a.threads = min(state["thread_benchmark"], key=lambda x: x["seconds"])["threads"]
        del probe
    faiss.omp_set_num_threads(a.threads)
    state["assignment_threads"] = a.threads
    apath = out / "assign.npy"
    assign = np.lib.format.open_memmap(apath, mode="r+" if apath.exists() else "w+",
                                       dtype=np.int32, shape=(n,))
    start = state["completed_rows"]
    seg = dict(start=start, threads=a.threads, start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    state["assignment_segments"].append(seg)
    t = time.perf_counter()
    old_seconds = state["assignment_s"]
    for s in range(start, n, a.chunk):
        e = min(s+a.chunk, n)
        x = np.asarray(data[s:e], dtype=np.float32)
        _, labels = quant.search(x, 1)
        assign[s:e] = labels[:, 0]
        del x, labels
        assign.flush()
        elapsed = time.perf_counter() - t
        state.update(completed_rows=e, assignment_s=old_seconds+elapsed,
                     peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
        seg.update(end=e, elapsed_s=elapsed)
        atomic_json(state_path, state)
        print(f"ASSIGN {e:,}/{n:,} segment={elapsed:.1f}s rate={(e-start)/elapsed:,.0f}/s "
              f"eta={(n-e)/max((e-start)/elapsed,1)/60:.1f}min rss={state['peak_rss_gib']:.2f}GiB", flush=True)
    counts = np.bincount(assign, minlength=a.nlist)
    np.save(out / "counts.npy", counts)
    state.update(complete=True, cluster_min=int(counts.min()), cluster_max=int(counts.max()),
                 cluster_mean=float(counts.mean()), cluster_p99=float(np.percentile(counts, 99)),
                 peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                 finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    atomic_json(state_path, state)
    print("COMPLETE " + json.dumps(state), flush=True)


if __name__ == "__main__":
    main()
