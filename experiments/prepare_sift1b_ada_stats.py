"""Prepare Ada-IVF's reference-order squared-norm statistics from an IVF cache.

Example (run from the repository root)::

    python experiments/prepare_sift1b_ada_stats.py --base /data/base.bin \
        --cache /data/sift1b_cache

This reads the complete cache, verifies every posting ID against its assignment,
and preserves the reference float32 reduction over each whole partition. It
does not run an experiment or estimate any measurements. Smaller generated
caches are useful for correctness checks, not paper results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from experiments.build_sift1b_scale import binary_matrix, sample_fingerprint


DEFINITION = ("float(np.sum(x*x)) with whole stable-order partition x float32, "
              "matching repository AdaIVF initialization")
INPUTS = {"build_state_sha256": "build_state.json",
          "counts_sha256": "counts.npy", "ids_sha256": "ids.npy",
          "assign_sha256": "assign.npy", "postings_state_sha256": "postings_state.json"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    """Write metadata atomically without sharing a fixed temporary filename."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as handle:
            tmp = Path(handle.name)
            json.dump(value, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def validate_cache(base_path, cache):
    build = json.loads((cache / "build_state.json").read_text())
    config = build["config"]
    n, nlist, dim = (int(config[k]) for k in ("n", "nlist", "dim"))
    if not (build.get("complete") is True and build.get("completed_rows") == n
            and n > 0 and nlist > 0 and dim > 0):
        raise RuntimeError("Build cache is incomplete or has invalid dimensions")
    if sample_fingerprint(base_path) != config["base"]:
        raise RuntimeError("Base corpus fingerprint differs from the completed build")
    base = binary_matrix(base_path, np.uint8)
    if len(base) < n or base.shape[1] != dim:
        raise RuntimeError("Base corpus dimensions differ from the completed build")
    base = base[:n]
    postings = json.loads((cache / "postings_state.json").read_text())
    if postings.get("complete") is not True or postings.get("n") != n:
        raise RuntimeError("Posting cache is incomplete or has the wrong corpus size")
    counts = np.load(cache / "counts.npy", allow_pickle=False)
    ids = np.load(cache / "ids.npy", mmap_mode="r", allow_pickle=False)
    assignments = np.load(cache / "assign.npy", mmap_mode="r", allow_pickle=False)
    if (counts.shape != (nlist,) or counts.dtype.kind not in "iu"
            or np.any(counts < 0) or sum(map(int, counts)) != n):
        raise RuntimeError("Partition counts have invalid shape, values, or total")
    if ids.shape != (n,) or ids.dtype != np.dtype("uint32"):
        raise RuntimeError("Posting IDs must be a uint32 array of the corpus size")
    if assignments.shape != (n,) or assignments.dtype != np.dtype("int32"):
        raise RuntimeError("Assignments must be an int32 array of the corpus size")
    bounds = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    for c in range(nlist):
        part = ids[bounds[c]:bounds[c+1]]
        if not len(part):
            continue
        # Strict order forbids duplicates within a partition. Assignment matches
        # forbid duplicates across partitions. Together with n valid occurrences
        # this proves that every original ID occurs exactly once, without a bitmap.
        if int(part[-1]) >= n or np.any(part[1:] <= part[:-1]):
            raise RuntimeError(f"Partition {c}: IDs are out of range or not strictly ordered")
        if not np.all(assignments[part] == c):
            raise RuntimeError(f"Partition {c}: posting IDs disagree with assignments")
    hashes = {key: digest(cache / name) for key, name in INPUTS.items()}
    return base, ids, counts, bounds, config, hashes


def validate_existing(values_path, meta, n, nlist, hashes):
    required = ("build_state_sha256", "counts_sha256", "ids_sha256")
    if (meta.get("n") != n or meta.get("nlist") != nlist
            or meta.get("definition") != DEFINITION
            or any(meta.get(key) != hashes[key] for key in required)
            or any(key in meta and meta[key] != value for key, value in hashes.items())):
        raise RuntimeError("Existing complete statistics have mismatched provenance; refusing overwrite")
    if not values_path.exists() or meta.get("sha256") != digest(values_path):
        raise RuntimeError("Existing complete statistics are missing or have a hash mismatch")
    values = np.load(values_path, allow_pickle=False)
    if (values.shape != (nlist,) or values.dtype != np.dtype("float64")
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise RuntimeError("Existing complete statistics have invalid values")
    return values


def prepare(base_path, cache):
    cache = Path(cache)
    if not cache.is_dir():
        raise RuntimeError("Cache directory does not exist")
    values_path = cache / "squared_norm_sums_reference.npy"
    meta_path = cache / "squared_norm_sums_reference.json"
    lock_path = cache / "squared_norm_sums_reference.lock"
    try:
        lock = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"Preparation lock exists: {lock_path}; check its recorded process before removing a stale lock") from exc
    tmp = None
    start = time.perf_counter()
    try:
        with os.fdopen(lock, "w") as handle:
            json.dump(dict(pid=os.getpid(), host=platform.node(), started_utc=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())), handle)
        base, ids, counts, bounds, config, hashes = validate_cache(base_path, cache)
        n, nlist = len(base), len(counts)
        existing = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if existing and existing.get("complete") is True:
            validate_existing(values_path, existing, n, nlist, hashes)
            print("VERIFIED REUSE", values_path, "elapsed_s", time.perf_counter()-start, flush=True)
            return existing
        if existing:
            if (existing.get("n") != n or existing.get("nlist") != nlist
                    or any(existing.get(key) != value for key, value in hashes.items())):
                raise RuntimeError("Incomplete preparation has different provenance; refusing overwrite")
        elif values_path.exists():
            raise RuntimeError("Statistics file exists without metadata; refusing overwrite")
        meta = dict(complete=False, n=n, nlist=nlist, definition=DEFINITION,
                    source_sha256=digest(Path(__file__)), source_file=Path(__file__).name,
                    base_fingerprint=config["base"], numpy=np.__version__, python=sys.version,
                    validation="All n IDs occur exactly once in their assigned partition, in strictly increasing order",
                    maximum_partition_rows=int(counts.max()),
                    started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **hashes)
        atomic_json(meta_path, meta)
        q = np.zeros(nlist, dtype=np.float64)
        calculation_start = time.perf_counter()
        with np.errstate(over="raise", invalid="raise"):
            for c in range(nlist):
                x = base[ids[bounds[c]:bounds[c+1]]].astype(np.float32)
                # Deliberately not chunked or reduced in float64: both would
                # change the reference initialization's rounding and thresholds.
                q[c] = float(np.sum(x*x))
                del x
                if (c+1) % 128 == 0 or c+1 == nlist:
                    print("QNORMS", c+1, nlist, "elapsed_s", time.perf_counter()-start, flush=True)
        if not np.isfinite(q).all():
            raise RuntimeError("Non-finite squared-norm statistics")
        calculation_s = time.perf_counter()-calculation_start
        if sample_fingerprint(base_path) != config["base"]:
            raise RuntimeError("Base corpus changed during preparation")
        if any(digest(cache / name) != hashes[key] for key, name in INPUTS.items()):
            raise RuntimeError("Input cache changed during preparation")
        with tempfile.NamedTemporaryFile(dir=cache, prefix=values_path.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            tmp = Path(handle.name)
            np.save(handle, q, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        value_hash = digest(tmp)
        tmp.replace(values_path)
        meta.update(complete=True, sha256=value_hash, min=float(q.min()), max=float(q.max()),
                    calculation_s=calculation_s, elapsed_s=time.perf_counter()-start,
                    finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        atomic_json(meta_path, meta)
        print("COMPLETE", values_path, "elapsed_s", meta["elapsed_s"], flush=True)
        return meta
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Original BIGANN uint8 binary corpus")
    parser.add_argument("--cache", required=True, help="Completed build and stable posting cache")
    args = parser.parse_args()
    prepare(args.base, args.cache)


if __name__ == "__main__":
    main()
