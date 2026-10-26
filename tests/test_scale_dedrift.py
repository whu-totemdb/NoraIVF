"""Operation-by-operation fidelity checks; test inputs are not measurements."""
import os
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.getcwd())

import faiss
import numpy as np

from experiments.run_claim_evidence_annbench import L2DeDriftIVF
from noraivf.lean import IntMap
from noraivf.scale_dedrift import CompactDeDrift, initialize_dedrift
from noraivf.scale_storage import ArrayPosting, RawVectors, search


def compact_from_reference(ref, data, extra=256):
    idx = CompactDeDrift.__new__(CompactDeDrift)
    for name in ("d", "n", "nlist", "k", "seed", "next_id"):
        setattr(idx, name, getattr(ref, name))
    idx.rng = np.random.RandomState(ref.seed)
    idx.rng.set_state(ref.rng.get_state())
    idx.centroids = ref.centroids.copy()
    idx.assignments = ref.assignments.copy()
    idx.data = None
    idx.id_to_vec = RawVectors(data, len(data) + extra)
    idx.posting_lists = [ArrayPosting(np.asarray(p, dtype=np.uint32)) for p in ref.posting_lists]
    for name in ("total_insert_time", "total_delete_time", "total_split_time", "total_merge_time"):
        setattr(idx, name, 0.0)
    for name in ("n_inserts", "n_deletes", "n_splits", "n_merges", "n_reassigns"):
        setattr(idx, name, 0)
    idx.split_history = []
    pointer = IntMap(len(data) + extra)
    pointer._a[:len(data)] = ref.assignments
    return initialize_dedrift(idx, np.array([len(p) for p in ref.posting_lists]),
                              np.array(ref.cluster_sums), pointer,
                              k2=ref.k2, rebalance_every=ref.rebalance_every)


def equivalent(ref, compact):
    assert ref.k == compact.k and ref.n == compact.n == len(compact.id_to_vec)
    assert [list(p) for p in ref.posting_lists] == [list(p) for p in compact.posting_lists]
    assert np.array_equal(ref.centroids, compact.centroids)
    assert np.array_equal(ref.cluster_sums, compact.cluster_sums)
    for name in ("n_inserts", "n_deletes", "n_splits", "n_merges", "n_reassigns", "ops_since_rebalance"):
        assert getattr(ref, name) == getattr(compact, name), name
    ref_rng, compact_rng = ref.rng.get_state(), compact.rng.get_state()
    assert ref_rng[0] == compact_rng[0] and np.array_equal(ref_rng[1], compact_rng[1])
    assert ref_rng[2:] == compact_rng[2:]
    expected = np.full(compact.next_id, -1, dtype=np.int32)
    seen = set()
    for c, posting in enumerate(compact.posting_lists):
        for i in posting:
            assert i not in seen and i in compact.id_to_vec and i in ref.id_to_vec
            assert np.array_equal(compact.id_to_vec[i], ref.id_to_vec[i])
            seen.add(i)
            expected[i] = c
    assert seen == set(ref.id_to_vec) and len(seen) == ref.n
    assert np.array_equal(compact.partition_map._a[:compact.next_id], expected)


def run_fidelity(data):
    ref = L2DeDriftIVF(data.astype(np.float32), nlist=16, seed=42)
    compact = compact_from_reference(ref, data)
    equivalent(ref, compact)
    # Each deletion must invoke the native rebalancer, including after its
    # prior write-back changed the deleted ID's partition.
    for i in (7, 113):
        ref.delete(i)
        compact.delete(i)
        equivalent(ref, compact)
    assert len(compact.rebalance_events) == 2
    # The unchanged insertion interval must also trigger naturally at 100.
    for j in range(100):
        v = data[(j * 19 + 31) % len(data)].astype(np.float32)
        assert ref.insert(v) == compact.insert(v)
        equivalent(ref, compact)
    assert len(compact.rebalance_events) == 3
    assert [e["trigger"] for e in compact.rebalance_events] == ["delete", "delete", "insert_interval"]
    assert all(e["rows_reclustered"] == sum(e["selected_sizes_before"])
               == sum(e["selected_sizes_after"]) for e in compact.rebalance_events)
    assert all(e["n_clusters_before"] == e["n_clusters_after"] == 16
               for e in compact.rebalance_events)
    # Shared query scan must retain the same top-k distance ranks (ties may
    # select different IDs in the original reference's heap implementation).
    for q in data[:3].astype(np.float32):
        _, dist, _ = search(compact, q, k=10, nprobe=4)
        reference = ref.search(q, k=10, nprobe=4)
        reference_dist = [np.sum((ref.id_to_vec[i] - q) ** 2) for i in reference]
        assert np.array_equal(dist, reference_dist)
    return dict(pass_=True, source="test inputs only", operations=102,
                native_rebalances=len(compact.rebalance_events),
                triggers=[e["trigger"] for e in compact.rebalance_events],
                rows_reclustered=[e["rows_reclustered"] for e in compact.rebalance_events])


def test_generated_fidelity():
    data = np.random.default_rng(7).integers(0, 256, size=(2000, 16), dtype=np.uint8)
    run_fidelity(data)


if __name__ == "__main__":
    faiss.omp_set_num_threads(1)
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            n, d = struct.unpack("<II", f.read(8))
        data = np.memmap(sys.argv[1], dtype=np.uint8, offset=8, mode="r", shape=(n, d))[:10000]
        label = "real BIGANN first 10000 rows, fidelity validation only"
    else:
        data = np.random.default_rng(7).integers(0, 256, size=(2000, 16), dtype=np.uint8)
        label = "generated unit-test inputs, not an experiment"
    print(label, run_fidelity(data), flush=True)
