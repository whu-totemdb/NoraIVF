"""Fidelity checks for compact storage; generated data is only test data."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, os.getcwd())
import faiss
import numpy as np
from experiments.run_claim_evidence_annbench import L2NovaIVF, L2SPFreshIVF
from noraivf.scale_storage import ArrayPosting, BatchedSPFresh, make_index, search


def equivalent(ref, compact):
    assert ref.k == compact.k
    assert ref.n == compact.n
    assert [list(p) for p in ref.posting_lists] == [list(p) for p in compact.posting_lists]
    assert np.array_equal(ref.centroids, compact.centroids), float(np.max(np.abs(ref.centroids-compact.centroids)))
    for attr in ("n_splits", "n_merges", "n_reassigns", "n_inserts", "n_deletes"):
        assert getattr(ref, attr) == getattr(compact, attr), attr


def run_fidelity(data, nlist=16):
    report = {}
    for method, cls in [("NoraIVF", L2NovaIVF), ("SPFresh", L2SPFreshIVF)]:
        ref = cls(data.astype(np.float32), nlist, seed=42)
        order = np.argsort(ref.assignments, kind="stable").astype(np.uint32)
        counts = np.bincount(ref.assignments, minlength=nlist)
        bounds = np.concatenate(([0], np.cumsum(counts)))
        sums = np.array([data[order[bounds[c]:bounds[c+1]]].astype(np.float32).sum(axis=0)
                         for c in range(nlist)])
        compact = make_index(method, data, ref.centroids, ref.assignments,
                             order, counts, sums)
        equivalent(ref, compact)
        split = "_split" if method == "NoraIVF" else "_lire_split"
        for j in np.argsort(-counts,kind="stable")[:3]:
            getattr(ref, split)(int(j))
            getattr(compact, split)(int(j))
            equivalent(ref, compact)
        # Matched deletions/reinsertions exercise PPM updates after the split.
        for i in [int(x) for x in np.arange(20)*7]:
            ref.delete(i)
            compact.delete(i)
            assert ref.insert(data[i]) == compact.insert(data[i])
        equivalent(ref, compact)
        # Force an actual merge through the public deletion path, then check
        # ordered postings and every live pointer after the remapping.
        before = ref.n_merges
        c = next(c for c,p in enumerate(ref.posting_lists) if len(p) > ref.T_merge+2)
        for i in list(ref.posting_lists[c]):
            ref.delete(i)
            compact.delete(i)
            if ref.n_merges > before:
                break
        assert ref.n_merges == before+1
        equivalent(ref, compact)
        if method == "NoraIVF":
            for c, posting in enumerate(compact.posting_lists):
                for off, i in enumerate(posting):
                    assert compact.ppm[i] == c
                    assert compact.positions[i] == off
        else:
            for c,posting in enumerate(compact.posting_lists):
                for i in posting:
                    assert compact.partition_map[i] == c
        for q in data[:4].astype(np.float32):
            got, dist, _ = search(compact, q, k=10, nprobe=4)
            reference = ref.search(q, k=10, nprobe=4)
            # Integer vectors may have exact distance ties at rank k.
            ref_dist = [np.sum((ref.id_to_vec[i]-q)**2) for i in reference]
            assert np.array_equal(dist, ref_dist)
        report[method] = dict(pass_=True, k=compact.k, splits=compact.n_splits,
                              merges=compact.n_merges, reassignments=compact.n_reassigns)
    return report


def test_copy_on_write():
    original = np.array([3, 8, 11], dtype=np.uint32)
    p = ArrayPosting(original)
    p[0] = 7
    p.append(13)
    p.extend([15, 16])
    p.remove(11)
    assert p.pop() == 16
    assert list(p) == [7, 8, 13, 15]
    assert original.tolist() == [3, 8, 11]


def test_repair_cancellation_guard():
    # Norm-expansion BLAS can collapse both tiny distances to zero and pick
    # the wrong centroid. The direct reference expression chooses centroid 1.
    idx = BatchedSPFresh.__new__(BatchedSPFresh)
    idx.d = 128
    idx.k = 2
    idx.centroids = np.array([[100.002]*128,[100.001]*128],dtype=np.float32)
    x = np.full((32,128),100,dtype=np.float32)
    quant = faiss.IndexFlatL2(128)
    quant.add(idx.centroids)
    labels,fallbacks,checks = idx._repair_labels(x,quant,float(np.sum(idx.centroids.astype(np.float64)**2,axis=1).max()))
    assert np.all(labels == 1)
    assert fallbacks == 32 and checks == 5


if __name__ == "__main__":
    faiss.omp_set_num_threads(1)
    test_copy_on_write()
    test_repair_cancellation_guard()
    if len(sys.argv) > 1:
        import struct
        with open(sys.argv[1], "rb") as f:
            n, d = struct.unpack("<II", f.read(8))
        data = np.memmap(sys.argv[1], dtype=np.uint8, offset=8, mode="r", shape=(n, d))[:10000]
        label = "real BIGANN first 10000 rows"
    else:
        data = np.random.default_rng(7).integers(0, 256, size=(2000, 16), dtype=np.uint8)
        label = "generated unit-test inputs, not an experiment"
    print(label, run_fidelity(data), flush=True)
