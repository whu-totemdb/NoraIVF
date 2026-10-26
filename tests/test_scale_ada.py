"""Small fidelity checks; these validation inputs are never paper results.

Optional real data: python tests/test_scale_ada.py --sift-base /path/base.bin
The file must have a little-endian uint32 (N, 128) header followed by uint8
rows. Stable-distance equivalence and original-float32-reference fidelity are
separate tests: original-reference divergence must not be relabeled a pass.
"""
import argparse
import copy
from decimal import Decimal, localcontext
import hashlib
import json
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np

from experiments.run_claim_evidence_annbench import L2AdaIVF
from noraivf.lean import IntMap
from noraivf.scale_ada import CompactAdaIVF, UnsupportedGlobalRebuild, initialize_ada
from noraivf.scale_storage import ArrayPosting, RawVectors, search


SIFT_CONFIG = None


class StableL2AdaIVF(L2AdaIVF):
    """Same algorithm, direct float64 maintenance distances as its oracle.

    Deliberately independent of the adapter's fast norm/dot implementation.
    Constructor/splits, triggers, updates, queries, and sums are unchanged.
    """

    def _similarity_matrix(self, a, b):
        a64 = np.asarray(a, dtype=np.float64)
        result = np.empty((len(a), len(b)), dtype=np.float64)
        for c, center in enumerate(np.asarray(b, dtype=np.float64)):
            result[:, c] = -np.sum((a64 - center) ** 2, axis=1)
        return result


def real_sift_subset(path, rows, seed=42):
    """Sample one actual row per stratum, using only O(rows * 128) RAM."""
    path = Path(path)
    header = np.fromfile(path, dtype="<u4", count=2)
    if len(header) != 2:
        raise ValueError("SIFT base file has no complete (N, D) header")
    n, d = map(int, header)
    if d != 128 or path.stat().st_size != 8 + n * d:
        raise ValueError("Expected exact uint8 base.bin payload with D=128")
    if not 128 <= rows <= n:
        raise ValueError("--sift-rows must be between 128 and corpus N")
    edges = np.linspace(0, n, rows + 1, dtype=np.int64)
    ids = edges[:-1] + np.random.default_rng(seed).integers(0, np.diff(edges))
    corpus = np.memmap(path, mode="r", dtype=np.uint8, offset=8, shape=(n, d))
    data = np.asarray(corpus[ids]).copy()
    metadata = dict(source=str(path.resolve()), source_rows=n, dimension=d,
                    sampled_rows=rows, seed=seed, sampler="one-per-equal-stratum",
                    sample_ids_sha256=hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
                    sample_data_sha256=hashlib.sha256(data.tobytes()).hexdigest())
    return data, metadata


def compact_from_reference(ref, data, batch=257):
    idx = CompactAdaIVF.__new__(CompactAdaIVF)
    idx.__dict__.update(copy.deepcopy(ref.__dict__))
    idx.id_to_vec = RawVectors(data, len(data) + 4096)
    idx.data = None
    idx.posting_lists = [ArrayPosting(np.array(p, dtype=np.uint32))
                         for p in ref.posting_lists]
    pointer = IntMap(len(data) + 4096)
    for c, posting in enumerate(idx.posting_lists):
        pointer._a[posting.a] = c
    counts = np.array([len(p) for p in ref.posting_lists])
    initialize_ada(idx, counts, ref.a_c, pointer, ref.q_c)
    idx.scan_batch = batch
    return idx


def assert_equivalent(test, ref, compact):
    test.assertEqual(len(ref.posting_lists), len(compact.posting_lists))
    ref_postings = [np.asarray(p, dtype=np.uint32) for p in ref.posting_lists]
    for c, (expected, actual) in enumerate(zip(ref_postings, compact.posting_lists)):
        np.testing.assert_array_equal(expected, actual.a[:actual.n], err_msg=f"posting {c}")
    for attr in ("k", "n", "next_id", "n_inserts", "n_deletes", "n_splits",
                 "n_merges", "n_reassigns", "ops_since_check", "n0", "sigma0", "eps0"):
        test.assertEqual(getattr(ref, attr), getattr(compact, attr), attr)
    for attr in ("centroids", "a_c", "q_c", "T", "mu0"):
        np.testing.assert_array_equal(getattr(ref, attr), getattr(compact, attr),
                                      err_msg=attr)
    test.assertEqual(ref.ada_stats, compact.ada_stats)
    np.testing.assert_array_equal(ref._detect_violators(), compact._detect_violators())
    test.assertEqual(ref._global_indicator(), compact._global_indicator())
    # Independently check every live ID and every dense-PAM entry, including
    # released/empty slots and deleted IDs. Matching summary counts alone
    # would miss a dropped ID, duplicate posting, or stale PAM entry.
    ids = np.concatenate(ref_postings)
    test.assertEqual(len(ids), ref.n)
    test.assertEqual(len(np.unique(ids)), len(ids), "duplicate posting IDs")
    test.assertEqual(len(ref.id_to_vec), ref.n)
    test.assertEqual(len(compact.id_to_vec), compact.n)
    np.testing.assert_array_equal(np.sort(ids), np.sort(list(ref.id_to_vec)))
    np.testing.assert_array_equal(np.sort(ids), np.sort(list(ref.pam)))
    expected_pam = np.full_like(compact.pam._a, -1)
    for c, posting in enumerate(ref_postings):
        expected_pam[posting] = c
        np.testing.assert_array_equal([ref.pam[int(i)] for i in posting],
                                      np.full(len(posting), c), err_msg=f"reference PAM {c}")
    np.testing.assert_array_equal(expected_pam, compact.pam._a, err_msg="entire dense PAM")
    test.assertEqual(int(np.count_nonzero(compact.pam._a >= 0)), compact.n)
    expected_vectors = np.asarray([ref.id_to_vec[int(i)] for i in ids], dtype=np.float32)
    np.testing.assert_array_equal(expected_vectors, compact.id_to_vec.gather(ids))
    a, b = ref.rng.get_state(), compact.rng.get_state()
    test.assertEqual(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    test.assertEqual(a[2:], b[2:])


class CompactAdaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        faiss.omp_set_num_threads(1)

    def test_public_operations_temperature_and_native_reindex(self):
        data = np.random.default_rng(17).integers(0, 256, (2400, 16), dtype=np.uint8)
        ref = StableL2AdaIVF(data.astype(np.float32), 16, seed=42)
        compact = compact_from_reference(ref, data)
        assert_equivalent(self, ref, compact)
        self.assertEqual((compact.tau_f, compact.r_c, compact.iota,
                          compact.check_every, compact.tau_g), (.75, 25, 5, 100, 5.0))
        # Real query-temperature hooks, followed by ordinary update checks;
        # maintenance is triggered by the unmodified detector, not forced.
        for _ in range(35):
            ref.search(data[0], k=10, nprobe=4)
            compact.observe_query(data[0], nprobe=4)
        np.testing.assert_array_equal(ref.T, compact.T)
        for i in range(60):
            ref.delete(i)
            compact.delete(i)
            assert_equivalent(self, ref, compact)
            self.assertEqual(ref.insert(data[i]), compact.insert(data[i]))
            assert_equivalent(self, ref, compact)
            self.assertIsNone(compact.pam.get(i))
        self.assertGreater(ref.ada_stats["n_local_reindex"], 0)
        self.assertGreater(ref.n_splits, 0)
        self.assertEqual(ref.ada_stats["n_global_rebuild"], 0)
        # The common evaluation scan must not heat the index.
        temperatures = compact.T.copy()
        search(compact, data[3].astype(np.float32), k=10, nprobe=4)
        np.testing.assert_array_equal(temperatures, compact.T)

    def test_chunked_accumulation_past_float32_integer_precision(self):
        # A single cluster's coordinate sum exceeds 2**24, so accumulating
        # per-batch sums instead of reference-ordered add.at would diverge.
        data = np.random.default_rng(8).integers(0, 256, (140000, 4), dtype=np.uint8)
        ref = StableL2AdaIVF(data.astype(np.float32), 1, seed=42)
        compact = compact_from_reference(ref, data, batch=4093)
        ref._local_reindex([0])
        compact._local_reindex([0])
        assert_equivalent(self, ref, compact)
        self.assertGreater(float(ref.a_c.max()), 2**24)

    def test_balancing_keeps_reference_order_for_equal_scores(self):
        data = np.full((16, 4), 100, dtype=np.uint8)
        ref = StableL2AdaIVF(data.astype(np.float32), 1, seed=42)
        compact = compact_from_reference(ref, data, batch=257)
        ids = np.array([7, 1, 4, 0, 6, 3, 2, 5], dtype=np.uint32)
        centroids = np.full((2, 4), 100, dtype=np.float32)
        assigns = np.ones(8, dtype=np.int32)
        expected = ref._balance(assigns.copy(), data[ids].astype(np.float32), centroids, 4)
        got = compact._balance_ids(assigns.copy(), ids, centroids, 4)
        np.testing.assert_array_equal(expected, got)

    def test_near_ties_match_high_precision_direct_oracle_for_many_chunks(self):
        idx = CompactAdaIVF.__new__(CompactAdaIVF)
        idx.d = 128
        centroids = np.array([[100.002] * 128, [100.001] * 128,
                              [100.003] * 128], dtype=np.float32)
        cases = [(1003, b) for b in (1, 2, 3, 7, 31, 37, 63, 127, 128, 129, 257, 8192)]
        cases.extend([(10003, 257), (8193, 8192)])
        for size, batch in cases:
            with self.subTest(size=size, batch=batch):
                data = np.full((size, 128), 100, dtype=np.uint8)
                idx.id_to_vec = RawVectors(data, size)
                idx.scan_batch = batch
                # Decimal supplies an independent high-precision oracle even
                # on systems where np.longdouble is only binary64.
                with localcontext() as ctx:
                    ctx.prec = 80
                    direct = [sum((Decimal(int(x)) - Decimal.from_float(float(y))) ** 2
                                  for x, y in zip(data[0], center)) for center in centroids]
                    nearest = min(range(len(direct)), key=direct.__getitem__)
                reference = np.full(size, nearest)
                compact = idx._assign_ids(np.arange(size, dtype=np.uint32), centroids)
                np.testing.assert_array_equal(reference, compact)

    def test_random_and_exact_ties_match_direct_oracle(self):
        rng = np.random.default_rng(191)
        for dimension in (1, 16, 128, 257):
            data = rng.integers(0, 256, (513, dimension), dtype=np.uint8)
            centroids = rng.uniform(0, 255, (17, dimension)).astype(np.float32)
            centroids[1] = centroids[0]  # argmin must preserve first-center ties
            direct = np.empty((len(data), len(centroids)), dtype=np.longdouble)
            for c, center in enumerate(centroids.astype(np.longdouble)):
                direct[:, c] = np.sum((data.astype(np.longdouble) - center) ** 2, axis=1)
            reference = np.argmin(direct, axis=1)
            for batch in (1, 17, 128, 257, 8192):
                with self.subTest(dimension=dimension, batch=batch):
                    idx = CompactAdaIVF.__new__(CompactAdaIVF)
                    idx.d, idx.scan_batch = dimension, batch
                    idx.id_to_vec = RawVectors(data, len(data))
                    got = idx._assign_ids(np.arange(len(data), dtype=np.uint32), centroids)
                    np.testing.assert_array_equal(reference, got)

    def _real_sift_check(self, reference_class):
        if SIFT_CONFIG is None or SIFT_CONFIG.sift_base is None:
            self.skipTest("Provide --sift-base for actual 128D SIFT subset validation")
        label = "original" if reference_class is L2AdaIVF else "stable"
        if SIFT_CONFIG.sift_reference not in ("both", label):
            self.skipTest(f"--sift-reference excludes {label}")
        data, metadata = real_sift_subset(SIFT_CONFIG.sift_base, SIFT_CONFIG.sift_rows)
        print("SIFT_VALIDATION_START " + json.dumps(dict(metadata, reference=label),
                                                   sort_keys=True), flush=True)
        nlist = min(32, len(data) // 64)
        ref = reference_class(data.astype(np.float32), nlist, seed=42)
        compact = compact_from_reference(ref, data)
        assert_equivalent(self, ref, compact)
        # Query a row from the most size-imbalanced initial partition. Repeated
        # genuine query calls heat it; the unchanged 100-op detector triggers
        # maintenance during ordinary public updates, never by a forced call.
        sizes = np.asarray([len(p) for p in ref.posting_lists])
        hot = int(np.argmax(np.abs(sizes - ref.tau_s)))
        query = data[ref.posting_lists[hot][0]]
        for _ in range(80):
            ref.search(query, k=10, nprobe=4)
            compact.observe_query(query, nprobe=4)
        assert_equivalent(self, ref, compact)
        start = time.perf_counter()
        for i in range(60):
            ref.delete(i)
            compact.delete(i)
            assert_equivalent(self, ref, compact)
            self.assertEqual(ref.insert(data[i]), compact.insert(data[i]))
            assert_equivalent(self, ref, compact)
        self.assertGreater(ref.ada_stats["n_local_reindex"], 0)
        self.assertEqual(ref.ada_stats["n_global_rebuild"], 0)
        metadata.update(reference=label, result="strict-all-state-pass", nlist=nlist,
                        public_updates=120, operational_queries=80,
                        local_reindexes=ref.ada_stats["n_local_reindex"],
                        pool_rows=[e["rows"] for e in compact.scale_events],
                        n_splits=ref.n_splits, n_reassigns=ref.n_reassigns,
                        numeric_kernel=compact.scale_numeric_kernel,
                        elapsed_update_validation_s=time.perf_counter() - start)
        print("SIFT_VALIDATION " + json.dumps(metadata, sort_keys=True), flush=True)

    def test_real_sift_subset_stable_reference(self):
        self._real_sift_check(StableL2AdaIVF)

    def test_real_sift_subset_original_reference(self):
        self._real_sift_check(L2AdaIVF)

    def test_global_trigger_stops_explicitly(self):
        data = np.random.default_rng(9).integers(0, 256, (400, 8), dtype=np.uint8)
        ref = L2AdaIVF(data.astype(np.float32), 4, seed=42)
        compact = compact_from_reference(ref, data)
        compact.eps0 = 1e-20
        self.assertGreater(compact._global_indicator(), compact.tau_g)
        with self.assertRaisesRegex(UnsupportedGlobalRebuild, "global indicator exceeded"):
            compact._run_checks()
        self.assertEqual(compact.ada_stats["n_local_reindex"], 0)
        self.assertEqual(compact.ada_stats["n_global_rebuild"], 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--sift-base", type=Path)
    parser.add_argument("--sift-rows", type=int, default=8192)
    parser.add_argument("--sift-reference", choices=("both", "stable", "original"), default="both")
    SIFT_CONFIG, remaining = parser.parse_known_args()
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
