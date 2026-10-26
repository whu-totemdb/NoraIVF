"""Compact, chunked execution of the existing L2 AdaIVF maintenance policy.

This adapts containers and local-reindex working storage, not trigger policy.
Maintenance distances deliberately use stable float64 squared L2 instead of
the reference's cancellation-prone float32 norm/dot expression. This numerical
change is tested separately from typical-data fidelity to the original code;
it does not claim universal bitwise equivalence with that original expression.
In particular, the global-rebuild safety valve remains active and raises an
explicit error: this adapter does not claim to implement a billion-row rebuild.
"""
from __future__ import annotations

import time
import numpy as np

from experiments.run_claim_evidence_annbench import L2AdaIVF, squared_l2_to_centroids
from noraivf.core import kmeans_faiss
from noraivf.lean import IntMap
from noraivf.scale_storage import ArrayPosting, RawVectors, posting_array


class UnsupportedGlobalRebuild(RuntimeError):
    """Ada's unchanged global trigger fired; the measurement must stop."""


class _DensePAM(IntMap):
    """IntMap-compatible view with the reference Ada deletion API."""

    def __init__(self, pointer):
        self._a = pointer._a
        self._fill = pointer._fill

    def pop(self, key, default=None):
        value = self.get(key, default)
        if key in self:
            del self[key]
        return value


def direct_squared_l2(a, b):
    """Float64 direct squared differences, without an N x C x D temporary.

    Each row/center pair uses the same contiguous coordinate reduction,
    independent of BLAS implementation and the number of other rows. The
    caller bounds N x C output and N x D working storage by row chunking.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    distances = np.empty((len(a), len(b)), dtype=np.float64)
    for c in range(len(b)):
        diff = a - b[c]
        np.square(diff, out=diff)
        distances[:, c] = np.sum(diff, axis=1)
    return distances


def stable_l2_argmin(a, b, diagnostics=None):
    """Fast float64 GEMM assignment, with direct evaluation of near ties.

    Norm/dot is used only to certify separated winners. A deliberately
    conservative forward-error band, based on gamma_(2d+10) and squared
    norms, includes uncertainty in the dot, norms, subtraction, and direct
    coordinate reduction. Candidates in the band are evaluated by direct
    float64 squared differences; argmin retains the first center on ties.
    This targets finite float32 SIFT vectors/centers, not arbitrary overflow
    or non-finite inputs. No float32 GEMM or batch-tail heuristic is used.
    An optional diagnostics dict accumulates calls, rows, ambiguous rows,
    and time in the direct fallback without changing the distance arithmetic.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not len(b):
        raise ValueError("Squared-L2 assignment requires at least one center")
    if diagnostics is not None:
        diagnostics["calls"] = diagnostics.get("calls", 0) + 1
        diagnostics["rows"] = diagnostics.get("rows", 0) + len(a)
        diagnostics.setdefault("ambiguous_rows", 0)
        diagnostics.setdefault("fallback_wall_s", 0.0)
    a2 = np.sum(a * a, axis=1)
    b2 = np.sum(b * b, axis=1)
    distances = a @ b.T
    distances *= -2.0
    distances += a2[:, None]
    distances += b2[None, :]
    winners = np.argmin(distances, axis=1)
    if len(b) == 1 or not len(a):
        return winners.astype(np.int32)
    rows = np.arange(len(a))
    best = distances[rows, winners].copy()
    distances[rows, winners] = np.inf
    eps = np.finfo(np.float64).eps
    terms = 2 * a.shape[1] + 10
    gamma = terms * eps / (1.0 - terms * eps)
    # ||a||^2 + max ||b||^2 bounds every candidate's norm/dot error.
    error_band = (32.0 * gamma) * (a2 + np.max(b2))
    ambiguous = np.flatnonzero(np.min(distances, axis=1) - best <= error_band)
    distances[rows, winners] = best
    if diagnostics is not None and len(ambiguous):
        fallback_started = time.perf_counter()
    for row in ambiguous:
        candidates = np.flatnonzero(distances[row] <= best[row] + error_band[row])
        diff = b[candidates] - a[row]
        np.square(diff, out=diff)
        exact = np.sum(diff, axis=1)
        winners[row] = candidates[np.argmin(exact)]
    if diagnostics is not None and len(ambiguous):
        diagnostics["ambiguous_rows"] += len(ambiguous)
        diagnostics["fallback_wall_s"] += time.perf_counter() - fallback_started
    return winners.astype(np.int32)


class CompactAdaIVF(L2AdaIVF):
    """L2AdaIVF with array postings and bounded local Lloyd working memory.

    Construct common index state without calling the corpus-copying reference
    constructor, then call ``initialize_ada``. Ordered postings must be disjoint
    and PAM must identify every live posting ID, as in the reference index.
    ``scale_progress`` may be set to a callable accepting keyword event fields.
    """

    scan_batch = 8192
    similarity_bytes = 32 * 1024**2
    max_split_training_bytes = 2 * 1024**3
    scale_numeric_kernel = "float64-norm-dot/direct-near-ties-v1"

    def _progress(self, event, **fields):
        hook = getattr(self, "scale_progress", None)
        if hook is not None:
            hook(event=event, **fields)

    def _batch_rows(self, n_centroids):
        # Pairwise float64 output plus conversions/direct-difference scratch.
        bytes_per_row = 8 * max(1, n_centroids) + 24 * self.d + 64
        return max(1, min(self.scan_batch,
                          self.similarity_bytes // bytes_per_row))

    def _row_slices(self, size, n_centroids):
        rows = self._batch_rows(n_centroids)
        for start in range(0, size, rows):
            yield start, min(start + rows, size)

    def _similarity_matrix(self, a, b):
        # Candidate KNN and balancing need complete, reproducible rankings.
        # Their direct distances also define the numerical reference for the
        # fast Lloyd assignment's certified winner / near-tie fallback.
        distances = direct_squared_l2(a, b)
        np.negative(distances, out=distances)
        return distances

    def _live_posting(self, c):
        ids = posting_array(self.posting_lists[c])
        # All maintenance uses the same PAM, so this is the vectorized version
        # of the reference's membership filter, without a Python object per ID.
        live = self.pam._a[ids] >= 0
        return ids if live.all() else ids[live]

    def _assign_ids(self, ids, centroids):
        if not len(centroids):
            raise RuntimeError("Ada local reindex produced an empty candidate pool")
        assigns = np.empty(len(ids), dtype=np.int32)
        tracked = getattr(self, "scale_progress", None) is not None
        diagnostics = {} if tracked else None
        if tracked:
            phase_started = time.perf_counter()
            next_progress = phase_started + 30.0
            self._progress("ada_assignment", processed_rows=0, total_rows=len(ids),
                           candidates=len(centroids), calls=0, fallback_rows=0,
                           fallback_wall_s=0.0, phase_wall_s=0.0)
        for start, end in self._row_slices(len(ids), len(centroids)):
            vectors = self.id_to_vec.gather(ids[start:end])
            assigns[start:end] = stable_l2_argmin(vectors, centroids, diagnostics)
            if tracked:
                now = time.perf_counter()
                if now >= next_progress or end == len(ids):
                    self._progress("ada_assignment", processed_rows=end, total_rows=len(ids),
                                   candidates=len(centroids), calls=diagnostics["calls"],
                                   fallback_rows=diagnostics["ambiguous_rows"],
                                   fallback_wall_s=diagnostics["fallback_wall_s"],
                                   phase_wall_s=now - phase_started)
                    next_progress = now + 30.0
        return assigns

    def _sums(self, ids, assigns, n_centroids, squared_norms=False):
        sums = np.zeros((n_centroids, self.d), dtype=np.float32)
        norms = np.zeros(n_centroids, dtype=np.float64) if squared_norms else None
        tracked = getattr(self, "scale_progress", None) is not None
        if tracked:
            phase_started = time.perf_counter()
            next_progress = phase_started + 30.0
            self._progress("ada_sums", processed_rows=0, total_rows=len(ids),
                           candidates=n_centroids, squared_norms=squared_norms,
                           phase_wall_s=0.0)
        for start in range(0, len(ids), self.scan_batch):
            end = min(start + self.scan_batch, len(ids))
            vectors = self.id_to_vec.gather(ids[start:end])
            labels = assigns[start:end]
            # Keep the reference point order and one float32 addition per
            # point; summing batches first would change rounding and policy.
            np.add.at(sums, labels, vectors)
            if squared_norms:
                np.add.at(norms, labels, np.sum(vectors * vectors, axis=1))
            if tracked:
                now = time.perf_counter()
                if now >= next_progress or end == len(ids):
                    self._progress("ada_sums", processed_rows=end, total_rows=len(ids),
                                   candidates=n_centroids, squared_norms=squared_norms,
                                   phase_wall_s=now - phase_started)
                    next_progress = now + 30.0
        return sums, norms

    def _lloyd_ids(self, ids, initial):
        centroids = initial.astype(np.float32).copy()
        for iteration in range(self.iota):
            assigns = self._assign_ids(ids, centroids)
            sums, _ = self._sums(ids, assigns, len(centroids))
            counts = np.bincount(assigns, minlength=len(centroids)).astype(np.float64)
            nonempty = counts > 0
            next_centroids = np.zeros_like(centroids)
            next_centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
            next_centroids[~nonempty] = centroids[~nonempty]
            centroids = next_centroids
            self._progress("ada_lloyd", iteration=iteration + 1,
                           rows=len(ids), candidates=len(centroids))
        return centroids, self._assign_ids(ids, centroids)

    def _balance_ids(self, assigns, ids, centroids, min_size):
        counts = np.bincount(assigns, minlength=len(centroids)).astype(np.int64)
        if counts.sum() < min_size * len(centroids):
            return assigns
        for recipient in np.where(counts < min_size)[0]:
            need = int(min_size - counts[recipient])
            while need > 0:
                donors = np.where(counts - min_size > 0)[0]
                if not len(donors):
                    break
                donor = int(donors[np.argmax(counts[donors])])
                give = min(need, int(counts[donor] - min_size))
                donor_points = np.flatnonzero(assigns == donor)
                sims = np.empty(len(donor_points), dtype=np.float64)
                for start, end in self._row_slices(len(donor_points), 1):
                    vectors = self.id_to_vec.gather(ids[donor_points[start:end]])
                    sims[start:end] = self._similarity_matrix(
                        vectors, centroids[recipient:recipient + 1]).ravel()
                # Use the same full one-dimensional argsort as the reference:
                # partial selection would change its deterministic tie order.
                take = donor_points[np.argsort(-sims)[:give]]
                assigns[take] = recipient
                counts[donor] -= give
                counts[recipient] += give
                need -= give
        return assigns

    def _local_reindex(self, violators):
        t0 = time.perf_counter()
        released = set()
        candidates = []
        parts = []
        splits = 0
        for c in map(int, violators):
            ids = self._live_posting(c)
            size = len(ids)
            if not size:
                continue
            released.add(c)
            if size > self.tau_s:
                required = size * self.d * np.dtype(np.float32).itemsize
                if required > self.max_split_training_bytes:
                    raise MemoryError(
                        f"Ada reference split needs {required} float32 bytes for "
                        f"partition {c}; exceeds explicit adapter limit "
                        f"{self.max_split_training_bytes}; measurement stopped")
                k_c = max(2, min(int(np.ceil(size / self.tau_s)), size))
                vectors = self.id_to_vec.gather(ids)
                ctrs, _ = kmeans_faiss(vectors, k_c, niter=10,
                                      seed=self.rng.randint(0, 2**31))
                candidates.extend(ctrs)
                splits += 1
                del vectors
            else:
                candidates.append(self.centroids[c].copy())
            parts.append(ids)

        remaining = [c for c in range(self.k) if c not in released]
        if remaining:
            rem_centroids = self.centroids[remaining]
            pool = np.asarray(candidates, dtype=np.float32)
            neighbors = set()
            for start, end in self._row_slices(len(pool), len(remaining)):
                sims = self._similarity_matrix(pool[start:end], rem_centroids)
                top = np.argsort(-sims, axis=1)[:, :min(self.r_c, len(remaining))]
                for row in top:
                    for r in row:
                        neighbors.add(remaining[r])
            # Preserve the reference's integer-set iteration order exactly.
            for c in neighbors:
                parts.append(self._live_posting(c))
                candidates.append(self.centroids[c].copy())
                released.add(c)

        ids = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint32)
        # A valid PAM/posting state has disjoint slots: reference's defensive
        # dict.fromkeys therefore leaves this ordered array unchanged.
        n_pool = len(ids)
        min_size = max(2, int(0.75 * self.tau_s))
        pool = np.asarray(candidates, dtype=np.float32).reshape(-1, self.d)
        self._progress("ada_pool", rows=n_pool, candidates=len(pool),
                       released=len(released))
        if n_pool:
            max_candidates = max(1, n_pool // min_size)
            if len(pool) > max_candidates:
                assigns = self._assign_ids(ids, pool)
                mass = np.bincount(assigns, minlength=len(pool))
                alive = np.where(mass > 0)[0]
                if len(alive) > max_candidates:
                    alive = alive[np.argsort(-mass[alive])[:max_candidates]]
                pool = pool[alive]
            centroids, assigns = self._lloyd_ids(ids, pool)
            assigns = self._balance_ids(assigns, ids, centroids, min_size)
            sums, norms = self._sums(ids, assigns, len(pool), squared_norms=True)
            counts = np.bincount(assigns, minlength=len(pool)).astype(np.float64)
            centroids = np.zeros((len(pool), self.d), dtype=np.float32)
            nonempty = counts > 0
            centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
        else:
            centroids = pool.copy()
            sums = centroids.copy()
            norms = np.zeros(len(pool), dtype=np.float64)
            assigns = np.empty(0, dtype=np.int32)
            counts = np.zeros(len(pool), dtype=np.int64)

        n_new = len(centroids)
        slots = sorted(released)
        while len(slots) < n_new:
            slots.append(self.k)
            self.posting_lists.append(ArrayPosting(np.empty(0, dtype=np.uint32)))
            self.centroids = np.vstack((self.centroids, np.zeros((1, self.d), dtype=np.float32)))
            self.mu0 = np.vstack((self.mu0, np.zeros((1, self.d), dtype=np.float32)))
            self.a_c = np.vstack((self.a_c, np.zeros((1, self.d), dtype=np.float32)))
            self.q_c = np.append(self.q_c, 0.0)
            self.T = np.append(self.T, 1.0)
            self.k += 1

        # Stable integer sorting preserves the reference posting order while
        # avoiding a full pool-sized boolean scan for every output partition.
        order = np.argsort(assigns, kind="stable")
        ordered_ids = ids[order]
        bounds = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        moved = 0
        for j, c in enumerate(slots):
            if j < n_new:
                new_ids = ordered_ids[bounds[j]:bounds[j + 1]]
                moved += int(np.count_nonzero(self.pam._a[new_ids] != c))
                self.pam._a[new_ids] = c
                self.posting_lists[c] = ArrayPosting(new_ids)
                self.centroids[c] = centroids[j]
                self.mu0[c] = centroids[j].copy()
                self.a_c[c] = sums[j]
                self.q_c[c] = float(norms[j])
            else:
                self.posting_lists[c] = ArrayPosting(np.empty(0, dtype=np.uint32))
                self.centroids[c] = 0
                self.mu0[c] = 0
                self.a_c[c] = 0
                self.q_c[c] = 0.0
            self.T[c] = 1.0
        self.n_reassigns += moved
        self.n_splits += splits
        self.ada_stats["n_local_reindex"] += 1
        self.ada_stats["n_points_reclustered"] += n_pool
        elapsed = time.perf_counter() - t0
        self.total_split_time += elapsed
        self.scale_events.append(dict(rows=n_pool, candidates=n_new,
                                      released=len(released), splits=splits,
                                      reassigned=moved, elapsed_s=elapsed))

    def _global_rebuild(self):
        raise UnsupportedGlobalRebuild(
            "AdaIVF global indicator exceeded tau_g; full reference global "
            "rebuild is not implemented by this compact adapter. Stop this "
            "measurement; do not report skipped maintenance as AdaIVF.")

    def observe_query(self, query, nprobe=8):
        """Apply the reference L2 query-temperature update, without searching.

        Call only for workload queries. Evaluation-only shared native scans
        must not call this hook. Like L2AdaIVF.search, the hook heats/cools once
        per call; it does not itself run maintenance checks.
        """
        query = np.asarray(query, dtype=np.float32)
        sq = squared_l2_to_centroids(query, self.centroids[:self.k])
        candidates = np.argsort(sq)[:min(nprobe, self.k)]
        if len(candidates):
            nearest = float(sq[candidates[0]])
            ratio = (1.0 + sq[candidates]) / (1.0 + nearest)
            factor = np.clip(1.0 + self.eta * ratio, 1.0, 2.0)
            self.T[candidates] *= factor
            unselected = np.ones(self.k, dtype=bool)
            unselected[candidates] = False
            self.T[unselected] = np.maximum(self.T[unselected] * (1.0 - self.nu), 1.0)
        return candidates


def initialize_ada(idx, counts, sums, pointer, squared_norm_sums):
    """Populate Ada state over already initialized compact common storage.

    ``sums`` and ``squared_norm_sums`` must be computed in each posting's
    reference order with the reference float32 NumPy reductions. In particular,
    q_c is float(np.sum(vectors * vectors)), not a float64 re-summation.
    """
    if not isinstance(idx, CompactAdaIVF) or not isinstance(idx.id_to_vec, RawVectors):
        raise TypeError("initialize_ada requires CompactAdaIVF with RawVectors")
    counts = np.asarray(counts)
    sums = np.asarray(sums)
    norms = np.asarray(squared_norm_sums)
    if counts.shape != (idx.k,) or sums.shape != (idx.k, idx.d) or norms.shape != (idx.k,):
        raise ValueError("Ada initial statistics have incompatible shapes")
    if sums.dtype != np.float32 or not np.all(np.isfinite(sums)) or not np.all(np.isfinite(norms)):
        raise ValueError("Ada requires finite reference float32 sums and squared-norm sums")
    if int(counts.sum()) != idx.n or any(len(p) != n for p, n in zip(idx.posting_lists, counts)):
        raise ValueError("Ada counts do not match live posting lengths")
    idx.use_temperature = True
    idx.normalize = False
    idx.tau_s = max(8, int(round(idx.n / idx.nlist)))
    idx.alpha, idx.beta, idx.tau_f = 1.0, 0.5, 0.75
    idx.r_c, idx.iota, idx.check_every = 25, 5, 100
    idx.gamma, idx.tau_g, idx.eta, idx.nu = 0.5, 5.0, 0.1, 0.01
    idx.T = np.ones(idx.k, dtype=np.float64)
    idx.mu0 = idx.centroids.copy()
    idx.a_c = sums.copy()
    idx.q_c = norms.astype(np.float64, copy=True)
    idx.pam = _DensePAM(pointer)
    idx.ops_since_check = 0
    idx.n0 = idx.n
    idx.sigma0 = float(counts.astype(np.float64).std())
    idx.eps0 = float(idx._mean_recon_error())
    idx.ada_stats = dict(n_violations=0, n_local_reindex=0,
                         n_global_rebuild=0, n_points_reclustered=0)
    idx.scale_events = []
    return idx
