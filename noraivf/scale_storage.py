"""Compact containers and a batched LIRE port for bounded large-corpus probes.

The uint8 backing is exact for BIGANN/SIFT vectors.  Unchanged postings are
views over a stable ID sort; mutation copies only the affected partition.
NoraIVF uses the existing L2 reference maintenance methods unchanged.
SPFresh batches its frozen-centroid repair assignment using exact FAISS L2,
preserving partition order and per-vector float32 sum updates.  This is an
explicit scaled implementation, not the upstream disk SPFresh system.
"""
from __future__ import annotations

import time
import numpy as np
import faiss

from noraivf.lean import IntMap
from experiments.run_claim_evidence_annbench import L2NovaIVF, L2SPFreshIVF
from noraivf.core import kmeans_faiss


class ArrayPosting:
    def __init__(self, values):
        self.a = np.asarray(values, dtype=np.uint32)
        self.n = len(self.a)
        self.owned = False

    def __len__(self):
        return self.n

    def __iter__(self):
        return map(int, self.a[:self.n])

    def __getitem__(self, i):
        v = self.a[:self.n][i]
        return v if isinstance(v, np.ndarray) else int(v)

    def _reserve(self, extra=0):
        if not self.owned or self.n + extra > len(self.a):
            capacity = max(self.n + extra, self.n + max(16, self.n // 16))
            a = np.empty(capacity, dtype=np.uint32)
            a[:self.n] = self.a[:self.n]
            self.a = a
            self.owned = True

    def __setitem__(self, i, value):
        self._reserve()
        self.a[:self.n][i] = value

    def append(self, i):
        self._reserve(1)
        self.a[self.n] = i
        self.n += 1

    def extend(self, values):
        values = np.asarray(values, dtype=np.uint32)
        self._reserve(len(values))
        self.a[self.n:self.n+len(values)] = values
        self.n += len(values)

    def pop(self):
        self._reserve()
        i = int(self.a[self.n-1])
        self.n -= 1
        return i

    def index(self, i):
        found = np.flatnonzero(self.a[:self.n] == i)
        if not len(found):
            raise ValueError(i)
        return int(found[0])

    def remove(self, i):
        p = self.index(i)
        self._reserve()
        self.a[p:self.n-1] = self.a[p+1:self.n]
        self.n -= 1


def posting_array(posting):
    if isinstance(posting, ArrayPosting):
        return posting.a[:posting.n]
    return np.asarray(posting, dtype=np.uint32)


class RawVectors:
    def __init__(self, base, capacity):
        self.base = base
        self.base_n = len(base)
        self._d = base.shape[1]
        self._cap = capacity
        self.deleted = set()
        self.extra = {}
        self.n = len(base)

    def __len__(self):
        return self.n

    def __contains__(self, i):
        return (0 <= i < self.base_n and i not in self.deleted) or i in self.extra

    def __getitem__(self, i):
        return self.base[i].astype(np.float32) if i < self.base_n else self.extra[i]

    def __setitem__(self, i, vector):
        if i < self.base_n:
            raise ValueError("Original corpus rows are immutable; reinsert using a fresh ID")
        if i not in self.extra:
            self.n += 1
        self.extra[i] = np.asarray(vector, dtype=np.float32).copy()

    def __delitem__(self, i):
        if i < self.base_n:
            if i not in self.deleted:
                self.deleted.add(int(i))
                self.n -= 1
        elif i in self.extra:
            del self.extra[i]
            self.n -= 1

    def gather(self, ids):
        ids = np.asarray(ids)
        if not len(ids):
            return np.empty((0, self._d), dtype=np.float32)
        normal = ids < self.base_n
        if normal.all():
            return np.asarray(self.base[ids], dtype=np.float32)
        out = np.empty((len(ids), self._d), dtype=np.float32)
        out[normal] = self.base[ids[normal]]
        for at in np.flatnonzero(~normal):
            out[at] = self.extra[int(ids[at])]
        return out


class BatchedSPFresh(L2SPFreshIVF):
    """Same two-means/LIRE protocol; batch exact repair against frozen centres."""
    repair_batch = 32768

    def _repair_labels(self, x, quant, centroid_max_norm):
        """Guard BLAS norm-expansion cancellation with the reference kernel.

        The top-two gap is certified against a conservative float32 dot/sum
        error bound.  Ambiguous points use exactly the reference subtraction,
        square, and NumPy row-sum expression over ALL centroids.  A small
        deterministic direct-kernel audit additionally checks each batch.
        """
        distance, labels = quant.search(x, 2)
        best = labels[:,0].astype(np.int32)
        unit_roundoff = np.finfo(np.float32).eps / 2
        gamma = (self.d+4)*unit_roundoff/(1-(self.d+4)*unit_roundoff)
        x64 = x.astype(np.float64)
        norm = np.einsum("ij,ij->i",x64,x64)
        magnitude = norm + centroid_max_norm + 2*np.sqrt(norm*centroid_max_norm)
        # Both the expanded FAISS distance and the direct reference distance
        # can round, for both the winner and runner-up. Eight gamma provides
        # a conservative cushion beyond their four accumulated error terms.
        margin = 8*gamma*magnitude
        fallback = np.flatnonzero(distance[:,1]-distance[:,0] <= margin)
        for i in fallback:
            diff = self.centroids[:self.k]-x[i]
            best[i] = np.argmin(np.sum(diff*diff,axis=1))
        audit = np.unique(np.linspace(0,len(x)-1,min(5,len(x)),dtype=np.int64))
        for i in audit:
            diff = self.centroids[:self.k]-x[i]
            direct = int(np.argmin(np.sum(diff*diff,axis=1)))
            if int(best[i]) != direct:
                raise RuntimeError("Uncertified FAISS/direct repair-label mismatch")
        return best,len(fallback),len(audit)

    def _find_cluster(self, i):
        return self.partition_map.get(i)

    def insert(self, vec):
        # The base method invokes split before it returns.  Registering the ID
        # before maintenance ensures a repair can update its partition map.
        i = self.next_id
        self.next_id += 1
        vec = np.asarray(vec, dtype=np.float32)
        self.id_to_vec[i] = vec
        self.n += 1
        t = time.perf_counter()
        c = self._find_nearest_centroid(vec)
        self.posting_lists[c].append(i)
        self.partition_map[i] = c
        self._post_insert(c, i, vec)
        self.total_insert_time += time.perf_counter() - t
        self.n_inserts += 1
        return i

    def delete(self, i):
        super().delete(i)
        del self.partition_map[i]

    def _merge(self, j):
        moved = list(self.posting_lists[j])
        if not moved:
            return
        diff = self.centroids[:self.k] - self.centroids[j]
        dist = np.sum(diff * diff, axis=1)
        dist[j] = np.inf
        dest = int(np.argmin(dist))
        super()._merge(j)
        self.partition_map._a[moved] = dest

    def _lire_split(self, j):
        t0 = time.perf_counter()
        ids = posting_array(self.posting_lists[j]).copy()
        if len(ids) < 4:
            return
        x = self.id_to_vec.gather(ids)
        t = time.perf_counter()
        centroids_2, labels = kmeans_faiss(x, 2, niter=10, seed=self.rng.randint(0, 2**31))
        self.total_split_clustering_time += time.perf_counter() - t
        new_c = self.k
        self.k += 1
        old_ids, new_ids = ids[labels == 0], ids[labels == 1]
        self.posting_lists[j] = ArrayPosting(old_ids)
        self.posting_lists.append(ArrayPosting(new_ids))
        self.cluster_sums.append(np.zeros(self.d, dtype=np.float32))
        if len(old_ids):
            old_x = x[labels == 0]
            self.centroids[j] = old_x.mean(axis=0)
            self.cluster_sums[j] = old_x.sum(axis=0)
        if len(new_ids):
            new_x = x[labels == 1]
            new_centroid = new_x.mean(axis=0)
            self.cluster_sums[new_c] = new_x.sum(axis=0)
        else:
            new_centroid = centroids_2[1]
        self.centroids = np.vstack([self.centroids, new_centroid])
        self.partition_map._a[old_ids] = j
        self.partition_map._a[new_ids] = new_c
        del x, labels

        if self.do_reassign:
            t = time.perf_counter()
            diff = self.centroids[:self.k] - self.centroids[j]
            dist = np.sum(diff * diff, axis=1)
            dist[j] = np.inf
            dist[new_c] = np.inf
            nearest = np.argsort(dist)[:min(self.L, self.k-2)]
            affected = set([j, new_c] + list(nearest))
            quant = faiss.IndexFlatL2(self.d)
            quant.add(self.centroids[:self.k])
            centroid_max_norm = float(np.sum(self.centroids[:self.k].astype(np.float64)**2,axis=1).max())
            moved = scanned = fallback_count = direct_checks = 0
            for c in affected:
                source = posting_array(self.posting_lists[c]).copy()
                best = np.empty(len(source), dtype=np.int32)
                for s in range(0, len(source), self.repair_batch):
                    e = min(s+self.repair_batch, len(source))
                    lab,nfallback,nchecks = self._repair_labels(self.id_to_vec.gather(source[s:e]),quant,centroid_max_norm)
                    best[s:e] = lab
                    fallback_count += nfallback
                    direct_checks += nchecks
                scanned += len(source)
                change = best != c
                change_ids, destinations = source[change], best[change]
                self.posting_lists[c] = ArrayPosting(source[~change])
                # Preserve the reference's sequential float32 sum updates.
                for idx, destination in zip(change_ids, destinations):
                    v = self.id_to_vec[int(idx)]
                    self.cluster_sums[c] -= v
                    self.cluster_sums[int(destination)] += v
                for destination in np.unique(destinations):
                    incoming = change_ids[destinations == destination]
                    posting = self.posting_lists[int(destination)]
                    if isinstance(posting, ArrayPosting):
                        posting.extend(incoming)
                    else:
                        posting.extend(map(int, incoming))
                self.partition_map._a[change_ids] = destinations
                moved += len(change_ids)
            self.n_reassigns += moved
            for c in affected:
                if len(self.posting_lists[c]):
                    self.centroids[c] = self.cluster_sums[c] / len(self.posting_lists[c])
            self.total_split_reassign_time += time.perf_counter() - t
            self.repair_events.append(dict(scanned=scanned,moved=moved,affected=sorted(map(int,affected)),
                                           direct_near_tie_fallbacks=fallback_count,
                                           deterministic_direct_label_checks=direct_checks,
                                           observed_audit_label_mismatches=0,
                                           direct_audit_is_exhaustive=False))
        self.n_splits += 1
        self.total_split_time += time.perf_counter() - t0
        self.split_history.append((j, new_c, self.centroids[j].copy(), self.centroids[new_c].copy()))


def make_index(method, base, centroids, assignments, order, counts, sums,
               seed=42, capacity_extra=4096):
    """Bulk-initialize reference state; vectorize only container population."""
    cls = L2NovaIVF if method == "NoraIVF" else BatchedSPFresh
    idx = cls.__new__(cls)
    idx.d = base.shape[1]
    idx.n = len(base)
    idx.nlist = idx.k = len(centroids)
    idx.seed = seed
    idx.rng = np.random.RandomState(seed)
    idx.centroids = centroids.copy()
    idx.assignments = assignments
    idx.data = None
    idx.id_to_vec = RawVectors(base, len(base)+capacity_extra)
    idx.next_id = len(base)
    bounds = np.concatenate(([0], np.cumsum(counts)))
    idx.posting_lists = [ArrayPosting(order[bounds[c]:bounds[c+1]]) for c in range(idx.k)]
    for name in ["total_insert_time", "total_delete_time", "total_split_time", "total_merge_time"]:
        setattr(idx, name, 0.0)
    for name in ["n_inserts", "n_deletes", "n_splits", "n_merges", "n_reassigns"]:
        setattr(idx, name, 0)
    idx.split_history = []
    idx.split_factor = 2.0
    idx.merge_factor = .25
    idx.avg_size = idx.n / idx.k
    idx.T_split = 2 * idx.avg_size
    idx.T_merge = .25 * idx.avg_size
    idx.L = 5
    pointer = IntMap(len(base)+capacity_extra)
    pointer._a[:len(base)] = assignments
    if method == "NoraIVF":
        idx.n_j = counts.copy()
        idx.a_j = sums.copy()
        idx.s_j = np.array([float(np.dot(s, s)) for s in sums], dtype=np.float64)
        idx.ppm = pointer
        idx.positions = IntMap(len(base)+capacity_extra)
        for c in range(idx.k):
            ids = order[bounds[c]:bounds[c+1]]
            idx.positions._a[ids] = np.arange(len(ids), dtype=np.int32)
        idx.use_scdkm = True
        idx.total_scdkm_time = 0.0
    else:
        idx.partition_map = pointer
        idx.cluster_sums = [s.copy() for s in sums]
        idx.do_reassign = True
        idx.total_split_clustering_time = 0.0
        idx.total_split_reassign_time = 0.0
        idx.repair_events = []
    return idx


def search(index, query, k=100, nprobe=8, batch=65536):
    """Shared exact L2 scan of selected postings; uint8 distances are lossless."""
    diff = index.centroids[:index.k] - query
    nearest = np.argsort(np.sum(diff * diff, axis=1))[:nprobe]
    best_ids = np.empty(0, dtype=np.uint32)
    best_dist = np.empty(0, dtype=np.float32)
    scanned = 0
    for c in nearest:
        ids = posting_array(index.posting_lists[c])
        scanned += len(ids)
        for s in range(0, len(ids), batch):
            chunk_ids = ids[s:s+batch]
            if getattr(index,"native_scan",None) is not None:
                dist = index.native_scan.distances(index.id_to_vec,chunk_ids,query)
            else:
                x = index.id_to_vec.gather(chunk_ids)
                x -= query
                dist = np.einsum("ij,ij->i", x, x)
            # Keep the globally smallest (distance, ID) keys, including ties at
            # each chunk boundary. Arbitrary argpartition ties can lose IDs
            # which should survive the global merge.
            m = min(k, len(dist))
            cut = np.partition(dist, m-1)[m-1]
            below = np.flatnonzero(dist < cut)
            tied = np.flatnonzero(dist == cut)
            tied = tied[np.argsort(chunk_ids[tied], kind="stable")[:m-len(below)]]
            take = np.concatenate((below, tied))
            all_dist = np.concatenate((best_dist, dist[take]))
            all_ids = np.concatenate((best_ids, chunk_ids[take]))
            choose = np.lexsort((all_ids, all_dist))[:k]
            best_ids, best_dist = all_ids[choose], all_dist[choose]
    return best_ids, best_dist, scanned
