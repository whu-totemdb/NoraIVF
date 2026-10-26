"""Compact storage adapter for this repository's raw-L2 DeDrift port.

This is not the upstream DeDrift system.  The reference post-insert and
post-delete hooks are inherited unchanged: every deletion rebalances, as does
every ``rebalance_every``-th insertion.  A rebalance reclusters the largest
and smallest partitions together; it is not a single-partition split.

Only vector gathering, posting construction, and ID-to-partition lookup use
compact arrays.  Partition selection/order, FAISS input order and RNG, ordered
postings, and float32 row-order reductions match ``L2DeDriftIVF``.
"""
from __future__ import annotations

import time

import numpy as np

from experiments.run_claim_evidence_annbench import L2DeDriftIVF
from noraivf.core import kmeans_faiss
from noraivf.scale_storage import ArrayPosting, posting_array


def initialize_dedrift(idx, counts, sums, pointer, *, k2=10, rebalance_every=100):
    """Initialize DeDrift-specific state after the common bulk bootstrap.

    The caller supplies a ``CompactDeDrift.__new__(CompactDeDrift)`` with the
    normal IVF base fields, zeroed generic timers/counters, ``RawVectors``,
    ``ArrayPosting`` lists, and the common initial centroids.  ``pointer`` is
    an IntMap already populated with the initial assignments.  ``sums`` must
    use the reference's float32 reduction in each posting's row order.
    No full-corpus Python dictionaries or vector copies are constructed here.
    """
    counts = np.asarray(counts)
    sums = np.asarray(sums)
    if counts.shape != (idx.k,) or sums.shape != (idx.k, idx.d):
        raise ValueError("DeDrift counts/sums do not match the initialized index")
    if sums.dtype != np.float32:
        raise ValueError("Reference DeDrift cluster sums must be float32")
    if not np.array_equal(counts, [len(p) for p in idx.posting_lists]):
        raise ValueError("DeDrift counts do not match initialized postings")
    if k2 < 2 or k2 % 2 or rebalance_every < 1:
        raise ValueError("Compact DeDrift requires positive even k2 and a positive interval")
    idx.k2 = int(k2)
    idx.rebalance_every = int(rebalance_every)
    idx.ops_since_rebalance = 0
    idx.cluster_sums = [s.copy() for s in sums]
    idx.partition_map = pointer
    idx.rebalance_events = []
    idx._rebalance_trigger = "direct"
    return idx


class CompactDeDrift(L2DeDriftIVF):
    """Reference DeDrift policy with compact postings and dense ID lookup."""

    def _find_cluster(self, vid):
        return self.partition_map.get(vid)

    def insert(self, vec):
        # Register the pointer before the inherited post-hook can rebalance.
        vec = np.asarray(vec, dtype=np.float32)
        i = self.next_id
        self.next_id += 1
        self.id_to_vec[i] = vec.copy()
        self.n += 1
        t0 = time.perf_counter()
        c = self._find_nearest_centroid(vec)
        self.posting_lists[c].append(i)
        self.partition_map[i] = c
        previous_trigger = self._rebalance_trigger
        self._rebalance_trigger = "insert_interval"
        try:
            self._post_insert(c, i, vec)
        finally:
            self._rebalance_trigger = previous_trigger
        self.total_insert_time += time.perf_counter() - t0
        self.n_inserts += 1
        return i

    def delete(self, i):
        previous_trigger = self._rebalance_trigger
        self._rebalance_trigger = "delete"
        try:
            super().delete(i)
            del self.partition_map[i]
        finally:
            self._rebalance_trigger = previous_trigger

    def _rebalance(self):
        t0 = time.perf_counter()
        sizes = [(c, len(pl)) for c, pl in enumerate(self.posting_lists) if len(pl) > 0]
        if len(sizes) < 5:
            return
        sizes.sort(key=lambda x: -x[1])
        k = min(self.k2 // 2, len(sizes) // 2)
        largest, smallest = sizes[:k], sizes[-k:]
        selected = largest + smallest
        # The reference indexes k2 released slots.  Do not silently change its
        # policy if a smaller index cannot supply that many nonempty slots.
        if len(selected) != self.k2:
            raise ValueError("DeDrift rebalance requires at least k2 nonempty partitions")
        ids = np.concatenate([posting_array(self.posting_lists[c]) for c, _ in selected])
        if len(ids) < self.k2 * 2:
            return
        old_partition = self.partition_map._a[ids].copy()
        vecs = self.id_to_vec.gather(ids)
        split_seed = int(self.rng.randint(0, 2**31))
        t_cluster = time.perf_counter()
        _, assigns = kmeans_faiss(vecs, self.k2, niter=10, seed=split_seed)
        clustering_s = time.perf_counter() - t_cluster

        for c_orig, _ in selected:
            self.posting_lists[c_orig] = ArrayPosting(np.empty(0, dtype=np.uint32))
            self.cluster_sums[c_orig] = np.zeros(self.d, dtype=np.float32)
            self.centroids[c_orig] = np.zeros(self.d, dtype=np.float32)

        target_slots = []
        for c_target in range(self.k2):
            c_actual = (largest[c_target % len(largest)][0] if c_target < len(largest)
                        else smallest[c_target - len(largest)][0])
            target_slots.append(c_actual)
            mask = assigns == c_target
            idxs = ids[mask]
            self.posting_lists[c_actual] = ArrayPosting(idxs)
            if len(idxs):
                # Boolean indexing retains the reference's row order and a
                # contiguous float32 matrix for the identical axis-0 sum.
                self.cluster_sums[c_actual] = vecs[mask].sum(axis=0)
                self.centroids[c_actual] = self.cluster_sums[c_actual] / len(idxs)
                self.partition_map._a[idxs] = c_actual

        new_partition = np.asarray(target_slots, dtype=np.int32)[assigns]
        membership_changes = int(np.count_nonzero(new_partition != old_partition))
        self.n_splits += 1  # Retained reference counter: one *rebalance* event.
        dt = time.perf_counter() - t0
        self.total_split_time += dt
        self.rebalance_events.append(dict(
            event="rebalance", trigger=self._rebalance_trigger,
            selected_partitions=[int(c) for c, _ in selected],
            selected_sizes_before=[int(n) for _, n in selected],
            selected_sizes_after=[len(self.posting_lists[c]) for c, _ in selected],
            rows_reclustered=len(ids), n_clusters_before=self.k, n_clusters_after=self.k,
            partition_membership_changes=membership_changes,
            seed=split_seed, clustering_wall_s=clustering_s, wall_s=dt,
        ))
