"""Real full-corpus query / bounded-maintenance probe on BIGANN SIFT1B.

This is not a billion-update stream.  It builds all 1B postings, evaluates
official corpus queries, forces a declared number of matched partition splits,
then deletes and reinserts a small deterministic set of original vectors.
Reinserted IDs map back to original IDs for unchanged full-corpus ground truth.
All timings are measurements; no timing extrapolation is performed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import faiss
import numpy as np

from experiments.build_sift1b_scale import atomic_json, binary_matrix, sample_fingerprint
from noraivf.scale_storage import make_index, posting_array, search


def memory():
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:")):
            key, value, _ = line.split()
            name = "process_cumulative_VmHWM_gib" if key == "VmHWM:" else key.rstrip(":")+"_gib"
            values[name] = int(value)/1024**2
    r = resource.getrusage(resource.RUSAGE_SELF)
    values.update(process_cumulative_peak_rss_gib=r.ru_maxrss/1024**2,
                  process_cumulative_major_page_faults=r.ru_majflt,
                  process_cumulative_minor_page_faults=r.ru_minflt)
    return values


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_postings(cache, base, assignments, counts):
    """Stable compact posting IDs and reference-order float32 sums."""
    path = cache / "postings_state.json"
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("complete"):
            return np.load(cache/"ids.npy", mmap_mode="r"), np.load(cache/"sums.npy"), state
    state = dict(complete=False, n=len(base), order="stable ascending ID within partition")
    t = time.perf_counter()
    order64 = np.argsort(assignments, kind="stable")
    order = np.lib.format.open_memmap(cache/"ids.npy", mode="w+", dtype=np.uint32,
                                     shape=(len(base),))
    for s in range(0, len(base), 10000000):
        order[s:s+10000000] = order64[s:s+10000000]
    order.flush()
    del order64
    gc.collect()
    state["sort_and_persist_s"] = time.perf_counter()-t
    atomic_json(path, state)
    print(f"POSTINGS sorted {len(base):,} IDs in {state['sort_and_persist_s']:.1f}s", flush=True)
    bounds = np.concatenate(([0], np.cumsum(counts)))
    sums = np.zeros((len(counts), base.shape[1]), dtype=np.float32)
    t = time.perf_counter()
    for c in range(len(counts)):
        ids = order[bounds[c]:bounds[c+1]]
        if len(ids):
            # Match the reference's float32 row-order summation exactly.
            sums[c] = base[ids].astype(np.float32).sum(axis=0)
        if (c+1) % 256 == 0:
            print(f"SUMS {c+1}/{len(counts)} elapsed={time.perf_counter()-t:.1f}s", flush=True)
    np.save(cache/"sums.npy", sums)
    state.update(sums_s=time.perf_counter()-t, complete=True, memory=memory())
    atomic_json(path, state)
    return order, sums, state


def prefault_mappings(base, assignments, order):
    """Explicit untimed page touch for cached/reused as well as fresh mappings."""
    before = memory()
    t = time.perf_counter()
    rows = []
    page_bytes = os.sysconf("SC_PAGE_SIZE")
    for name, array in [("corpus",base),("assignments",assignments),("posting_ids",order)]:
        flat = array.reshape(-1)
        stride = max(1,page_bytes//flat.dtype.itemsize)
        checksum = int(np.sum(flat[::stride],dtype=np.uint64))
        checksum += int(flat[-1])
        rows.append(dict(mapping=name,bytes=array.nbytes,page_stride_bytes=page_bytes,
                         touched_elements=len(flat[::stride])+1,page_sample_sum=checksum))
    after = memory()
    result = dict(wall_s=time.perf_counter()-t,mappings=rows,before_memory=before,after_memory=after,
                  minor_page_faults=after["process_cumulative_minor_page_faults"]-before["process_cumulative_minor_page_faults"],
                  major_page_faults=after["process_cumulative_major_page_faults"]-before["process_cumulative_major_page_faults"],
                  purpose="One untimed page touch of every shared corpus/assignment/posting mapping before either method")
    print(f"MAPPINGS PREFAULT {json.dumps(result)}",flush=True)
    return result


def evaluate(idx, queries, gt, nprobes, canonical):
    records = []
    for nprobe in nprobes:
        # Three explicit untimed warm-ups on the first three measured queries.
        for q in queries[:3]:
            search(idx, q, k=100, nprobe=nprobe)
        lat, r10, r100, scanned = [], [], [], []
        for q, truth in zip(queries, gt):
            t = time.perf_counter()
            got, _, n = search(idx, q, k=100, nprobe=nprobe)
            lat.append(time.perf_counter()-t)
            got = [canonical.get(int(x), int(x)) for x in got]
            r10.append(len(set(got[:10]) & set(map(int, truth[:10])))/10)
            r100.append(len(set(got[:100]) & set(map(int, truth[:100])))/100)
            scanned.append(n)
        records.append(dict(nprobe=nprobe, n_queries=len(queries),
                            recall_at_10=float(np.mean(r10)), recall_at_100=float(np.mean(r100)),
                            qps=len(queries)/sum(lat), p50_ms=float(np.percentile(lat,50)*1000),
                            p95_ms=float(np.percentile(lat,95)*1000), p99_ms=float(np.percentile(lat,99)*1000),
                            mean_candidates=float(np.mean(scanned)),
                            per_query_seconds=lat, per_query_recall_at_10=r10,
                            per_query_recall_at_100=r100, per_query_candidates=scanned))
        print(f"QUERY nprobe={nprobe} R10={np.mean(r10):.4f} R100={np.mean(r100):.4f} "
              f"QPS={len(queries)/sum(lat):.3f}", flush=True)
    return records


def partition_audit(idx, sample_ids):
    assert np.isfinite(idx.centroids[:idx.k]).all(), "Non-finite maintained centroids"
    pointer = idx.ppm if hasattr(idx, "ppm") else idx.partition_map
    ids = [int(x) for x in sample_ids if int(x) in idx.id_to_vec]
    x = idx.id_to_vec.gather(ids)
    q = faiss.IndexFlatL2(idx.d)
    q.add(idx.centroids[:idx.k])
    _, nearest = q.search(x, 1)
    assigned = pointer._a[ids]
    npa = int(np.count_nonzero(assigned != nearest[:,0]))
    if hasattr(idx, "n_j"):
        assert np.array_equal(idx.n_j, [len(p) for p in idx.posting_lists])
        assert np.isfinite(idx.a_j).all() and np.isfinite(idx.s_j).all()
    assert sum(map(len, idx.posting_lists)) == idx.n == len(idx.id_to_vec)
    # Sample membership/pointers after actual maintenance.
    for i in ids[:100]:
        c = pointer[i]
        assert i in posting_array(idx.posting_lists[c])
        if hasattr(idx, "positions"):
            assert idx.posting_lists[c][idx.positions[i]] == i
    return dict(sampled_live_ids=len(ids), nearest_centroid_violations=npa,
                violation_rate=npa/max(len(ids),1), corpus_membership_count=idx.n)


def verify_full_ground_truth(base, queries, truth, query_ids):
    """Independent exact scan of the entire corpus for selected supplied GT rows."""
    best = [np.empty(0,dtype=np.float32) for _ in queries]
    t = time.perf_counter()
    for s in range(0,len(base),1000000):
        block = base[s:s+1000000].astype(np.float32)
        for qi,q in enumerate(queries):
            diff = block-q
            distance = np.einsum("ij,ij->i",diff,diff)
            take = np.partition(distance,min(99,len(distance)-1))[:100]
            best[qi] = np.sort(np.concatenate([best[qi],take]))[:100]
        if (s+len(block)) % 50000000 == 0:
            print(f"GT VERIFY {s+len(block):,}/{len(base):,} elapsed={time.perf_counter()-t:.1f}s",flush=True)
    rows=[]
    for qi,q in enumerate(queries):
        diff = base[truth[qi]].astype(np.float32)-q
        supplied = np.einsum("ij,ij->i",diff,diff)
        same = np.array_equal(best[qi],supplied)
        rows.append(dict(query_id=int(query_ids[qi]), distance_ranks_match=bool(same),
                         exact_top100_squared_distances=best[qi].tolist(),
                         supplied_top100_squared_distances=supplied.tolist()))
        if not same:
            raise RuntimeError(f"Supplied GT failed independent full-corpus exact scan for query {query_ids[qi]}")
    return dict(rows_scanned=len(base), n_queries=len(queries), wall_s=time.perf_counter()-t,
                criterion="all top-100 exact integer squared-distance ranks match; tied IDs may differ",rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=["NoraIVF", "SPFresh"], choices=["NoraIVF", "SPFresh"])
    ap.add_argument("--force-splits", type=int, default=3)
    ap.add_argument("--update-pairs", type=int, default=1000)
    ap.add_argument("--query-count", type=int, default=100)
    ap.add_argument("--nprobes", nargs="+", type=int, default=[1,8,32])
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit-sample", type=int, default=20000)
    ap.add_argument("--verify-gt-queries", type=int, default=1)
    ap.add_argument("--scan-library", default="",help="Optional common exact single-thread uint8 scan shared library")
    ap.add_argument("--reuse-gt-from",default="",help="Reuse only a validated exact GT phase from an earlier saved probe JSON")
    ap.add_argument("--subset-smoke", action="store_true",
                    help="Allow a subset cache and recompute exact subset GT; never report as 1B")
    a = ap.parse_args()
    faiss.omp_set_num_threads(a.threads)
    np.seterr(divide="raise",over="raise",invalid="raise")
    cache = Path(a.cache)
    build = json.loads((cache/"build_state.json").read_text())
    if not build.get("complete") or build["completed_rows"] != build["config"]["n"]:
        raise RuntimeError("The full requested assignment cache is not complete")
    if sample_fingerprint(a.base) != build["config"]["base"]:
        raise RuntimeError("Corpus provenance does not match the assignment cache")
    n = build["config"]["n"]
    if not a.subset_smoke and n != 1000000000:
        raise RuntimeError("Full-scale run requires exactly 1,000,000,000 vectors")
    available_kib = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:")))
    # Raw corpus pages + maps + sorting scratch + generous working reserve.
    needed_bytes = n * (build["config"]["dim"] + 32) + 20 * 1024**3
    if available_kib * 1024 < needed_bytes:
        raise RuntimeError(f"Available RAM is below conservative {needed_bytes/1024**3:.1f} GiB probe reserve")
    disk = os.statvfs(cache)
    if disk.f_bavail * disk.f_frsize < n * 4 + 5 * 1024**3:
        raise RuntimeError("Insufficient disk space for compact postings plus 5 GiB reserve")
    base = binary_matrix(a.base, np.uint8)[:n]
    query_all = binary_matrix(a.queries, np.uint8)
    gt_all = binary_matrix(a.gt, np.int32)
    if query_all.shape[0] != gt_all.shape[0] or query_all.shape[1] != base.shape[1]:
        raise RuntimeError("Query and GT dimensions do not match")
    rng = np.random.default_rng(a.seed)
    query_ids = rng.choice(len(query_all), size=min(a.query_count,len(query_all)), replace=False)
    queries = query_all[query_ids].astype(np.float32)
    gt = gt_all[query_ids,:100]
    if a.subset_smoke:
        exact = faiss.IndexFlatL2(base.shape[1])
        exact.add(base.astype(np.float32))
        _, gt = exact.search(queries,100)
        del exact
    assert np.all((gt >= 0) & (gt < n))
    assignments = np.load(cache/"assign.npy", mmap_mode="r")
    centroids = np.load(cache/"centroids.npy")
    counts = np.load(cache/"counts.npy")
    assert int(counts.sum()) == n == len(assignments)
    if not np.isfinite(centroids).all():
        raise RuntimeError("Non-finite centroids")
    # Check supplied neighbor ranks by exact integer SIFT distances.
    if not a.subset_smoke:
        for q, row in zip(queries, gt):
            diff = base[row].astype(np.float32)-q
            distance = np.einsum("ij,ij->i",diff,diff)
            assert np.all(distance[1:] >= distance[:-1]), "Supplied GT is not distance-sorted"
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise RuntimeError("Output exists; choose a new output file to preserve prior results")
    start = time.perf_counter()
    result = dict(complete=False,
                  scope="subset validation only" if a.subset_smoke else "full 1B corpus, bounded maintenance probe",
                  config=vars(a), n=n, dim=base.shape[1], nlist=len(centroids),
                  query_ids=query_ids.tolist(), initial_build=build,
                  started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                  provenance=dict(query_sha256=file_hash(a.queries), gt_sha256=file_hash(a.gt),
                                  code_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                                  storage_source_sha256=file_hash("noraivf/scale_storage.py"),
                                  harness_source_sha256=file_hash(__file__)),
                  environment=dict(host=platform.node(), python=sys.version, numpy=np.__version__,
                                   faiss=faiss.__version__, threads=a.threads,
                                   cpu=next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),"unknown")),
                  implementation=dict(vectors="original immutable uint8 plus float32 reinsert overlay",
                                      postings="uint32, copy-on-write per modified partition",
                                      nora="existing L2NovaIVF maintenance methods unchanged",
                                      spfresh="same two-means/LIRE port with batched frozen-centroid assignment, conservative float32 near-tie fallback to reference direct kernel and per-batch direct-label audit; added int32 ID-to-partition map; NumPy linear find-and-shift deletion within a posting",
                                      search="same exact single-thread native uint8 L2 posting scan for both methods" if a.scan_library else "same batched exact NumPy L2 posting scan for both methods",
                                      parallelism="Nora coordinate loop is serial; SPFresh kmeans/repair uses configured FAISS threads; sequential query loop",
                                      warmup="Explicit untimed page touch of all corpus, assignment and posting-ID mappings before either method, then 3 untimed queries before each measured nprobe setting",
                                      search_tie_order="Global (exact squared distance, current ID) ascending order, including per-chunk boundary ties; reinsert IDs canonicalized for recall after retrieval",
                                      sums="float32, same order as reference; numerical limits retained"),
                  limitations=[f"{a.force_splits} forced split events are not a complete 1B-update streaming workload.",
                               "Timing describes these Python/FAISS memory implementations, not upstream native SPFresh.",
                               "Scaled storage/query/baseline optimizations differ from earlier small-scale scripts; their latency values are not one continuous scaling curve.",
                               "Recall uses supplied BIGANN ground truth; ties may have multiple valid neighbor IDs.",
                               "QPS is sequential, warm-cache, and returns top 100; percentile estimates use the configured query sample.",
                               "RSS includes mmap-resident corpus pages and shared cache mappings.",
                               "Peak RSS and page-fault counters are process cumulative across preparation and both methods; only VmRSS is the current phase-boundary value."], methods={})
    atomic_json(out,result)
    if a.scan_library:
        result["provenance"].update(native_scan_library_sha256=file_hash(a.scan_library),
                                     native_scan_cpp_sha256=file_hash("noraivf/uint8_scan.cpp"),
                                     native_scan_python_sha256=file_hash("noraivf/native_scan.py"))
        result["implementation"]["native_build"] = "g++ (Ubuntu 11.4.0) -O3 -std=c++17 -march=native -shared -fPIC; no OpenMP; exact uint32 accumulation converted to float32"
    if a.reuse_gt_from:
        prior=json.loads(Path(a.reuse_gt_from).read_text())
        validation=prior["independent_gt_validation"]
        assert prior["n"] == n and prior["query_ids"] == query_ids.tolist()
        assert prior["initial_build"]["config"] == build["config"]
        assert prior["provenance"]["query_sha256"] == result["provenance"]["query_sha256"]
        assert prior["provenance"]["gt_sha256"] == result["provenance"]["gt_sha256"]
        assert validation["rows_scanned"] == n and validation["n_queries"] >= a.verify_gt_queries
        assert all(r["distance_ranks_match"] for r in validation["rows"])
        result["independent_gt_validation"] = validation
        result["provenance"]["gt_validation_reused_from"] = dict(path=a.reuse_gt_from,sha256=file_hash(a.reuse_gt_from))
        atomic_json(out,result)
    elif a.verify_gt_queries:
        nv = min(a.verify_gt_queries,len(queries))
        result["independent_gt_validation"] = verify_full_ground_truth(base,queries[:nv],gt[:nv],query_ids[:nv])
        atomic_json(out,result)
    order, sums, posting_meta = prepare_postings(cache,base,assignments,counts)
    result["postings_build"] = posting_meta
    result["mapping_prefault"] = prefault_mappings(base,assignments,order)
    atomic_json(out,result)
    targets = np.argsort(-counts, kind="stable")[:a.force_splits].tolist()
    result["forced_original_partition_ids"] = targets
    audit_ids = rng.integers(0,n,size=min(a.audit_sample,n),dtype=np.int64)
    # Deterministic real IDs, identical for both methods. Select from the
    # largest initial partition so the update burst also stresses maintained
    # partitions; no synthetic points or invented result rows are involved.
    bounds = np.concatenate(([0],np.cumsum(counts)))
    c = int(targets[0]) if targets else int(np.argmax(counts))
    initial_ids = order[bounds[c]:bounds[c+1]]
    update_ids = initial_ids[np.linspace(0,len(initial_ids)-1,min(a.update_pairs,len(initial_ids)),dtype=np.int64)]
    result["update_original_ids"] = list(map(int,update_ids))
    for method in a.methods:
        print(f"METHOD {method}: bootstrap",flush=True)
        t = time.perf_counter()
        idx = make_index(method,base,centroids,assignments,order,counts,sums,a.seed,len(update_ids)+4096)
        if a.scan_library:
            from noraivf.native_scan import NativeUint8Scan
            idx.native_scan=NativeUint8Scan(a.scan_library)
        rec = dict(complete=False, bootstrap_s=time.perf_counter()-t, initial_memory=memory(),
                   numeric_policy="NumPy divide/overflow/invalid raise immediately; every saved audit requires finite centroids",
                   split_events=[], update_events=[], frontier={}, audits={})
        result["methods"][method] = rec
        atomic_json(out,result)
        print(f"METHOD {method}: initial queries",flush=True)
        rec["frontier"]["initial"] = evaluate(idx,queries,gt,a.nprobes,{})
        rec["audits"]["initial"] = partition_audit(idx,audit_ids)
        atomic_json(out,result)
        split_name = "_split" if method == "NoraIVF" else "_lire_split"
        for c in targets:
            size = len(idx.posting_lists[c])
            t = time.perf_counter()
            reassign_before = idx.n_reassigns
            getattr(idx,split_name)(int(c))
            event = dict(original_partition=int(c), size_before=size,
                         wall_s=time.perf_counter()-t, n_clusters_after=idx.k,
                         reassignments=idx.n_reassigns-reassign_before,
                         split_child_sizes=[len(idx.posting_lists[c]),len(idx.posting_lists[-1])],
                         minimum_child_size=min(len(idx.posting_lists[c]),len(idx.posting_lists[-1])),
                         memory=memory())
            selected=[]
            for child in [c,idx.k-1]:
                child_ids=posting_array(idx.posting_lists[child])
                if len(child_ids):
                    selected.extend(child_ids[np.linspace(0,len(child_ids)-1,min(100,len(child_ids)),dtype=np.int64)])
            event["child_audit"]=partition_audit(idx,np.asarray(selected,dtype=np.int64))
            rec["split_events"].append(event)
            atomic_json(out,result)
            print(f"SPLIT {method} {json.dumps(event)}",flush=True)
        rec["frontier"]["after_splits"] = evaluate(idx,queries,gt,a.nprobes,{})
        rec["audits"]["after_splits"] = partition_audit(idx,audit_ids)
        canonical = {}
        splits_before, merges_before = idx.n_splits, idx.n_merges
        for old_id in update_ids:
            old_id = int(old_id)
            vector = base[old_id].astype(np.float32)
            t = time.perf_counter()
            idx.delete(old_id)
            delete_s = time.perf_counter()-t
            t = time.perf_counter()
            new_id = idx.insert(vector)
            insert_s = time.perf_counter()-t
            canonical[new_id] = old_id
            rec["update_events"].append(dict(original_id=old_id,new_id=new_id,
                                              delete_s=delete_s,insert_s=insert_s))
        rec["update_summary"] = dict(pairs=len(update_ids), vectors_are_real_corpus_rows=True,
                                    additional_splits=idx.n_splits-splits_before,
                                    additional_merges=idx.n_merges-merges_before,
                                    delete_wall_s=sum(x["delete_s"] for x in rec["update_events"]),
                                    insert_wall_s=sum(x["insert_s"] for x in rec["update_events"]))
        atomic_json(out,result)
        rec["frontier"]["after_updates"] = evaluate(idx,queries,gt,a.nprobes,canonical)
        rec["audits"]["after_updates"] = partition_audit(idx,np.concatenate([audit_ids,np.array(list(canonical))]))
        rec.update(complete=True, n_live=idx.n, n_clusters=idx.k, n_splits=idx.n_splits,
                   n_merges=idx.n_merges,n_reassignments=idx.n_reassigns,final_memory=memory())
        if method == "NoraIVF":
            rec["scdkm_wall_s"] = idx.total_scdkm_time
        else:
            rec.update(clustering_wall_s=idx.total_split_clustering_time,
                       repair_wall_s=idx.total_split_reassign_time,repair_events=idx.repair_events)
        atomic_json(out,result)
        del idx
        gc.collect()
    result.update(complete=True,elapsed_s=time.perf_counter()-start,
                  finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()), final_memory=memory())
    atomic_json(out,result)
    print(f"COMPLETE {out}",flush=True)


if __name__ == "__main__":
    main()
