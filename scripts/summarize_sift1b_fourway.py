"""Validate a completed SIFT1B four-way run and derive its compact table rows.

The three query repeats reuse one final index; they are not independent update
trials. All displayed values come from the input record, with update cost and
query aggregates recomputed from its individual operation/query measurements.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import tempfile


METHODS = ("SPFresh", "AdaIVF", "DeDrift", "NoraIVF")
LABELS = {"SPFresh": "SPFresh", "AdaIVF": "Ada-IVF",
          "DeDrift": "DeDrift", "NoraIVF": "NoraIVF"}
BUDGETS = (1, 8, 32)
N = 1_000_000_000
PAIRS = 1_000
QUERIES = 100
REPEATS = 3


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, where, minimum=0.0, positive=False):
    require(type(value) in (int, float) and math.isfinite(value), where + ": non-finite/non-numeric value")
    require(value > minimum if positive else value >= minimum, where + ": out-of-range value")
    return float(value)


def integer(value, where, minimum=0):
    require(type(value) is int and value >= minimum, where + ": expected an integer")
    return value


def close(reported, computed, where):
    actual = number(reported, where)
    require(math.isclose(actual, computed, rel_tol=1e-12, abs_tol=1e-12),
            f"{where}: reported {actual!r} differs from recomputed {computed!r}")


def unique_ids(value, size, where, upper=None):
    require(isinstance(value, list) and len(value) == size, where + ": wrong number of IDs")
    for item in value:
        integer(item, where)
        require(upper is None or item < upper, where + ": ID out of range")
    require(len(set(value)) == size, where + ": duplicate IDs")
    return value


def query_measurements(records, order, where):
    require(isinstance(records, list) and len(records) == len(BUDGETS), where + ": missing probe budget")
    by_budget = {}
    for record in records:
        budget = integer(record["nprobe"], where + " nprobe", 1)
        require(budget in BUDGETS and budget not in by_budget, where + ": unexpected/duplicate probe budget")
        require(record["n_queries"] == QUERIES, where + ": wrong query count")
        arrays = {}
        for key in ("per_query_seconds", "per_query_recall_at_10",
                    "per_query_recall_at_100", "per_query_candidates"):
            values = record[key]
            require(isinstance(values, list) and len(values) == QUERIES,
                    where + " " + key + ": wrong length")
            arrays[key] = values
        seconds = [number(v, where + " query seconds", positive=True) for v in arrays["per_query_seconds"]]
        recalls = {}
        for k in (10, 100):
            values = [number(v, where + f" recall@{k}") for v in arrays[f"per_query_recall_at_{k}"]]
            require(all(v <= 1 and abs(v*k-round(v*k)) < 1e-10 for v in values),
                    where + f": impossible per-query recall@{k}")
            mean = math.fsum(values) / QUERIES
            close(record[f"recall_at_{k}"], mean, where + f" recall@{k}")
            recalls[k] = dict(zip(order, values))
        candidates = [integer(v, where + " candidates", 1) for v in arrays["per_query_candidates"]]
        require(all(v <= N for v in candidates), where + ": candidate count exceeds corpus size")
        close(record["mean_candidates"], math.fsum(candidates)/QUERIES, where + " mean candidates")
        qps = QUERIES / math.fsum(seconds)
        close(record["qps"], qps, where + " QPS")
        by_budget[budget] = dict(recall_at_100=math.fsum(recalls[100].values())/QUERIES,
                                qps=qps, recalls=recalls, candidates=dict(zip(order, candidates)))
    return by_budget


def validate_identity(result, updates, evaluation, operational):
    identity = result["identity"]
    canonical = identity["canonical_json"]
    require(isinstance(canonical, str), "Missing canonical run identity")
    require(hashlib.sha256(canonical.encode()).hexdigest() == identity["sha256"], "Run identity hash mismatch")
    descriptor = json.loads(canonical)
    require(json.dumps(descriptor, sort_keys=True, separators=(",", ":")) == canonical,
            "Run identity is not canonical JSON")
    require(descriptor["config"] == result["config"], "Configuration differs from run identity")
    require(descriptor["evaluation_query_ids"] == evaluation
            and descriptor["operational_query_ids"] == operational,
            "Query trace differs from run identity")
    trace_hash = hashlib.sha256(struct.pack("<" + "q"*len(updates), *updates)).hexdigest()
    require(descriptor["update_original_ids_sha256"] == trace_hash, "Update trace differs from run identity")
    provenance = result["provenance"]
    for key in ("source_sha256", "source_paths", "cache_sha256", "cache_paths",
                "base_sampled_fingerprint", "native_library_sha256", "query_sha256", "gt_sha256"):
        require(descriptor[key] == provenance[key], "Provenance differs from run identity: " + key)
    return descriptor


def method_summary(method, record, result, descriptor, updates, evaluation, operational):
    where = method
    require(record.get("complete") is True and "pending_update" not in record,
            where + ": method is incomplete")
    require(record["n_live"] == N, where + ": wrong final live count")
    events = record["update_events"]
    require(isinstance(events, list) and len(events) == PAIRS, where + ": incomplete update trace")
    deletes, inserts = [], []
    for i, event in enumerate(events):
        require(event["pair"] == i+1 and event["original_id"] == updates[i]
                and event["new_id"] == N+i, where + ": update trace mismatch")
        deletes.append(number(event["delete_s"], where + " delete time", positive=True))
        inserts.append(number(event["insert_s"], where + " insert time", positive=True))
    delete_s, insert_s = math.fsum(deletes), math.fsum(inserts)
    total = delete_s + insert_s
    update_summary = record["update_summary"]
    require(update_summary["pairs"] == PAIRS and update_summary["workload_queries"] == QUERIES,
            where + ": update summary has wrong workload size")
    for key, value in (("delete_wall_s", delete_s), ("insert_wall_s", insert_s), ("total_update_wall_s", total)):
        close(update_summary[key], value, where + " " + key)
    workload = record["workload_query_events"]
    require(isinstance(workload, list) and len(workload) == QUERIES, where + ": missing workload query")
    for i, event in enumerate(workload):
        require(event["after_pair"] == 10*(i+1) and event["query_id"] == operational[i],
                where + ": workload query trace mismatch")
        number(event["wall_s"], where + " workload query time", positive=True)
        require(integer(event["candidates"], where + " workload candidates", 1) <= N,
                where + ": workload candidate count exceeds corpus size")
    workload_s = math.fsum(event["wall_s"] for event in workload)
    close(update_summary["workload_query_wall_s"], workload_s, where + " workload query total")
    audits = record["audits"]
    for phase in ("initial", "final"):
        require(audits[phase]["corpus_membership_count"] == N, where + ": invalid " + phase + " audit")
    audit = audits["final_full_id_integrity"]
    require(audit.get("complete") is True and audit.get("included_in_performance_timings") is False,
            where + ": full ID audit is incomplete or contaminates timings")
    for key, expected in (("expected_live_ids", N), ("expected_deleted_ids", PAIRS),
                          ("expected_overlay_ids", PAIRS), ("scanned_posting_entries", N),
                          ("unique_posting_ids", N), ("original_live_ids_present", N-PAIRS)):
        require(audit[key] == expected, where + ": full ID audit disagrees with trace: " + key)
    partitions = record["final_partition_sizes"]
    require(isinstance(partitions, list) and len(partitions) == record["n_clusters"]
            and audit["scanned_partitions"] == len(partitions), where + ": partition audit incomplete")
    require(sum(integer(v, where + " partition size") for v in partitions) == N,
            where + ": partition sizes have wrong total")
    audit_identity = audit["identity"]
    harness = "experiments/run_sift1b_fourway.py"
    require(audit_identity["method"] == method
            and audit_identity["run_identity_sha256"] == result["identity"]["sha256"]
            and audit_identity["harness_sha256"] == descriptor["source_sha256"][harness]
            and audit_identity["harness_path"] == descriptor["source_paths"][harness],
            where + ": full ID audit belongs to another run/method/harness")
    initial = query_measurements(record["frontier"]["initial"], evaluation, where + " initial")
    repeats = record["frontier"]["final_repeats"]
    require(isinstance(repeats, list) and len(repeats) == REPEATS, where + ": missing final query repeat")
    final = []
    for i, repeat in enumerate(repeats):
        require(repeat.get("complete") is True and repeat["repeat"] == i,
                where + ": incomplete or misnumbered query repeat")
        order = unique_ids(repeat["query_order"], QUERIES, where + " query order")
        require(set(order) == set(evaluation), where + ": query repeat has a different sample")
        measurements = query_measurements(repeat["measurements"], order, where + f" repeat {i}")
        if final:
            for budget in BUDGETS:
                require(measurements[budget]["recalls"] == final[0][budget]["recalls"],
                        where + f": per-query recall changed between repeats at nprobe={budget}")
                require(measurements[budget]["candidates"] == final[0][budget]["candidates"],
                        where + f": per-query candidate count changed between repeats at nprobe={budget}")
        final.append(measurements)
    query = {str(budget): dict(recall_at_100=final[0][budget]["recall_at_100"],
                              qps_median=statistics.median(row[budget]["qps"] for row in final),
                              qps_repeats=[row[budget]["qps"] for row in final]) for budget in BUDGETS}
    return dict(total_update_wall_s=total, delete_wall_s=delete_s, insert_wall_s=insert_s,
                workload_query_wall_s=workload_s, final_queries=query,
                final_partition_count=len(partitions), full_id_integrity_verified=True), initial


def summarize(result, input_sha256):
    require(result.get("complete") is True and "failure" not in result, "Run is incomplete or failed")
    config = result["config"]
    require(config.get("pilot") is False and config.get("subset_smoke") is False,
            "Pilot/subset results cannot produce the paper table")
    require(result["n"] == N and result["nlist"] == 4096, "Expected the full billion-vector, 4,096-partition initial index")
    require(result["initial_build"].get("complete") is True
            and result["initial_build"]["completed_rows"] == N
            and result["initial_build"]["config"]["n"] == N,
            "Initial corpus build is incomplete")
    declared_methods = config["methods"]
    require(isinstance(declared_methods, list) and len(declared_methods) == len(METHODS)
            and set(declared_methods) == set(METHODS) and set(result["methods"]) == set(METHODS),
            "Expected exactly SPFresh, AdaIVF, DeDrift, and NoraIVF")
    for key, expected in (("update_pairs", PAIRS), ("workload_query_every", 10),
                          ("query_count", QUERIES), ("query_repeats", REPEATS), ("threads", 32)):
        require(config[key] == expected, "Unexpected workload/configuration: " + key)
    require(config["nprobes"] == list(BUDGETS), "Expected probe budgets 1, 8, 32")
    require(result["implementation"]["forced_splits"] == 0, "Forced-split run is not the four-policy workload")
    require(result["timing_definitions"]["primary_metric"] == "total_update_wall_s", "Unexpected primary metric")
    updates = unique_ids(result["update_original_ids"], PAIRS, "update IDs", N)
    evaluation = unique_ids(result["evaluation_query_ids"], QUERIES, "evaluation IDs", 10000)
    operational = unique_ids(result["operational_query_ids"], QUERIES, "workload query IDs", 10000)
    require(not set(evaluation).intersection(operational), "Evaluation/workload query samples overlap")
    descriptor = validate_identity(result, updates, evaluation, operational)
    methods, first_initial = {}, None
    for method in METHODS:
        methods[method], initial = method_summary(method, result["methods"][method], result,
                                                  descriptor, updates, evaluation, operational)
        if first_initial is None:
            first_initial = initial
        else:
            require(all(initial[b]["recalls"] == first_initial[b]["recalls"] for b in BUDGETS),
                    "Initial query recalls differ across supposedly identical index states")
    summary = dict(input_sha256=input_sha256, run_identity_sha256=result["identity"]["sha256"],
                   complete=True, indexed_vectors=N, update_pairs=PAIRS, workload_queries=QUERIES,
                   evaluation_queries=QUERIES, nprobes=list(BUDGETS), method_order=declared_methods,
                   table_row_order=list(METHODS), query_repeats=REPEATS,
                   query_summary="Median QPS across three query repeats of one final index; not independent update trials",
                   timing_definition=result["timing_definitions"]["total_update_wall_s"],
                   implementation=result["implementation"], limitations=result["limitations"], methods=methods)
    return summary


def latex_rows(summary):
    display = {}
    for method in METHODS:
        item = summary["methods"][method]
        values = [item["total_update_wall_s"]]
        values += [item["final_queries"][str(b)]["recall_at_100"] for b in BUDGETS]
        values += [item["final_queries"][str(b)]["qps_median"] for b in BUDGETS]
        display[method] = [format(v, f".{p}f") for v, p in zip(values, (1, 4, 4, 4, 2, 2, 2))]
    best = []
    for column in range(7):
        values = {method: Decimal(display[method][column]) for method in METHODS}
        optimum = min(values.values()) if column == 0 else max(values.values())
        best.append([method for method in METHODS if values[method] == optimum])
    lines = ["% Derived from input SHA-256: " + summary["input_sha256"],
             "% Method | update seconds | Recall@100 (1,8,32) | median QPS (1,8,32)"]
    for method in METHODS:
        cells = ["\\" + ("bestnum" if method in best[c] else "tabnum") + "{" + v + "}"
                 for c, v in enumerate(display[method])]
        lines.append(LABELS[method] + " & " + " & ".join(cells) + r" \\")
        summary["methods"][method]["display_values"] = display[method]
    summary["best_at_display_precision"] = best
    return "\n".join(lines) + "\n"


def no_duplicate_keys(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, "Duplicate JSON key: " + key)
        obj[key] = value
    return obj


def atomic_text(path, text):
    tmp = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as handle:
            tmp = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--rows", required=True)
    args = parser.parse_args()
    paths = [Path(p).resolve() for p in (args.result, args.summary, args.rows)]
    for i, path in enumerate(paths):
        for other in paths[:i]:
            require(path != other and not (path.exists() and other.exists() and path.samefile(other)),
                    "Input, summary, and row paths must be different files")
    data = paths[0].read_bytes()
    result = json.loads(data, object_pairs_hook=no_duplicate_keys,
                        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Invalid JSON number: " + value)))
    summary = summarize(result, hashlib.sha256(data).hexdigest())
    rows = latex_rows(summary)
    summary["latex_rows_sha256"] = hashlib.sha256(rows.encode()).hexdigest()
    # No files are created until the entire input and every output row validate.
    atomic_text(paths[1], json.dumps(summary, indent=2, allow_nan=False) + "\n")
    atomic_text(paths[2], rows)
    print("WROTE", paths[1], paths[2])


if __name__ == "__main__":
    main()
