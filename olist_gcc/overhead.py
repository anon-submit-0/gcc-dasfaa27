"""Operational overhead of the compiled gate on the online path.

For each recommendation response, the gate decides all of its candidate
claims and appends their ledger rows in one transaction, as an inline
deployment would.  We report per-response and per-claim latency, throughput
and ledger storage, at several workload sizes.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import defaultdict

from common import load_olist
from gate import Ablation, Gate, build_db
from workload import AsOf, Faults, generate


def pct(xs, q):
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1)))))
    return xs[k]


def ledger_bytes(con):
    try:
        r = con.execute("SELECT SUM(pgsize) FROM dbstat WHERE name IN ('coverage_ledger', "
                        "'sqlite_autoindex_coverage_ledger_1')").fetchone()
        return r[0]
    except Exception:
        return None


def contract_bytes(con):
    try:
        r = con.execute("SELECT SUM(pgsize) FROM dbstat WHERE name LIKE 'claim_registry%' OR name LIKE "
                        "'evidence_binding%' OR name LIKE 'sqlite_autoindex_claim_registry%' OR name LIKE "
                        "'sqlite_autoindex_evidence_binding%'").fetchone()
        return r[0]
    except Exception:
        return None


def measure(ol, asof, rpw, seed, dbpath):
    wl = generate(ol, Faults(), requests_per_week=rpw, seed=seed, asof=asof)
    if os.path.exists(dbpath):
        os.remove(dbpath)
    con = build_db(dbpath, ol, wl.funnel)
    g = Gate(con, Ablation(), run_id="online")
    by_snap = defaultdict(list)
    for c in wl.claims:
        by_snap[c["snapshot_id"]].append(c)
    lat = []
    per_claim = []
    t_all = time.perf_counter()
    for snap, cs in by_snap.items():
        t0 = time.perf_counter()
        g.run(cs, write_ledger=True)
        dt = time.perf_counter() - t0
        lat.append(dt * 1000)
        per_claim.append(dt * 1e6 / len(cs))
    total = time.perf_counter() - t_all
    n = len(wl.claims)
    lb = ledger_bytes(con)
    res = {
        "requests_per_week": rpw, "responses": len(by_snap), "claims": n,
        "claims_per_response_mean": round(n / len(by_snap), 1),
        "response_ms_p50": round(pct(lat, 50), 3), "response_ms_p95": round(pct(lat, 95), 3),
        "response_ms_p99": round(pct(lat, 99), 3), "response_ms_mean": round(statistics.mean(lat), 3),
        "claim_us_p50": round(pct(per_claim, 50), 1), "claim_us_p95": round(pct(per_claim, 95), 1),
        "throughput_claims_per_s": round(n / total), "total_s": round(total, 2),
        "ledger_bytes": lb, "ledger_bytes_per_claim": round(lb / n, 1) if lb else None,
        "contract_table_bytes": contract_bytes(con),
        "db_file_bytes": os.path.getsize(dbpath),
    }
    con.close()
    return res


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results"
    sizes = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["75", "150", "300", "600"])]
    ol = load_olist()
    asof = AsOf(ol)
    rows = []
    for rpw in sizes:
        r = measure(ol, asof, rpw, 21, os.path.join(out, "overhead.db"))
        print(json.dumps(r))
        rows.append(r)
    json.dump({"platform": {"python": sys.version.split()[0], "sqlite": __import__("sqlite3").sqlite_version,
                            "cpus": os.cpu_count()}, "rows": rows}, open(os.path.join(out, "overhead.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
