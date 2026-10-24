"""Run the Olist-GCC comparison, ablations and fault-rate sweep.

Usage: python3 run_eval.py [--rpw 150] [--seed 7] [--out results]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict

from baselines import Evidence, b0, b1, b2, b3, b4
from common import load_olist
from gate import RANK, Ablation, Gate, build_db
from oracle import Oracle
from workload import AsOf, Faults, generate

CAUSES = ["needs_caveat", "internal", "G1", "G2", "G3", "G4", "G5", "VAL"]


def metrics(outs, labels):
    lic = [l for l, _ in labels]
    cause = [c for _, c in labels]
    n = len(outs)
    unl = [i for i in range(n) if outs[i] == "surface" and lic[i] != "surface"]
    hard = [i for i in unl if lic[i] in ("refuse", "degrade")]
    soft = [i for i in unl if lic[i] == "caveat"]
    by_cause = Counter(cause[i] for i in hard)
    safe = [i for i in range(n) if lic[i] in ("surface", "caveat")]
    retained = sum(1 for i in safe if outs[i] in ("surface", "caveat"))
    lic_surface = [i for i in range(n) if lic[i] == "surface"]
    surfaced_ok = sum(1 for i in lic_surface if outs[i] == "surface")
    upgrades = sum(1 for i in range(n) if RANK[outs[i]] > RANK[lic[i]])
    downgrades = sum(1 for i in range(n) if RANK[outs[i]] < RANK[lic[i]])
    surf = sum(1 for o in outs if o == "surface")
    return {
        "claims": n,
        "surfaced": surf,
        "unlicensed_surfaced": len(unl),
        "unlicensed_surfaced_pct": round(100 * len(unl) / n, 2),
        "hard_violations": len(hard),
        "hard_violation_pct": round(100 * len(hard) / n, 2),
        "missing_caveat": len(soft),
        "missing_caveat_pct": round(100 * len(soft) / n, 2),
        "hard_by_cause": {k: by_cause.get(k, 0) for k in CAUSES},
        "precision_of_surfaced_pct": round(100 * (surf - len(unl)) / surf, 2) if surf else None,
        "safe_retention_pct": round(100 * retained / len(safe), 2),
        "licensed_surface_recall_pct": round(100 * surfaced_ok / len(lic_surface), 2),
        "upgrades": upgrades,
        "downgrades": downgrades,
        "exact_agreement_pct": round(100 * sum(1 for i in range(n) if outs[i] == lic[i]) / n, 2),
    }


def run_once(ol, faults, rpw, seed, dbpath, asof=None, oracle=None, ev=None, ablations=True, timing=True):
    t0 = time.time()
    wl = generate(ol, faults, requests_per_week=rpw, seed=seed, asof=asof)
    t_gen = time.time() - t0
    req_cat = {s[0]: s[4] for s in wl.snapshots}
    if os.path.exists(dbpath):
        os.remove(dbpath)
    con = build_db(dbpath, ol, wl.funnel)
    oracle = oracle or Oracle(ol)
    labels = [oracle.label(c, req_cat[c["snapshot_id"]]) for c in wl.claims]
    ev = ev or Evidence(ol)
    res = {}
    outs = {}
    outs["B0 no gate"] = b0(wl.claims)
    outs["B1 table validation"], checks = b1(ol, wl.claims)
    outs["B2 post-hoc verifier (oracle)"] = b2(ev, wl.claims)
    outs["B3 freshness-aware RAG verifier"] = b3(ev, wl.claims)
    outs["B4 B3 + field allowlist"] = b4(ev, wl.claims)
    abl = [Ablation()]
    if ablations:
        abl += [Ablation(clock=False, name="GCC - caliber"), Ablation(grain=False, name="GCC - grain"),
                Ablation(policy=False, name="GCC - policy"), Ablation(lineage=False, name="GCC - lineage")]
    gate_time = {}
    for ab in abl:
        g = Gate(con, ab, run_id=ab.name)
        t1 = time.time()
        outs[ab.name] = g.run(wl.claims, write_ledger=(ab.name == "GCC"))
        gate_time[ab.name] = time.time() - t1
    for k, o in outs.items():
        res[k] = metrics(o, labels)
    lic_dist = Counter(l for l, _ in labels)
    cause_dist = Counter(c for _, c in labels)
    inj = Counter(c["injected"] for c in wl.claims)
    fields = Counter(c["field"] for c in wl.claims)
    # disagreements between compiled gate and oracle (should be zero)
    dis = [(c["claim_id"], c["field"], outs["GCC"][i], labels[i]) for i, c in enumerate(wl.claims) if outs["GCC"][i] != labels[i][0]]
    reasons = Counter(c["_reason"]["GCC"] for c in wl.claims)
    summary = {
        "faults": vars(faults), "requests_per_week": rpw, "seed": seed,
        "snapshots": len(wl.snapshots), "ranked_items": sum(1 for f in wl.funnel if f[2] == "ranked"),
        "claims": len(wl.claims), "fields": dict(fields), "injected": dict(inj),
        "licensed_distribution": dict(lic_dist), "cause_distribution": dict(cause_dist),
        "table_checks_all_pass": all(all(v.values()) for v in checks.values()),
        "gcc_oracle_disagreements": len(dis), "disagreement_examples": dis[:10],
        "gcc_reasons": dict(reasons), "gate_seconds": gate_time, "generation_seconds": t_gen,
        "methods": res,
    }
    return summary, wl, con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpw", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ol = load_olist()
    asof, oracle, ev = AsOf(ol), Oracle(ol), Evidence(ol)
    summary, wl, con = run_once(ol, Faults(), a.rpw, a.seed, os.path.join(a.out, "olist_gcc.db"), asof, oracle, ev)
    json.dump(summary, open(os.path.join(a.out, "main.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "methods"}, indent=1)[:3000])
    for k, v in summary["methods"].items():
        print(f"{k:36s} hard={v['hard_violations']:6d} ({v['hard_violation_pct']:5.2f}%) soft={v['missing_caveat']:6d} {v['hard_by_cause']} "
              f"prec={v['precision_of_surfaced_pct']} ret={v['safe_retention_pct']} rec={v['licensed_surface_recall_pct']} agree={v['exact_agreement_pct']}")
    if a.sweep:
        sweep = []
        for h in [0.0, 0.05, 0.10, 0.20, 0.30]:
            for seed in [11, 12, 13]:
                s, _, c2 = run_once(ol, Faults(h=h), a.rpw, seed, os.path.join(a.out, "sweep.db"), asof, oracle, ev,
                                    ablations=False)
                c2.close()
                sweep.append({"h": h, "seed": seed, "methods": {k: {kk: v[kk] for kk in
                              ("unlicensed_surfaced_pct", "hard_violation_pct", "missing_caveat_pct", "precision_of_surfaced_pct", "safe_retention_pct",
                               "licensed_surface_recall_pct")} for k, v in s["methods"].items()},
                              "gcc_oracle_disagreements": s["gcc_oracle_disagreements"], "claims": s["claims"]})
                print("sweep", h, seed, s["claims"], {k: v["hard_violation_pct"] for k, v in s["methods"].items()})
        json.dump(sweep, open(os.path.join(a.out, "sweep.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
