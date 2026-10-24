"""Evaluate LLM-generated claims with the oracle, the compiled gate, baselines and ablations."""
from __future__ import annotations

import csv
import glob
import gzip
import json
import os
from collections import Counter

from baselines import Evidence, b0, b1, b2, b3, b4
from gate import Ablation, Gate, build_db
from oracle import Oracle
from run_eval import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
# Configurations started but not evaluated (kept in the shipped generations file for transparency).
EXCLUDED = {}
FIELDS_ORDER = ["category", "price_offer", "price_product", "ships_from", "avail_local", "weight", "dims", "rating",
                "local_popularity", "rank_reason", "marketing_claim"]


def load_generations():
    gens = {}
    files = sorted(glob.glob(os.path.join(HERE, "llm", "generations", "*.jsonl")) +
                   glob.glob(os.path.join(HERE, "llm", "generations", "*.jsonl.gz")))
    for f in files:
        fh = gzip.open(f, "rt", encoding="utf-8") if f.endswith(".gz") else open(f, encoding="utf-8")
        for line in fh:
            if line.strip():
                r = json.loads(line)
                gens.setdefault(r["config"], {})[r["req"]] = r
    return gens


def evaluate(ol, wl, ctxs):
    from llm_verbaliser import parse_json, text_audit, to_claims
    gens = load_generations()
    ctx_by = {c["id"]: c for c in ctxs}
    req_cat = {s[0]: s[4] for s in wl.snapshots}
    dbpath = os.path.join(HERE, "results", "llm_eval.db")
    if os.path.exists(dbpath):
        os.remove(dbpath)
    con = build_db(dbpath, ol, wl.funnel)
    oracle, ev = Oracle(ol), Evidence(ol)
    report = {"requests": len(ctxs), "items": sum(len(c["items"]) for c in ctxs), "configs": {}}
    rows_out = []
    report["excluded"] = {k: {"reason": v, "responses": len(gens.get(k, {}))} for k, v in EXCLUDED.items()}
    for cfg in sorted(gens):
        if cfg in EXCLUDED:
            continue
        stats = Counter()
        claims, tok_in, tok_out, secs = [], 0, 0, []
        audit_tot = audit_cov = 0
        items_with_claims = 0
        for rid, ctx in ctx_by.items():
            g = gens[cfg].get(rid)
            if g is None or g.get("error"):
                stats["call_failed"] += 1
                continue
            u = g.get("usage") or {}
            tok_in += u.get("prompt_tokens", 0)
            tok_out += u.get("completion_tokens", 0)
            secs.append(g.get("seconds", 0))
            try:
                parsed = parse_json(g["text"])
            except Exception:
                stats["unparseable_json"] += 1
                continue
            stats["responses_ok"] += 1
            cs = to_claims(ctx, parsed, stats)
            items_with_claims += len({c["product_id"] for c in cs})
            t, cv = text_audit(ctx, parsed)
            audit_tot += t
            audit_cov += cv
            claims += cs
        for i, c in enumerate(claims):
            c["claim_id"] = f"{cfg}-{i:06d}"
        labels = [oracle.label(c, req_cat[c["snapshot_id"]]) for c in claims]
        outs = {"B0 no gate": b0(claims)}
        outs["B1 table validation"], _ = b1(ol, claims)
        outs["B2 post-hoc verifier (oracle)"] = b2(ev, claims)
        outs["B3 freshness-aware RAG verifier"] = b3(ev, claims)
        outs["B4 B3 + field allowlist"] = b4(ev, claims)
        for ab in [Ablation(), Ablation(clock=False, name="GCC - caliber"), Ablation(grain=False, name="GCC - grain"),
                   Ablation(policy=False, name="GCC - policy"), Ablation(lineage=False, name="GCC - lineage")]:
            outs[ab.name] = Gate(con, ab, run_id=f"{cfg}:{ab.name}").run(claims, write_ledger=(ab.name == "GCC"))
        res = {k: metrics(o, labels) for k, o in outs.items()} if claims else {}
        dis = sum(1 for i in range(len(claims)) if outs["GCC"][i] != labels[i][0])
        fields = Counter(c["field"] for c in claims)
        causes = Counter(cause for _, cause in labels)
        lic = Counter(l for l, _ in labels)
        by_field = {}
        for f in FIELDS_ORDER:
            idx = [i for i, c in enumerate(claims) if c["field"] == f]
            if idx:
                by_field[f] = {"claims": len(idx),
                               "licensed": dict(Counter(labels[i][0] for i in idx)),
                               "cause": dict(Counter(labels[i][1] for i in idx))}
        report["configs"][cfg] = {
            "model": next(iter(gens[cfg].values())).get("model"),
            "variant": next(iter(gens[cfg].values())).get("variant"),
            "parse": dict(stats), "claims": len(claims), "claims_per_item": round(len(claims) / max(1, items_with_claims), 2),
            "fields": dict(fields), "licensed_distribution": dict(lic), "cause_distribution": dict(causes),
            "by_field": by_field, "gcc_oracle_disagreements": dis,
            "text_numbers": audit_tot, "text_numbers_in_claims": audit_cov,
            "text_number_coverage_pct": round(100 * audit_cov / audit_tot, 2) if audit_tot else None,
            "prompt_tokens": tok_in, "completion_tokens": tok_out,
            "median_seconds": sorted(secs)[len(secs) // 2] if secs else None,
            "methods": res,
        }
        for i, c in enumerate(claims):
            rows_out.append([cfg, c["claim_id"], c["snapshot_id"], c["product_id"], c["rank"], c["field"], c["key"],
                             c["value"], labels[i][0], labels[i][1], outs["GCC"][i], c["_reason"]["GCC"],
                             outs["B3 freshness-aware RAG verifier"][i], outs["B4 B3 + field allowlist"][i]])
        m = res.get("GCC", {})
        print(f"{cfg:28s} claims={len(claims):6d} parse={dict(stats)} dis={dis} "
              f"B0 hard={res['B0 no gate']['hard_violation_pct']}% B3 hard={res['B3 freshness-aware RAG verifier']['hard_violation_pct']}% "
              f"B4 hard={res['B4 B3 + field allowlist']['hard_violation_pct']}% GCC hard={m.get('hard_violation_pct')}%")
    con.close()
    os.remove(dbpath)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(report, open(os.path.join(HERE, "results", "llm_eval.json"), "w"), indent=2)
    with gzip.open(os.path.join(HERE, "results", "llm_claims.csv.gz"), "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "claim_id", "snapshot_id", "product_id", "rank", "field", "key", "value", "licensed",
                    "cause", "gcc", "gcc_reason", "b3", "b4"])
        w.writerows(rows_out)
    return report
