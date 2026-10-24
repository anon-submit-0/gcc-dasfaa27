"""Regenerate the main Boundary-O run and dump per-claim labels and outcomes.

Writes results/claims_main.csv.gz (one row per claim) and
results/organic_checks.json (facts quoted in the paper's text).
"""
from __future__ import annotations

import csv
import gzip
import json
import os
from collections import Counter

import pandas as pd

from baselines import Evidence, b0, b1, b2, b3, b4
from common import DAY, load_olist, serving_times
from gate import Ablation, Gate, build_db
from oracle import Oracle
from workload import AsOf, Faults, generate


def main(out="results", rpw=150, seed=7):
    ol = load_olist()
    wl = generate(ol, Faults(), requests_per_week=rpw, seed=seed, asof=AsOf(ol))
    req_cat = {s[0]: s[4] for s in wl.snapshots}
    db = os.path.join(out, "dump.db")
    if os.path.exists(db):
        os.remove(db)
    con = build_db(db, ol, wl.funnel)
    orc = Oracle(ol)
    labels = [orc.label(c, req_cat[c["snapshot_id"]]) for c in wl.claims]
    ev = Evidence(ol)
    outs = {"B0": b0(wl.claims), "B1": b1(ol, wl.claims)[0], "B2": b2(ev, wl.claims), "B3": b3(ev, wl.claims), "B4": b4(ev, wl.claims)}
    for ab in [Ablation(), Ablation(clock=False, name="GCC-caliber"), Ablation(grain=False, name="GCC-grain"),
               Ablation(policy=False, name="GCC-policy"), Ablation(lineage=False, name="GCC-lineage")]:
        outs[ab.name] = Gate(con, ab, run_id=ab.name).run(wl.claims, write_ledger=False)
    cols = ["claim_id", "snapshot_id", "t", "rank", "product_id", "field", "grain", "key", "value", "true_value",
            "injected", "licensed", "cause"] + list(outs)
    with gzip.open(os.path.join(out, "claims_main.csv.gz"), "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for i, c in enumerate(wl.claims):
            w.writerow([c["claim_id"], c["snapshot_id"], c["t"], c["rank"], c["product_id"], c["field"], c["grain"],
                        c["key"], c["value"], c["true_value"], c["injected"], labels[i][0], labels[i][1]]
                       + [outs[k][i] for k in outs])
    # organic checks quoted in the text
    cause_by_inj = Counter((labels[i][1], c["injected"]) for i, c in enumerate(wl.claims))
    days = pd.to_datetime(ol.offers.observed_at, unit="s").dt.floor("D")
    st, en = pd.Timestamp("2018-01-01"), pd.Timestamp("2018-08-26")
    per_day = days[(days >= st) & (days <= en)].value_counts()
    all_days = pd.date_range(st, en, freq="D")
    missing_days = [str(d.date()) for d in all_days if d not in per_day.index]
    b2_val = Counter((c["field"], c["injected"]) for i, c in enumerate(wl.claims)
                     if outs["B2"][i] == "surface" and labels[i][1] == "VAL")
    pp = [i for i, c in enumerate(wl.claims) if c["field"] == "price_product"]
    res = {
        "cause_by_injection": {f"{k[0]}|{k[1]}": v for k, v in sorted(cause_by_inj.items())},
        "offer_rows_per_day_min_2018": int(per_day.min()), "days_without_offer_rows_2018": missing_days,
        "b2_surviving_value_errors_by_field": {f"{k[0]}|{k[1]}": v for k, v in b2_val.items()},
        "price_product_claims": len(pp),
        "price_product_G3": sum(1 for i in pp if labels[i][1] == "G3"),
        "price_product_G3_uninjected": sum(1 for i in pp if labels[i][1] == "G3" and wl.claims[i]["injected"] == "none"),
        "G2_by_field": dict(Counter(c["field"] for i, c in enumerate(wl.claims) if labels[i][1] == "G2")),
        "G2_uninjected": sum(1 for i, c in enumerate(wl.claims) if labels[i][1] == "G2" and c["injected"] == "none"),
        "G4_uninjected": sum(1 for i, c in enumerate(wl.claims) if labels[i][1] == "G4" and c["injected"] == "none"),
        "lineage_items_outcomes_GCC-lineage": dict(Counter(labels[i][0] + "->" + outs["GCC-lineage"][i] for i, c in enumerate(wl.claims) if c["injected"] == "lineage")),
        "gcc_caliber_downgrades_by_field": dict(Counter(c["field"] for i, c in enumerate(wl.claims)
                                                     if outs["GCC-caliber"][i] != labels[i][0] and labels[i][0] == "surface")),
    }
    json.dump(res, open(os.path.join(out, "organic_checks.json"), "w"), indent=2)
    print(json.dumps(res, indent=1))
    con.close()
    os.remove(db)


if __name__ == "__main__":
    main()
