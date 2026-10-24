"""Turn result files into LaTeX macros, table rows and the sweep figure.

Usage: python3 make_numbers.py <results_dir> <paper_dir> <production_artifact_dir>
Every number quoted in the paper is produced here from a result file.
"""
from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict


from common import load_olist, serving_times

R, P, A = sys.argv[1], sys.argv[2], sys.argv[3]
main = json.load(open(os.path.join(R, "main.json")))
sweep = json.load(open(os.path.join(R, "sweep.json")))
ov = json.load(open(os.path.join(R, "overhead.json")))
org = json.load(open(os.path.join(R, "organic_checks.json")))
plat = json.load(open(os.path.join(R, "platform.json")))
ARTIFACT_URL = os.environ.get("ARTIFACT_URL", "https://anonymous.4open.science/r/gcc-dasfaa27")


def n(x):
    return f"{x:,}"


m = main["methods"]
B0, B1, B2, B3, B4, G = (m["B0 no gate"], m["B1 table validation"], m["B2 post-hoc verifier (oracle)"],
                         m["B3 freshness-aware RAG verifier"], m["B4 B3 + field allowlist"], m["GCC"])
ol = load_olist()
prod_claims = 0
prod_items, prod_fields = set(), set()
with open(os.path.join(A, "data", "claim_dispositions.csv")) as fh:
    for row in csv.DictReader(fh):
        prod_claims += 1
        prod_items.add(row["item_id"])
        prod_fields.add(row["source_field"])
with open(os.path.join(A, "data", "table_manifest.csv")) as fh:
    man = {r["table_name"]: int(r["released_rows"]) for r in csv.DictReader(fh)}
disp = json.load(open(os.path.join(A, "data", "replay_summary.json")))

mac = {
    "PClaims": n(prod_claims), "PSnapshots": n(disp["versioned_provenance_replay"]["snapshots"]), "PItems": n(len(prod_items)),
    "PFields": str(len(prod_fields)), "PTables": str(len(man)), "PRows": n(sum(man.values())), "PProducts": str(man["product_base"]),
    "PReplayS": f"{plat['production_replay']['wall_s']:.2f}", "PReplayMB": f"{plat['production_replay']['max_rss_mb']:.1f}",
    "VMcpus": str(plat["cpus"]), "VMmem": f"{plat['mem_gb']:.1f}", "VMpython": plat["python"].rsplit(".", 1)[0],
    "OProducts": n(len(ol.products)), "OSellers": n(len(ol.sellers)), "OOffers": n(len(ol.offers)),
    "ORpw": str(main["requests_per_week"]), "OWeeks": str(len(serving_times())),
    "OSnapshots": n(main["snapshots"]), "OItems": n(main["ranked_items"]), "OClaims": n(main["claims"]),
    "BzeroHard": n(B0["hard_violations"]), "BzeroHardPct": f"{B0['hard_violation_pct']:.2f}",
    "BzeroMiss": n(B0["missing_caveat"]),
    "BtwoHard": n(B2["hard_violations"]), "BtwoHardPct": f"{B2['hard_violation_pct']:.2f}",
    "BtwoVal": n(B2["hard_by_cause"]["VAL"]),
    "BthreeHard": n(B3["hard_violations"]), "BthreeHardPct": f"{B3['hard_violation_pct']:.2f}",
    "BfourHard": n(B4["hard_violations"]), "BfourHardPct": f"{B4['hard_violation_pct']:.2f}",
    "BfourGthree": n(B4["hard_by_cause"]["G3"]), "BfourGfive": n(B4["hard_by_cause"]["G5"]), "BfourVal": n(B4["hard_by_cause"]["VAL"]),
    "AsmCaveat": n(org["lineage_items_outcomes_GCC-lineage"].get("refuse->caveat", 0)),
    "AsmDegrade": n(org["lineage_items_outcomes_GCC-lineage"].get("refuse->degrade", 0)),
    "AsmSurface": n(org["lineage_items_outcomes_GCC-lineage"].get("refuse->surface", 0)),
    "OGtwoTotal": n(B0["hard_by_cause"]["G2"]), "OGthreeTotal": n(B0["hard_by_cause"]["G3"]),
    "GCCAgree": f"{G['exact_agreement_pct']:.0f}",
    "AblCaliberHard": n(m["GCC - caliber"]["hard_violations"]), "AblCaliberDown": n(m["GCC - caliber"]["downgrades"]),
    "AblGrainHard": n(m["GCC - grain"]["hard_violations"]), "AblPolicyHard": n(m["GCC - policy"]["hard_violations"]),
    "AblLineageHard": n(m["GCC - lineage"]["hard_violations"]), "AblLineageUp": n(m["GCC - lineage"]["upgrades"]),
    "SQLiteVersion": ov["platform"]["sqlite"], "ArtifactURL": ARTIFACT_URL,
}

# sweep aggregates
agg = defaultdict(lambda: defaultdict(list))
for row in sweep:
    for k, v in row["methods"].items():
        agg[k][row["h"]].append(v)
hs = sorted({row["h"] for row in sweep})
mean = lambda k, h, f: statistics.mean(x[f] for x in agg[k][h])
mac["SweepBtwoFloor"] = f"{mean('B2 post-hoc verifier (oracle)', 0.0, 'hard_violation_pct'):.1f}"
mac["SweepBthreeFloor"] = f"{mean('B3 freshness-aware RAG verifier', 0.0, 'hard_violation_pct'):.1f}"
def fprec(x):
    # precision of unqualified emissions w.r.t. hard violations: 1 - hard / surfaced
    surf = x["unlicensed_surfaced_pct"] / (1 - x["precision_of_surfaced_pct"] / 100) if x["precision_of_surfaced_pct"] < 100 else None
    return 100.0 if not surf else 100 * (1 - x["hard_violation_pct"] / surf)
b2p = [statistics.mean(fprec(x) for x in agg["B2 post-hoc verifier (oracle)"][h]) for h in hs]
mac["SweepBtwoPrecLow"] = f"{min(b2p):.1f}"
mac["SweepBtwoPrecHigh"] = f"{max(b2p):.1f}"
mac["SweepVerifMin"] = f"{min(min(mean(k, h, 'hard_violation_pct') for h in hs) for k in ('B2 post-hoc verifier (oracle)', 'B3 freshness-aware RAG verifier')):.1f}"
mac["SweepVerifMax"] = f"{max(max(mean(k, h, 'hard_violation_pct') for h in hs) for k in ('B2 post-hoc verifier (oracle)', 'B3 freshness-aware RAG verifier')):.1f}"
mac["SweepMaxGCC"] = f"{max(max(x['hard_violation_pct'] for x in agg['GCC'][h]) for h in hs):.1f}"
mac["SweepDisagree"] = str(sum(r["gcc_oracle_disagreements"] for r in sweep))
mac["SweepRuns"] = str(len(sweep))
_spans = [max(x["hard_violation_pct"] for x in agg[k][h]) - min(x["hard_violation_pct"] for x in agg[k][h])
          for k in ("B0 no gate", "B2 post-hoc verifier (oracle)", "B3 freshness-aware RAG verifier", "GCC") for h in hs]
mac["SweepRangeMax"] = f"{math.ceil(max(_spans) * 10 - 1e-9) / 10:.1f}"  # widest min-max range over seeds, rounded up

# overhead
rows = ov["rows"]
base = [r for r in rows if r["requests_per_week"] == 150][0]
mac["OvPfifty"] = f"{base['response_ms_p50']:.1f}"
mac["OvPninetyFive"] = f"{base['response_ms_p95']:.1f}"
mac["OvClaimsPerResp"] = f"{base['claims_per_response_mean']:.0f}"
mac["OvThroughput"] = n(int(round(statistics.mean(r["throughput_claims_per_s"] for r in rows), -2)))
mac["OvLedgerB"] = f"{statistics.mean(r['ledger_bytes_per_claim'] for r in rows):.0f}"
mac["OvContractKB"] = f"{base['contract_table_bytes'] / 1024:.0f}"
first, last = rows[0], rows[-1]
mac["OvThroughputFirst"] = n(first["throughput_claims_per_s"])
mac["OvThroughputLast"] = n(last["throughput_claims_per_s"])
mac["OvThroughputDrop"] = f"{100 * (first['throughput_claims_per_s'] - last['throughput_claims_per_s']) / first['throughput_claims_per_s']:.1f}"
mac["OvPninetyFiveRise"] = f"{last['response_ms_p95'] - first['response_ms_p95']:.1f}"

with open(os.path.join(P, "numbers.tex"), "w") as fh:
    for k, v in mac.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

NAMES = [("B0 no gate", "B0 no gate"), ("B1 table validation", "B1 table valid."),
         ("B2 post-hoc verifier (oracle)", "B2 oracle verif."), ("B3 freshness-aware RAG verifier", "B3 fresh.\\ RAG verif."), ("B4 B3 + field allowlist", "B4 B3 + allowlist"),
         ("GCC", "\\textbf{GCC}"), ("GCC - caliber", "\\quad$-$ caliber"), ("GCC - grain", "\\quad$-$ grain"),
         ("GCC - policy", "\\quad$-$ policy"), ("GCC - lineage", "\\quad$-$ assembly")]
with open(os.path.join(P, "tab_main_rows.tex"), "w") as fh:
    for key, label in NAMES:
        v = m[key]
        c = v["hard_by_cause"]
        cells = [n(v["hard_violations"]), f"{v['hard_violation_pct']:.2f}"] + \
                [n(c[x]) for x in ["G1", "G2", "G3", "G4", "G5", "VAL", "internal"]] + \
                [n(v["missing_caveat"]), f"{100 * (1 - v['hard_violations'] / v['surfaced']):.2f}", f"{v['safe_retention_pct']:.1f}"]
        if key == "GCC":
            cells = [f"\\textbf{{{x}}}" for x in cells]
        sep = "\\midrule\n" if key == "GCC" else ""
        fh.write(f"{sep}{label} & " + " & ".join(cells) + " \\\\\n")

with open(os.path.join(P, "tab_overhead_rows.tex"), "w") as fh:
    for r in rows:
        fh.write(" & ".join([n(r["responses"]), n(r["claims"]), f"{r['claims_per_response_mean']:.1f}",
                             f"{r['response_ms_p50']:.1f}", f"{r['response_ms_p95']:.1f}", f"{r['response_ms_p99']:.1f}",
                             n(r["throughput_claims_per_s"]), f"{r['ledger_bytes_per_claim']:.0f}"]) + " \\\\\n")

# figure data for pgfplots: mean, min and max over seeds per method and h
with open(os.path.join(P, "sweep_data.csv"), "w") as fh:
    keys = [("B0 no gate", "bz"), ("B2 post-hoc verifier (oracle)", "btwo"), ("B3 freshness-aware RAG verifier", "bthree"), ("GCC", "gcc")]
    fh.write("h," + ",".join(f"{t}mean,{t}lo,{t}hi" for _, t in keys) + "\n")
    for h in hs:
        cells = [f"{100 * h:g}"]
        for k, _ in keys:
            xs = [x["hard_violation_pct"] for x in agg[k][h]]
            cells += [f"{statistics.mean(xs):.3f}", f"{min(xs):.3f}", f"{max(xs):.3f}"]
        fh.write(",".join(cells) + "\n")
print(json.dumps(mac, indent=1))

# claim-field table (ground-truth licensed outcomes of the main run)
import gzip
from collections import Counter
cnt = defaultdict(Counter)
with gzip.open(os.path.join(R, "claims_main.csv.gz"), "rt") as fh:
    for row in csv.DictReader(fh):
        cnt[row["field"]][row["licensed"]] += 1
FIELDS = [("price_offer", "price, seller-scoped", "offer", "surface, 30\\,d"),
          ("price_product", "price, product-level", "product", "surface, single fresh price"),
          ("avail_local", "stocked in state", "prod.$\\times$state", "surface, 60\\,d"),
          ("rating", "rating with $n$", "product", "surface, $n\\ge5$, 365\\,d"),
          ("category", "category", "product", "caveat, no clock"),
          ("weight", "weight", "product", "caveat, no clock"),
          ("dims", "dimensions", "product", "caveat, no clock"),
          ("ships_from", "ships-from state", "seller", "caveat, no clock"),
          ("rank_reason", "ranking rationale", "product", "degrade, internal"),
          ("local_popularity", "buyers in city", "prod.$\\times$city", "refuse, customer data"),
          ("marketing_claim", "marketing text", "--", "refuse, unregistered")]
with open(os.path.join(P, "tab_fields_rows.tex"), "w") as fh:
    for f, lab, grain, pol in FIELDS:
        c = cnt[f]
        fh.write(" & ".join([lab, grain, pol, n(sum(c.values()))] + [n(c.get(o, 0)) for o in ("surface", "caveat", "degrade", "refuse")]) + " \\\\\n")
    tot = Counter()
    for f in cnt:
        tot.update(cnt[f])
    fh.write("\\midrule\nTotal & & & " + " & ".join([n(sum(tot.values()))] + [n(tot.get(o, 0)) for o in ("surface", "caveat", "degrade", "refuse")]) + " \\\\\n")
