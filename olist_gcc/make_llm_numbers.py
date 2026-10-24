"""LaTeX macros and table rows for the real-LLM verbalizer experiment (RQ5).

Usage: python3 make_llm_numbers.py <results_dir> <paper_dir>
"""
from __future__ import annotations

import json
import math
import os
import sys

R, P = sys.argv[1], sys.argv[2]
rep = json.load(open(os.path.join(R, "llm_eval.json")))
C = rep["configs"]
MODELS = [("deepseek-flash", "DeepSeek-Flash", "Ds"), ("gpt-6-sol", "GPT-6-Sol", "Gpt"), ("glm-5.3", "GLM-5.3", "Glm")]
VARIANTS = [("plain", "plain", "Plain"), ("careful", "careful", "Careful")]
CAUSES = ["G1", "G2", "G3", "G4", "G5", "VAL", "internal"]


def n(x):
    return f"{x:,}"


rows, mac = [], {}
agg = {k: [] for k in ("g5", "g1", "b0", "b2", "b3", "b4", "miss", "gcc", "ret", "cov", "val", "int", "g4", "g2", "b4g5")}
tot_claims = tot_resp = tot_calls = dis = 0
per = {}
for mid, mname, mkey in MODELS:
    first = True
    for v, vname, vkey in VARIANTS:
        cfg = f"{mid}__{v}"
        if cfg not in C:
            continue
        c = C[cfg]
        M = c["methods"]
        B0, B2, B3, B4, G = (M["B0 no gate"], M["B2 post-hoc verifier (oracle)"], M["B3 freshness-aware RAG verifier"],
                             M["B4 B3 + field allowlist"], M["GCC"])
        hb = B0["hard_by_cause"]
        tot_claims += c["claims"]
        tot_resp += c["parse"].get("responses_ok", 0)
        dis += c["gcc_oracle_disagreements"]
        agg["b0"].append(B0["hard_violation_pct"]); agg["b2"].append(B2["hard_violation_pct"])
        agg["b3"].append(B3["hard_violation_pct"]); agg["b4"].append(B4["hard_violation_pct"])
        agg["miss"].append(B4["missing_caveat_pct"]); agg["gcc"].append(G["hard_violation_pct"] + G["missing_caveat_pct"])
        agg["ret"].append(G["safe_retention_pct"]); agg["cov"].append(c["text_number_coverage_pct"])
        agg["val"].append(100 * hb["VAL"] / c["claims"]); agg["int"].append(100 * hb["internal"] / c["claims"])
        agg["g1"].append(100 * hb["G1"] / c["claims"]); agg["g5"].append(100 * hb["G5"] / c["claims"])
        agg["g4"].append(100 * hb["G4"] / c["claims"]); agg["g2"].append(100 * hb["G2"] / c["claims"])
        agg["b4g5"].append(B4["hard_by_cause"]["G5"] / max(1, B4["hard_violations"]))
        per[cfg] = (c, hb)
        cells = [mname if first else "", vname, n(c["parse"].get("responses_ok", 0)), n(c["claims"]),
                 f"{B0['hard_violation_pct']:.1f}"] + [n(hb[k]) for k in CAUSES] + [
                 f"{B3['hard_violation_pct']:.1f}", f"{B4['hard_violation_pct']:.2f}", f"{B4['missing_caveat_pct']:.1f}",
                 f"{G['hard_violation_pct']:.0f}", f"{G['safe_retention_pct']:.0f}"]
        rows.append(" & ".join(cells) + " \\\\")
        first = False
    if mid != MODELS[-1][0]:
        rows.append("\\midrule")


def floor1(x, d=1):
    return f"{math.floor(x * 10 ** d + 1e-9) / 10 ** d:.{d}f}"


def ceil1(x, d=1):
    return f"{math.ceil(x * 10 ** d - 1e-9) / 10 ** d:.{d}f}"


def rng(k, fmt="{:.1f}"):
    return fmt.format(min(agg[k])), fmt.format(max(agg[k]))


# careful vs plain: change in G4 claims and in B0 hard-violation share, per model
g4drop, b0drop = [], []
for mid, _, _ in MODELS:
    a, b = per.get(f"{mid}__plain"), per.get(f"{mid}__careful")
    if a and b:
        g4a, g4b = a[1]["G4"] / a[0]["claims"], b[1]["G4"] / b[0]["claims"]
        g4drop.append(100 * (1 - g4b / g4a))
        b0drop.append(abs(a[0]["methods"]["B0 no gate"]["hard_violation_pct"] - b[0]["methods"]["B0 no gate"]["hard_violation_pct"]))

aud_t = sum(c["text_numbers"] for c in C.values())
aud_c = sum(c["text_numbers_in_claims"] for c in C.values())
mac.update({
    "LAuditNumbers": n(aud_t), "LAuditMissed": n(aud_t - aud_c), "LAuditMissedPct": f"{100 * (aud_t - aud_c) / aud_t:.2f}",
    "LRequests": n(rep["requests"]), "LItems": n(rep["items"]), "LConfigs": str(len(C)),
    "LResponses": n(tot_resp), "LClaims": n(tot_claims), "LDis": n(dis),
    "LBzeroMin": rng("b0")[0], "LBzeroMax": rng("b0")[1], "LBtwoMin": rng("b2")[0], "LBtwoMax": rng("b2")[1],
    "LBthreeMin": rng("b3")[0], "LBthreeMax": rng("b3")[1],
    "LBfourMin": rng("b4", "{:.2f}")[0], "LBfourMax": rng("b4", "{:.2f}")[1],
    "LMissMin": rng("miss")[0], "LMissMax": rng("miss")[1],
    "LValMax": f"{max(agg['val']):.2f}", "LIntMin": rng("int")[0], "LIntMax": rng("int")[1],
    "LGfourMin": rng("g4")[0], "LGfourMax": rng("g4")[1], "LGtwoMin": rng("g2")[0], "LGtwoMax": rng("g2")[1],
    "LCovMin": floor1(min(agg['cov'])), "LGoneMax": ceil1(max(agg['g1'])), "LBfourGfiveMin": floor1(100 * min(agg['b4g5']), 0),
    "LGfiveMax": ceil1(max(agg['g5'])), "LMissFloor": floor1(min(agg['miss']), 0), "LMissCeil": ceil1(max(agg['miss']), 0),
    "LCarefulGfourDropMin": f"{min(g4drop):.0f}" if g4drop else "--", "LCarefulGfourDropMax": f"{max(g4drop):.0f}" if g4drop else "--",
    "LCarefulBzeroDropMax": ceil1(max(b0drop)) if b0drop else "--",
    "LGccMax": f"{max(agg['gcc']):.1f}", "LRetMin": f"{min(agg['ret']):.1f}",
})
with open(os.path.join(P, "llm_numbers.tex"), "w") as fh:
    for k, v in sorted(mac.items()):
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
with open(os.path.join(P, "tab_llm_rows.tex"), "w") as fh:
    fh.write("\n".join(rows) + "\n")
print(json.dumps(mac, indent=0))
print("\n".join(rows))
