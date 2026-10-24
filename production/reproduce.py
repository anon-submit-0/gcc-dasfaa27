#!/usr/bin/env python3
"""Reproduce the Grounded Consumption Contracts artifact summaries.

Command:
  python3 reproduce.py

The script uses only Python's standard library and files shipped in this
directory. It writes results/reproduced_results.json.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def pct(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def load_json(name: str):
    with (DATA / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reproduce_table_manifest():
    total = 0
    rows = []
    with (DATA / "table_manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            count = int(row["released_rows"])
            total += count
            rows.append({"table_name": row["table_name"], "released_rows": count})
    return {"tables": len(rows), "released_rows": total, "manifest": rows}


def reproduce_claim_dispositions():
    dispositions = Counter()
    tiers = Counter()
    fields = Counter()
    provenance_empty = 0
    recorded_null = 0
    external_truth = 0
    spec_claims = 0
    spec_external = 0
    total = 0

    with (DATA / "claim_dispositions.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            field = row["source_field"]
            fields[field] += 1
            dispositions[row["post_gate_disposition"]] += 1
            tiers[row["evidence_tier"]] += 1
            provenance_empty += row["pre_provenance_empty"] == "true"
            recorded_null += row["pre_recorded_at_null"] == "true"
            has_external = row["external_product_truth_available"] == "true"
            external_truth += has_external
            if field == "specs":
                spec_claims += 1
                spec_external += has_external

    grounded = dispositions["grounded"]
    caveat = dispositions["caveat"]
    degrade = dispositions["degrade_derived"] + dispositions["degrade_unmapped"]
    return {
        "total_surfaced_claims": total,
        "pre_gate_empty_provenance": {
            "count": provenance_empty,
            "pct": pct(provenance_empty, total),
        },
        "pre_gate_null_recorded_at": {
            "count": recorded_null,
            "pct": pct(recorded_null, total),
        },
        "post_gate_dispositions": dict(dispositions),
        "evidence_tiers": dict(tiers),
        "binding_or_caveat_coverage": {
            "count": grounded + caveat,
            "pct": pct(grounded + caveat, total),
        },
        "degraded_non_asserted_context": {"count": degrade, "pct": pct(degrade, total)},
        "top_source_fields": fields.most_common(20),
        "external_truth_coverage": {"count": external_truth, "pct": pct(external_truth, total)},
        "spec_external_truth_coverage": {
            "count": spec_external,
            "denominator": spec_claims,
            "pct": pct(spec_external, spec_claims),
        },
    }


def reproduce_ranking_audit():
    rows = 0
    consistent = 0
    with (DATA / "ranking_audit.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            consistent += row["rank_score_consistent"] == "true"
    return {
        "snapshots_with_items": rows,
        "rank_score_consistent": consistent,
        "pct": pct(consistent, rows),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    reproduced = {
        "table_manifest": reproduce_table_manifest(),
        "claim_dispositions": reproduce_claim_dispositions(),
        "disposition_summary_file": load_json("disposition_summary.json"),
        "upstream_integrity": load_json("upstream_summary.json"),
        "assertion_specificity": load_json("assertion_specificity_matrix.json"),
        "replay_summary": load_json("replay_summary.json"),
        "ranking_audit_from_csv": reproduce_ranking_audit(),
    }
    out = RESULTS / "reproduced_results.json"
    out.write_text(json.dumps(reproduced, indent=2), encoding="utf-8")
    print(json.dumps(reproduced, indent=2))


if __name__ == "__main__":
    main()
