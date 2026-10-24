# Boundary P: Anonymized Production Release

This directory contains the anonymized release files for the production
recommendation boundary. They reproduce the claim-disposition tables, the
governance-drop witness, the G1-G5 specificity matrix and the replay summary.

## Quick start

```bash
python3 reproduce.py
```

The script uses only the Python standard library, reads only files shipped in
this directory and writes `results/reproduced_results.json`. No database,
network access or non-standard packages are required.

## Contents

- `contracts/grounding_contracts.sql`: the three-table claim-grounding contract
  (claim registry, claim-to-evidence binding, coverage ledger) with seed rows.
- `data/table_manifest.csv`: aggregate row counts for the 19-table snapshot
  (10,141 rows).
- `data/claim_dispositions.csv`: 121,760 anonymized claim rows with sequential
  opaque claim, snapshot and item identifiers.
- `data/claim_field_manifest.csv`: field-level claim counts.
- `data/disposition_summary.json`: pre-gate provenance gap, post-gate
  dispositions, evidence tiers and operating points.
- `data/assertion_specificity_matrix.json`: G1-G5 perturbation matrix.
- `data/replay_summary.json`: version-pin completeness, typed replay and the
  score-only ranking auditability boundary.
- `data/upstream_summary.json`: upstream clock and funnel-integrity aggregates.

## Expected headline checks

- 121,760 surfaced claims; 121,760 / 121,760 with empty pre-gate provenance and
  null pre-gate recorded time.
- Post-gate evidence tiers: 12,486 clock-backed facts, 12,486 governed identities
  without a per-fact clock (together the 24,972 `grounded` rows), 79,770
  governed specifications without a per-fact clock (`caveat`), 11,460 derived
  context and 5,558 unmapped/media rows (`degrade`); 104,742 / 121,760 = 86.02%
  receive a governed surface-or-caveat disposition. Under the strict operating
  point OP1 only the 12,486 clock-backed facts surface without qualification.
- 1,130 / 1,130 funnel monotonicity; clean G1-G5 specificity diagonal.
- 100% same-source typed replay; score-only ranking reproducibility for
  844 / 1,130 snapshots (74.69%), reported as an auditability boundary.

## Release boundary

The release excludes the private database, raw product records, prices,
customer and order identifiers, internal hosts and authoring history.
