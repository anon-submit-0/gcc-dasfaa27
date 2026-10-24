# Manifest

## Top-Level Files

- `README.md`: artifact overview and quick-start instructions.
- `ANONYMIZATION.md`: anonymization policy and release boundary.
- `MANIFEST.md`: this manifest.
- `requirements.txt`: dependency declaration.
- `reproduce.py`: standard-library reproduction script.

## Contracts

- `contracts/grounding_contracts.sql`: anonymized SQL DDL and seed rows for the
  three claim-grounding tables.

## Data Files

- `data/table_manifest.csv`: 19 rows; released aggregate table counts sum to
  10,141.
- `data/claim_dispositions.csv`: 121,760 surfaced claim rows.
- `data/claim_field_manifest.csv`: field-level claim counts.
- `data/disposition_summary.json`: disposition, evidence-tier, and operating
  point summary.
- `data/upstream_summary.json`: upstream clock and funnel-integrity summary.
- `data/assertion_specificity_matrix.json`: G1-G5 perturbation matrix.
- `data/replay_summary.json`: replay and ranking-auditability summary.

## Generated Output

- `results/reproduced_results.json`: produced by `python3 reproduce.py`.
