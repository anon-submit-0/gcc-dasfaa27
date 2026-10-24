# Anonymization and Release Boundary

This artifact is prepared for double-blind review. It contains only the
minimal evidence needed to reproduce the paper's reported tables and checks.

## Identifier Policy

All claim, snapshot, and item identifiers are sequential opaque IDs:

- `c000001`, `c000002`, ...
- `s000001`, `s000002`, ...
- `i000001`, `i000002`, ...

These identifiers do not encode product names, customer identifiers, source
database keys, local paths, hashes of private values, or author information.

## Data Excluded

The package excludes raw production records, raw product catalogs, prices,
customer data, order-level details, internal database dumps, local snapshot
paths, internal host names, private access material, Git history, manuscript drafts, cover
letters, and review-process notes.

## Released Evidence

The released evidence is limited to:

- aggregate table counts;
- anonymized claim-level disposition rows;
- field-level counts;
- disposition and replay summaries;
- G1-G5 perturbation outcomes.

## Known Scope Limits

The artifact demonstrates deterministic reproduction of the released
claim-disposition evidence. It does not reproduce the private warehouse, raw
ranking model execution, or raw natural-language generation. Replay is scoped to
the typed grounding decision state over the released claim set.
