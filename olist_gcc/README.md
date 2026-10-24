# Olist-GCC Benchmark (Boundary O)

A reproducible benchmark for construction-time grounding of generated claims
over a public relational e-commerce database (Olist; 32,951 products, 3,095
sellers, timestamped order items, reviews and customer locations).

## Components

| File | Role |
| --- | --- |
| `common.py` | Loads Olist into products, sellers, offer observations, reviews and city-level sales; contract windows |
| `workload.py` | Serving simulator: 34 weekly instants (Jan-Aug 2018), candidate -> eligible -> ranked funnel, claim generator and fault injection |
| `gate.py` | The GCC gate: governed SQLite substrate plus the three additive tables `claim_registry`, `evidence_binding`, `coverage_ledger`; ablation flags |
| `oracle.py` | Independent ground-truth labeler (pandas/NumPy; shares only the data loader and contract constants in `common.py` with the gate, and no SQL or registry metadata) |
| `baselines.py` | B0 no gate, B1 table-level validation, B2 oracle post-hoc verifier, B3 freshness-aware RAG verifier, B4 = B3 plus a field allowlist |
| `run_eval.py` | Main comparison, ablations and fault-rate sweep |
| `overhead.py` | Online-path latency, throughput and ledger storage |
| `dump_claims.py` | Per-claim CSV and the data checks quoted in the paper text |
| `make_numbers.py` | Converts result files into the paper's LaTeX macros, table rows and figure data |
| `llm_verbaliser.py` | RQ5: builds the retrieved-data contexts shown to the LLMs, the two prompts, and maps the models' structured claims onto benchmark claims; text-number audit |
| `llm_eval.py` | RQ5: judges the LLM claims with the labeler, gate, baselines and ablations |
| `make_llm_numbers.py` | RQ5: LaTeX macros and rows of Table 5 |
| `llm/` | RQ5 inputs and raw model outputs (see `llm/README.md`) |

## Contract

| Claim field | Grain | Outcome when admissible | Caliber |
| --- | --- | --- | --- |
| `price_offer`, `price_product` | offer / product (roll-up must be single-valued over fresh offers) | surface | 30 days |
| `avail_local` | product x customer state | surface | 60 days |
| `rating` | product | surface (caveat if n < 5 or latest review > 365 days) | 365 days |
| `category`, `weight`, `dims`, `ships_from` | product / seller | caveat (governed, no per-fact clock) | none |
| `rank_reason` | product | degrade (internal context) | - |
| `local_popularity` | product x customer city | refuse (customer-derived, sensitive) | - |
| any other field | - | refuse (unregistered) | - |

## Faults

Data-driven: stale prices and availability, multi-seller price divergence at
product grain. Template-driven: social-proof sentences and ranking rationales.
Injected (configurable in
`workload.Faults`): value perturbation `h`, cross-item mix-up `m`, unregistered
marketing claims `u`, assembly (lineage) faults `l`.

## LLM verbalizers (RQ5)

Three LLMs replace the template generator on 170 requests (5 per serving week).
The raw outputs are shipped, so no model access is needed to re-evaluate:

```bash
python3 llm_verbaliser.py evaluate     # -> results/llm_eval.json, results/llm_claims.csv.gz (Table 5)
```

## Results

`results/main.json` (Table 3), `results/sweep.json` (Figure 2),
`results/overhead.json` (Table 4), `results/llm_eval.json` (Table 5), `results/platform.json` (VM and Boundary-P
replay footprint) and `results/organic_checks.json`. `python3 dump_claims.py`
regenerates `results/claims_main.csv.gz` (one row per claim with its licensed
outcome, cause and every method's outcome; 2.7 MB, deterministic), which
`make_numbers.py` needs for Table 1.
