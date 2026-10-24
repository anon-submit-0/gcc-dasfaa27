# Grounded Consumption Contracts: Reviewer Artifact

This anonymized artifact accompanies the paper "Grounded Consumption Contracts:
Binding LLM-Generated Claims to Governed Data before They Are Served". It
reproduces the evaluation results for both boundaries.

| Directory | Boundary | What it reproduces |
| --- | --- | --- |
| `production/` | P: anonymized production pilot | Table 2 (claim-grounding gap and outcomes by evidence tier), G1-G5 specificity matrix, replay summary |
| `olist_gcc/` | O: public Olist-GCC benchmark | Table 1 (claim fields), Table 3 (baselines and ablations), Figure 2 (fault-rate sweep), Table 4 (overhead), Table 5 (LLM verbalizers) |

## Quick start

Boundary P uses only the Python standard library and the shipped files:

```bash
cd production && python3 reproduce.py      # writes results/reproduced_results.json
```

Boundary O needs Python 3.10+ with pandas and NumPy, and the public
Olist dataset (MIT license), which the fetch script downloads from GitHub:

```bash
cd olist_gcc
pip install -r requirements.txt
bash fetch_olist.sh                         # clones olist/work-at-olist-data into ./olist-data
python3 run_eval.py --rpw 150 --sweep       # Table 3 and Figure 2 data (about 40 min on 2 vCPUs)
python3 overhead.py results 75,150,300,600  # Table 4
python3 dump_claims.py                      # Table 1, per-claim labels and outcomes, text-level checks
python3 llm_verbaliser.py evaluate          # Table 5 from the shipped LLM outputs (llm/)
```

Precomputed outputs of these commands are in `olist_gcc/results/`.

## Anonymization

No author, organization, product, brand, customer or host identifiers appear in
this artifact. See `production/ANONYMIZATION.md` for the release boundary of the
production data. Boundary O uses only public data.
