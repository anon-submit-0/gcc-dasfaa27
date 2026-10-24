# Real-LLM verbalizer run (Boundary O, RQ5)

| File | Content |
|---|---|
| `contexts.json.gz` | 340 requests sampled from the RQ2 workload (10 per serving week, seed 2027). Each entry holds the retrieved data shown to the model (`view`), the mapping from item and seller aliases to Olist ids, and the request context. The LLM calls cover the first five requests of each week (170 requests, `llm_verbaliser.evaluated_subset`). Rebuilt by `python3 llm_verbaliser.py contexts`. |
| `generations/llm_generations.jsonl.gz` | One line per call: `config`, `req` (snapshot id), `model`, `variant`, `temperature`, raw model output `text`, token `usage`, `finish_reason`, wall-clock `seconds` (including gateway queueing), `date`. |

Prompts: `llm_verbaliser.SYSTEM` (variants `plain` and `careful`) followed by `FORMAT`. The
user message is `"Retrieved data:\n"` plus the compact JSON of `view`
(`llm_verbaliser.prompt_for`). Every call used `temperature=0.7`, `max_tokens=8000` and the JSON
response format, through an OpenAI-compatible chat-completions endpoint, in October 2026.
Models: `deepseek-flash`, `gpt-6-sol`, `glm-5.3`. Calls that failed (timeouts, and a temporary
provider outage of the GLM models) were re-sent until all 1,020 calls (3 models x 2 prompts x 170
requests) had returned; the file holds exactly one successful response per call.

Re-evaluate the shipped generations (no model access needed):

```bash
python3 llm_verbaliser.py evaluate     # -> results/llm_eval.json, results/llm_claims.csv.gz
```

To regenerate with another endpoint, send `system_prompt(variant)` and `prompt_for(ctx)` for each
context in the evaluated subset and write the same JSONL fields.
