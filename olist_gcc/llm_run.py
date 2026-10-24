"""Issue the RQ5 calls against any OpenAI-compatible chat-completions endpoint.

    export LLM_BASE_URL=https://<host>/v1  LLM_API_KEY=<key>
    python3 llm_run.py deepseek-flash plain       # appends to llm/generations/run_<model>__<variant>.jsonl

The parameters are those of the shipped generations: temperature 0.7, max_tokens 8000, JSON
response format, system prompt `system_prompt(variant)`, user message `prompt_for(ctx)`, one call per
request of the evaluated subset.  Finished requests are skipped, so the script can be re-run until every
call has succeeded.  (The shipped outputs were produced with an equivalent JavaScript client.)
"""
from __future__ import annotations

import concurrent.futures as cf
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request

from llm_verbaliser import OUT, evaluated_subset, prompt_for, system_prompt


def call(base, key, body, timeout=600):
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    model, variant = sys.argv[1], sys.argv[2]
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    base, key = os.environ["LLM_BASE_URL"], os.environ["LLM_API_KEY"]
    f = os.path.join(OUT, "contexts.json")
    fh = open(f, encoding="utf-8") if os.path.exists(f) else gzip.open(f + ".gz", "rt", encoding="utf-8")
    ctxs = evaluated_subset(json.load(fh))
    path = os.path.join(OUT, "generations", f"run_{model}__{variant}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["req"] for l in open(path, encoding="utf-8") if l.strip()}
    todo = [c for c in ctxs if c["id"] not in done]

    def one(ctx):
        body = {"model": model, "temperature": 0.7, "max_tokens": 8000, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system_prompt(variant)},
                             {"role": "user", "content": prompt_for(ctx)}]}
        t0 = time.time()
        for attempt in range(6):
            try:
                j = call(base, key, body)
                m = j["choices"][0]["message"]
                return {"config": f"{model}__{variant}", "req": ctx["id"], "model": j.get("model", model),
                        "variant": variant, "temperature": 0.7, "text": (m.get("content") or "").strip(),
                        "usage": j.get("usage"), "finish_reason": j["choices"][0].get("finish_reason"),
                        "seconds": round(time.time() - t0, 1), "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError):
                time.sleep(min(60, 3 * 2 ** attempt))
        return None

    with open(path, "a", encoding="utf-8") as fo, cf.ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, todo):
            if r is not None:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                fo.flush()
    n = sum(1 for _ in open(path, encoding="utf-8"))
    print(f"{n}/{len(ctxs)} requests done -> {path}")


if __name__ == "__main__":
    main()
