"""Real-LLM verbaliser for the Olist-GCC benchmark.

Instead of the template generator, a large language model writes the item blurbs
for a sample of serving requests and reports each factual statement it made as a
structured claim.  The claims are then judged by the same oracle, gate, baselines
and ablations as the template workload.

    python3 llm_verbaliser.py contexts            # -> llm/contexts.json  (what the model sees)
    # run the calls (llmhub skill; see llm/README.md) -> llm/generations/<config>.jsonl
    python3 llm_verbaliser.py evaluate            # -> results/llm_eval.json

The prompt text and the retrieved-data serialisation are fixed here; `prompt_for`
rebuilds the exact user message sent for each request.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

from common import DAY, SALES_DAYS, load_olist
from workload import AsOf, Faults, generate

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm")
MAX_OFFER_ROWS = 6          # the retriever returns the six most recent offer observations

SYSTEM = {
    "plain": (
        "You write the product blurbs shown next to recommendations in a Brazilian online marketplace "
        "(prices in BRL). For each recommended item write one or two sentences for this customer, using the "
        "retrieved data. Make the copy informative and persuasive: shoppers respond to prices, where an item "
        "ships from, whether it is available in their state, size and weight, ratings and popularity."),
    "careful": (
        "You write the product blurbs shown next to recommendations in a Brazilian online marketplace "
        "(prices in BRL). For each recommended item write one or two sentences for this customer, using the "
        "retrieved data. Make the copy informative and persuasive: shoppers respond to prices, where an item "
        "ships from, whether it is available in their state, size and weight, ratings and popularity. "
        "Only state facts that the retrieved data supports for this customer at the given date. Offers may be "
        "old or come from different sellers, and some data may not be appropriate to show customers; when "
        "unsure, leave the fact out."),
}

FORMAT = (
    "\n\nReturn only JSON: {\"items\":[{\"item\":\"P1\",\"text\":\"...\",\"claims\":[...]}]}. "
    "List in \"claims\" every factual statement your text makes, one object per statement:\n"
    "{\"field\":\"category\",\"value\":\"<category>\"}\n"
    "{\"field\":\"price\",\"value\":<number>,\"seller\":\"S1\"}  (omit \"seller\" for a price stated for the item in general)\n"
    "{\"field\":\"ships_from\",\"value\":\"<state code>\",\"seller\":\"S1\"}\n"
    "{\"field\":\"available_in_state\",\"value\":\"<state code>\"}\n"
    "{\"field\":\"weight_g\",\"value\":<integer>}\n"
    "{\"field\":\"dimensions_cm\",\"value\":\"LxHxW\"}\n"
    "{\"field\":\"rating\",\"value\":{\"avg\":<number>,\"count\":<integer>}}\n"
    "{\"field\":\"local_popularity\",\"value\":<integer>}  (purchases by customers in the customer's city)\n"
    "{\"field\":\"recent_sales\",\"value\":<integer>}  (units sold in the last 90 days)\n"
    "{\"field\":\"other\",\"value\":\"<the statement>\"}  (any other factual statement, e.g. delivery, warranty, quality)")


def _d(ts: int) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def _jnum(x):
    """Integral floats as ints so Python and JavaScript serialise the view identically."""
    return int(x) if float(x).is_integer() else x


def system_prompt(variant: str) -> str:
    return SYSTEM[variant] + FORMAT


def prompt_for(ctx: dict) -> str:
    """User message: identical to JSON.stringify(ctx.view) in the browser runner."""
    return "Retrieved data:\n" + json.dumps(ctx["view"], ensure_ascii=False, separators=(",", ":"))


def build_contexts(ol, rpw=150, seed=7, per_week=10, sample_seed=2027):
    asof = AsOf(ol)
    # Assembly faults are kept (they are pipeline faults, not verbaliser faults); value faults are off:
    # the model makes its own mistakes.
    wl = generate(ol, Faults(h=0.0, m=0.0, u=0.0, l=0.01), requests_per_week=rpw, seed=seed, asof=asof)
    shown = defaultdict(list)
    for snap, p, stage in wl.funnel:
        if stage == "ranked":
            shown[snap].append(p)
    lineage = {(c["snapshot_id"], c["product_id"]): c["lineage_fault"] for c in wl.claims}
    by_week = defaultdict(list)
    for s in wl.snapshots:
        by_week[s[1]].append(s)
    rng = random.Random(sample_seed)
    picked = []
    for t in sorted(by_week):
        cand = sorted(by_week[t])
        picked += rng.sample(cand, min(per_week, len(cand)))
    off = ol.offers
    ctxs = []
    for snap, t, c_state, c_city, cat in picked:
        items, sel_alias, meta = [], {}, []
        for i, p in enumerate(shown[snap], start=1):
            rows = asof.offers_upto(p, t)
            if not rows:
                continue
            pr = asof.prod.loc[p]
            dims = None
            if not any(str(x) == "nan" for x in (pr.length_cm, pr.height_cm, pr.width_cm)):
                dims = f"{int(pr.length_cm)}x{int(pr.height_cm)}x{int(pr.width_cm)}"
            orows = []
            for ts, s, price in sorted(rows, reverse=True)[:MAX_OFFER_ROWS]:
                a = sel_alias.setdefault(s, f"S{len(sel_alias) + 1}")
                orows.append({"seller": a, "seller_state": asof.seller_state.get(s, ""), "price": _jnum(round(price, 2)),
                              "date": _d(ts)})
            avg, n, last = asof.rating(p, t)
            recent = sum(1 for ts, _, _ in rows if ts > t - SALES_DAYS * DAY)
            alias = f"P{len(items) + 1}"
            items.append({"item": alias, "category": str(pr.category),
                          "weight_g": None if str(pr.weight_g) == "nan" else int(pr.weight_g),
                          "dimensions_cm": dims, "offers": orows,
                          "reviews": None if not n else {"avg": _jnum(avg), "count": n, "last": _d(last)},
                          "units_bought_in_customer_city": asof.city_count(p, c_city, t),
                          "units_sold_last_90_days": recent})
            meta.append({"alias": alias, "product_id": p, "rank": i, "lineage_fault": lineage.get((snap, p), False)})
        view = {"date": _d(t), "customer": {"city": c_city, "state": c_state}, "requested_category": cat,
                "items": items}
        ctxs.append({"id": snap, "t": t, "c_state": c_state, "c_city": c_city, "category": cat, "view": view,
                     "items": meta, "sellers": {v: k for k, v in sel_alias.items()}})
    return ctxs, wl


def evaluated_subset(ctxs):
    """The calls cover 5 of the 10 sampled requests per week (the first five of each week's random draw)."""
    return [c for i, c in enumerate(ctxs) if i % 10 < 5]


# ---------------------------------------------------------------- parsing model output
def parse_json(text: str):
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, k = t.find("{"), t.rfind("}")
        if 0 <= i < k:
            return json.loads(t[i:k + 1])
        raise


def _num(v):
    if isinstance(v, bool):
        raise ValueError
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace("R$", "").replace(" ", "")
    if re.fullmatch(r"\d{1,3}(\.\d{3})*,\d+", s):      # 1.234,56
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d+,\d+", s):
        s = s.replace(",", ".")
    return float(s)


def to_claims(ctx: dict, parsed: dict, stats: Counter) -> list[dict]:
    """Map a model's structured claims onto benchmark claims (field, grain, key, value)."""
    pid = {m["alias"]: m for m in ctx["items"]}
    out = []
    for it in parsed.get("items", []) or []:
        m = pid.get(str(it.get("item", "")).strip())
        if m is None:
            stats["unknown_item"] += 1
            continue
        for c in it.get("claims", []) or []:
            f = str(c.get("field", "")).strip()
            v = c.get("value")
            seller = ctx["sellers"].get(str(c.get("seller") or "").strip(), "")
            rec = None
            try:
                if f == "category":
                    rec = ("category", "product", "", str(v))
                elif f == "price":
                    rec = ("price_offer", "offer", seller, f"{_num(v):.2f}") if seller else \
                        ("price_product", "product", "", f"{_num(v):.2f}")
                elif f == "ships_from":
                    rec = ("ships_from", "offer", seller, str(v).strip().upper())
                elif f == "available_in_state":
                    s = str(v).strip().upper()
                    rec = ("avail_local", "product_state", s, s)
                elif f == "weight_g":
                    rec = ("weight", "product", "", str(int(round(_num(v)))))
                elif f == "dimensions_cm":
                    d = [int(round(float(x))) for x in re.findall(r"\d+(?:\.\d+)?", str(v))]
                    if len(d) != 3:
                        raise ValueError
                    rec = ("dims", "product", "", "x".join(map(str, d)))
                elif f == "rating":
                    if isinstance(v, dict):
                        a, n = _num(v.get("avg")), int(_num(v.get("count")))
                    else:
                        a, n = _num(v), -1
                    rec = ("rating", "product", "", f"{round(a, 1):.1f}|{n}")
                elif f == "local_popularity":
                    rec = ("local_popularity", "product_city", ctx["c_city"], str(int(round(_num(v)))))
                elif f == "recent_sales":
                    rec = ("rank_reason", "product", "", str(int(round(_num(v)))))
                elif f == "other":
                    rec = ("marketing_claim", "product", "", str(v))
                else:
                    stats["unknown_field"] += 1
                    rec = ("marketing_claim", "product", "", f"{f}: {v}")
            except (TypeError, ValueError):
                stats["unparseable_value"] += 1
                rec = ("marketing_claim", "product", "", f"{f}: {v}")
            field, grain, key, value = rec
            out.append(dict(snapshot_id=ctx["id"], t=ctx["t"], rank=m["rank"], product_id=m["product_id"],
                            c_state=ctx["c_state"], c_city=ctx["c_city"], lineage_fault=m["lineage_fault"],
                            field=field, grain=grain, key=key, value=value, true_value=value,
                            injected="lineage" if m["lineage_fault"] else "llm", text=it.get("text", "")))
    return out


_TOK = re.compile(r"(?<![A-Za-z\d.,])\d+(?:[.,]\d+)*(?:\s*x\s*\d+(?:[.,]\d+)*){0,2}")


def _to_float(x: str) -> float:
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", x):
        return float(x.replace(".", "").replace(",", "."))
    return float(x.replace(",", "."))


def text_audit(ctx, parsed, uncovered=None):
    """Numbers written in the blurbs that are not backed by one of the item's reported claims.

    Normalisation: dimension strings (AxBxC) compare as a whole, kilograms are converted to grams,
    kilograms rounded to the shown precision count as the gram value, item and seller aliases (P1, S2)
    are not numbers, and fixed phrases carry no claim value: the 90-day window ("90 dias", "3 meses"),
    the rating scale ("/5", "de 5", "out of 5") and calendar years.  Returns (numbers, covered)."""
    tot = cov = 0
    for it in parsed.get("items", []) or []:
        claims = it.get("claims", []) or []
        vals = json.dumps(claims, ensure_ascii=False)
        nums = {float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", vals)}
        dims = {"x".join(str(int(round(float(y)))) for y in re.findall(r"\d+(?:\.\d+)?", str(c.get("value"))))
                for c in claims if c.get("field") == "dimensions_cm"}
        text = str(it.get("text", ""))
        for m in _TOK.finditer(text):
            tok = m.group(0)
            pre, post = text[max(0, m.start() - 4):m.start()], text[m.end():m.end() + 8].lower()
            tot += 1
            if "x" in tok:
                parts = [p.strip() for p in tok.split("x")]
                try:
                    ok = "x".join(str(int(round(_to_float(p)))) for p in parts) in dims
                except ValueError:
                    ok = False
                cov += ok
                if not ok and uncovered is not None:
                    uncovered.append(text[max(0, m.start() - 30):m.end() + 20])
                continue
            try:
                f = _to_float(tok)
            except ValueError:
                continue
            fixed = (f == 90 and re.match(r"\s*(dias|days|d\b)", post)) or (f == 3 and re.match(r"\s*(meses|months)", post)) or \
                    (f == 5 and (pre.endswith("/") or re.search(r"(de|of)\s$", pre))) or (1990 <= f <= 2030 and "," not in tok)
            ok = bool(fixed) or any(abs(f - y) < 0.006 for y in nums)
            if not ok and re.match(r"\s*kg", post):
                dec = len(tok.replace(".", ",").split(",")[1]) if "," in tok or "." in tok else 0
                ok = any(abs(f * 1000 - y) <= 500 * 10 ** -dec + 1e-6 for y in nums)  # rounded kilograms
            cov += ok
            if not ok and uncovered is not None:
                uncovered.append(text[max(0, m.start() - 30):m.end() + 20])
    return tot, cov


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "contexts"
    os.makedirs(OUT, exist_ok=True)
    ol = load_olist()
    ctxs, wl = build_contexts(ol)
    if cmd == "contexts":
        json.dump(ctxs, open(os.path.join(OUT, "contexts.json"), "w"), ensure_ascii=False)
        print(len(ctxs), "requests,", sum(len(c["items"]) for c in ctxs), "items")
        return
    if cmd == "evaluate":
        from llm_eval import evaluate
        evaluate(ol, wl, evaluated_subset(ctxs))


if __name__ == "__main__":
    main()
