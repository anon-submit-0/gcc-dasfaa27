"""Serving-layer simulator and claim generator for the Olist-GCC benchmark.

For each weekly serving instant t, the simulator draws recommendation
requests (a customer and a product category), assembles a
candidate -> eligible -> ranked funnel from data observed up to t, and
verbalises claims for every ranked item from the rows a retrieval step would
return (all offer observations, the product row, review rows and same-city
sales up to t).  The generator behaves like a typical template/RAG
verbaliser: it reads the most recent price row regardless of seller or age,
asserts local availability if any seller in the customer's state ever sold
the item, and adds social-proof and ranking-rationale sentences.

Model-style faults are injected at configurable rates:
  h  value perturbation of a numeric/categorical claim
  m  cross-item value mix-up inside one response
  u  unregistered free-text marketing claim per item
  l  lineage fault: a non-eligible item replaces a ranked item
"""
from __future__ import annotations

import bisect
import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import DAY, ELIGIBLE_DAYS, SALES_DAYS, TOP_K, Olist, serving_times


@dataclass
class Faults:
    h: float = 0.05
    m: float = 0.02
    u: float = 0.05
    l: float = 0.01


@dataclass
class Workload:
    claims: list = field(default_factory=list)       # dicts
    funnel: list = field(default_factory=list)       # (snapshot_id, product_id, stage)
    snapshots: list = field(default_factory=list)    # (snapshot_id, t, customer_state, customer_city, category)


class AsOf:
    """Per-product sorted observation arrays for fast as-of lookups."""

    def __init__(self, ol: Olist):
        self.prod = ol.products.set_index("product_id")
        self.seller_state = dict(zip(ol.sellers.seller_id, ol.sellers.state))
        self.offers = defaultdict(list)  # product -> list of (ts, seller, price)
        for p, s, pr, ts in ol.offers[["product_id", "seller_id", "price", "observed_at"]].itertuples(index=False):
            self.offers[p].append((int(ts), s, float(pr)))
        for p in self.offers:
            self.offers[p].sort()
        self.reviews = defaultdict(list)
        for p, sc, ts in ol.reviews[["product_id", "score", "created_at"]].itertuples(index=False):
            self.reviews[p].append((int(ts), int(sc)))
        for p in self.reviews:
            self.reviews[p].sort()
        self.city_sales = defaultdict(list)  # (product, city) -> sorted ts
        for p, city, ts in ol.sales[["product_id", "city", "purchased_at"]].itertuples(index=False):
            self.city_sales[(p, city)].append(int(ts))
        for k in self.city_sales:
            self.city_sales[k].sort()

    def offers_upto(self, p, t):
        rows = self.offers.get(p, [])
        i = bisect.bisect_right(rows, (t, "￿", float("inf")))
        return rows[:i]

    def rating(self, p, t):
        rows = self.reviews.get(p, [])
        i = bisect.bisect_right(rows, (t, 99))
        rs = rows[:i]
        if not rs:
            return None, 0, None
        return round(sum(s for _, s in rs) / len(rs), 1), len(rs), rs[-1][0]

    def city_count(self, p, city, t):
        rows = self.city_sales.get((p, city), [])
        return bisect.bisect_right(rows, t)


def _perturb(rng: random.Random, fieldname: str, value, ctx):
    if fieldname in ("price_offer", "price_product"):
        f = 1 + rng.choice([-1, 1]) * rng.uniform(0.1, 0.3)
        return f"{round(float(value) * f, 2):.2f}"
    if fieldname == "weight":
        f = 1 + rng.choice([-1, 1]) * rng.uniform(0.1, 0.3)
        return str(max(1, int(round(float(value) * f))))
    if fieldname == "dims":
        dims = [int(x) for x in value.split("x")]
        j = rng.randrange(3)
        dims[j] = max(1, int(round(dims[j] * (1 + rng.choice([-1, 1]) * rng.uniform(0.15, 0.4)))))
        return "x".join(map(str, dims))
    if fieldname == "rating":
        avg, n = value.split("|")
        a = float(avg) + rng.choice([-1, 1]) * rng.uniform(0.5, 1.0)
        return f"{min(5.0, max(1.0, round(a, 1))):.1f}|{n}"
    if fieldname == "category":
        return rng.choice([c for c in ctx["categories"] if c != value])
    if fieldname == "ships_from":
        return rng.choice([s for s in ctx["states"] if s != value])
    return value


PERTURBABLE = {"price_offer", "price_product", "weight", "dims", "rating", "category", "ships_from"}


def generate(ol: Olist, faults: Faults, requests_per_week: int = 150, seed: int = 7, asof: AsOf | None = None) -> Workload:
    rng = random.Random(seed)
    asof = asof or AsOf(ol)
    wl = Workload()
    offers = ol.offers
    prod_cat = dict(zip(ol.products.product_id, ol.products.category))
    categories = sorted(set(prod_cat.values()))
    states = sorted(set(ol.sellers.state))
    ctx = {"categories": categories, "states": states}
    off = offers.assign(category=offers.product_id.map(prod_cat))
    cid = 0
    sid = 0
    cust = ol.customers
    for t in serving_times():
        seen = off[off.observed_at <= t]
        last_obs = seen.groupby("product_id").observed_at.max()
        recent = seen[seen.observed_at > t - SALES_DAYS * DAY]
        sales_cnt = recent.groupby("product_id").size()
        cat_weight = recent.groupby("category").size()
        cats = list(cat_weight.index)
        weights = list(cat_weight.values)
        cand_by_cat = seen.groupby("category").product_id.unique()
        elig_cut = t - ELIGIBLE_DAYS * DAY
        pool = cust[cust.first_purchase <= t]
        cust_idx = rng.sample(range(len(pool)), requests_per_week)
        for k in range(requests_per_week):
            crow = pool.iloc[cust_idx[k]]
            c_state, c_city = crow.state, crow.city
            cat = rng.choices(cats, weights=weights, k=1)[0]
            cand = sorted(cand_by_cat.get(cat, []))
            elig = [p for p in cand if last_obs.get(p, 0) > elig_cut]
            ranked = sorted(elig, key=lambda p: (-int(sales_cnt.get(p, 0)), p))[:TOP_K]
            if not ranked:
                continue
            sid += 1
            snap = f"s{sid:06d}"
            wl.snapshots.append((snap, t, c_state, c_city, cat))
            for p in cand:
                wl.funnel.append((snap, p, "candidate"))
            for p in elig:
                wl.funnel.append((snap, p, "eligible"))
            shown = list(ranked)
            lineage_fault = [False] * len(shown)
            for i in range(len(shown)):
                if rng.random() < faults.l:
                    # a cached / stale item replaces the ranked one (not eligible at t)
                    stale = [p for p in cand if p not in elig]
                    other = stale if stale else [p for p in last_obs.index[:2000] if prod_cat.get(p) != cat]
                    other = [p for p in other if p not in shown]
                    if other:
                        shown[i] = rng.choice(other)
                        lineage_fault[i] = True
            for p in shown:
                wl.funnel.append((snap, p, "ranked"))
            resp_claims = []
            for rank, p in enumerate(shown, start=1):
                rows = asof.offers_upto(p, t)
                if not rows:
                    continue
                ts_last, s_last, pr_last = rows[-1]
                base = dict(snapshot_id=snap, t=t, rank=rank, product_id=p, c_state=c_state, c_city=c_city,
                            lineage_fault=lineage_fault[rank - 1])
                item_claims = []
                prow = asof.prod.loc[p]
                item_claims.append(dict(field="category", grain="product", key="", value=str(prow.category)))
                if rng.random() < 0.5:
                    item_claims.append(dict(field="price_offer", grain="offer", key=s_last, value=f"{pr_last:.2f}"))
                else:
                    item_claims.append(dict(field="price_product", grain="product", key="", value=f"{pr_last:.2f}"))
                item_claims.append(dict(field="ships_from", grain="offer", key=s_last, value=asof.seller_state.get(s_last, "")))
                if any(asof.seller_state.get(s) == c_state for _, s, _ in rows):
                    item_claims.append(dict(field="avail_local", grain="product_state", key=c_state, value=c_state))
                if not pd.isna(prow.weight_g):
                    item_claims.append(dict(field="weight", grain="product", key="", value=str(int(prow.weight_g))))
                if not (pd.isna(prow.length_cm) or pd.isna(prow.height_cm) or pd.isna(prow.width_cm)):
                    item_claims.append(dict(field="dims", grain="product", key="",
                                            value=f"{int(prow.length_cm)}x{int(prow.height_cm)}x{int(prow.width_cm)}"))
                avg, n, _ = asof.rating(p, t)
                if n > 0:
                    item_claims.append(dict(field="rating", grain="product", key="", value=f"{avg:.1f}|{n}"))
                cc = asof.city_count(p, c_city, t)
                if cc >= 1:
                    item_claims.append(dict(field="local_popularity", grain="product_city", key=c_city, value=str(cc)))
                if rank == 1:
                    item_claims.append(dict(field="rank_reason", grain="product", key="",
                                            value=str(int(sales_cnt.get(p, 0)))))
                if rng.random() < faults.u:
                    item_claims.append(dict(field="marketing_claim", grain="product", key="",
                                            value=rng.choice(["2-year warranty", "eco-friendly materials",
                                                              "free returns", "certified original"])))
                for c in item_claims:
                    c.update(base)
                    c["injected"] = "lineage" if base["lineage_fault"] else ("unregistered" if c["field"] == "marketing_claim" else "none")
                    c["true_value"] = c["value"]
                resp_claims.extend(item_claims)
            # value faults (perturbation and cross-item mix-up)
            for c in resp_claims:
                if c["field"] not in PERTURBABLE or c["injected"] != "none":
                    continue
                r = rng.random()
                if r < faults.h:
                    nv = _perturb(rng, c["field"], c["value"], ctx)
                    if nv != c["value"]:
                        c["value"], c["injected"] = nv, "perturb"
                elif r < faults.h + faults.m:
                    peers = [d for d in resp_claims if d["field"] == c["field"] and d["product_id"] != c["product_id"]
                             and d["true_value"] != c["value"]]
                    if peers:
                        c["value"], c["injected"] = rng.choice(peers)["true_value"], "mixup"
            for c in resp_claims:
                cid += 1
                c["claim_id"] = f"c{cid:07d}"
                wl.claims.append(c)
    return wl
