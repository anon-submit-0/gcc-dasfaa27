"""Baseline boundary mechanisms.

B0  no gate: every generated claim is emitted as an unqualified fact.
B1  table-level validation (Deequ / Great-Expectations style): completeness,
    range, key-uniqueness and table-freshness checks on each source table at
    serving time; claims are blocked only if their source table fails.
B2  oracle post-hoc verifier: a claim is emitted iff its value is exactly
    entailed by the evidence the retriever returned for that item (all rows
    observed up to t).  This is an upper bound for citation/NLI checking.
B3  freshness-aware RAG verifier: as B2, but retrieval only returns offer
    rows inside the attribute's freshness window (30 d price, 60 d stock).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from common import DAY, TAU_AVAIL, TAU_PRICE, Olist


class Evidence:
    def __init__(self, ol: Olist):
        self.cat = dict(zip(ol.products.product_id, ol.products.category))
        self.spec = {r.product_id: r for r in ol.products.itertuples(index=False)}
        self.sstate = dict(zip(ol.sellers.seller_id, ol.sellers.state))
        self.off = defaultdict(list)
        for p, s, pr, ts in ol.offers[["product_id", "seller_id", "price", "observed_at"]].itertuples(index=False):
            self.off[p].append((int(ts), s, float(pr)))
        self.rev = defaultdict(list)
        for p, sc, ts in ol.reviews[["product_id", "score", "created_at"]].itertuples(index=False):
            self.rev[p].append((int(ts), int(sc)))
        self.city = defaultdict(list)
        for p, city, ts in ol.sales[["product_id", "city", "purchased_at"]].itertuples(index=False):
            self.city[(p, city)].append(int(ts))
        self.ol = ol

    def retrieve(self, p, t, window_days=None):
        lo = -1 if window_days is None else t - window_days * DAY
        return [r for r in self.off[p] if lo < r[0] <= t]


def b0(claims):
    return ["surface"] * len(claims)


def table_checks(ol: Olist, t: int) -> dict:
    o = ol.offers[ol.offers.observed_at <= t]
    pr = ol.products
    rv = ol.reviews[ol.reviews.created_at <= t]
    cs = ol.sales[ol.sales.purchased_at <= t]
    return {
        "offer_obs": bool(o.price.notna().all() and (o.price > 0).all() and o.observed_at.max() >= t - 7 * DAY),
        "product": bool(pr.product_id.is_unique and pr.category.notna().mean() >= 0.95 and (pr.weight_g.dropna() >= 0).all()),
        "seller": bool(ol.sellers.seller_id.is_unique and ol.sellers.state.notna().all()),
        "review_obs": bool(rv.score.between(1, 5).all() and rv.product_id.notna().all()),
        "city_sales": bool(cs.city.notna().all() and cs.product_id.notna().all() and cs.purchased_at.max() >= t - 7 * DAY),
    }


SOURCE = {"category": "product", "weight": "product", "dims": "product", "price_offer": "offer_obs",
          "price_product": "offer_obs", "ships_from": "seller", "avail_local": "offer_obs", "rating": "review_obs",
          "local_popularity": "city_sales", "rank_reason": "offer_obs", "marketing_claim": "product"}


def b1(ol: Olist, claims):
    cache = {}
    out = []
    for c in claims:
        if c["t"] not in cache:
            cache[c["t"]] = table_checks(ol, c["t"])
        out.append("surface" if cache[c["t"]][SOURCE[c["field"]]] else "refuse")
    return out, cache


def _supported(ev: Evidence, c, window: bool) -> bool:
    f, p, t, v = c["field"], c["product_id"], c["t"], c["value"]
    sp = ev.spec[p]
    if f == "category":
        return ev.cat[p] == v
    if f == "weight":
        return not np.isnan(sp.weight_g) and int(sp.weight_g) == int(v)
    if f == "dims":
        d = (sp.length_cm, sp.height_cm, sp.width_cm)
        return not any(np.isnan(x) for x in d) and "x".join(str(int(x)) for x in d) == v
    if f in ("price_offer", "price_product"):
        rows = ev.retrieve(p, t, TAU_PRICE if window else None)
        return any(abs(pr - float(v)) < 0.005 for _, _, pr in rows)
    if f in ("ships_from",):
        rows = ev.retrieve(p, t, None)
        return any(ev.sstate.get(s) == v for _, s, _ in rows)
    if f == "avail_local":
        rows = ev.retrieve(p, t, TAU_AVAIL if window else None)
        return any(ev.sstate.get(s) == v for _, s, _ in rows)
    if f == "rating":
        rs = [s for ts, s in ev.rev[p] if ts <= t]
        if not rs:
            return False
        va, vn = v.split("|")
        return abs(round(sum(rs) / len(rs), 1) - float(va)) < 1e-9 and int(vn) == len(rs)
    if f == "local_popularity":
        return sum(1 for ts in ev.city[(p, c["key"])] if ts <= t) == int(v)
    if f == "rank_reason":
        return sum(1 for ts, _, _ in ev.off[p] if t - 90 * DAY < ts <= t) == int(v)
    return False  # free-text marketing claims have no supporting evidence


def b2(ev: Evidence, claims):
    return ["surface" if _supported(ev, c, False) else "refuse" for c in claims]


def b3(ev: Evidence, claims):
    return ["surface" if _supported(ev, c, True) else "refuse" for c in claims]


# B4: B3 plus a field allowlist -- only fields registered as external facts may be emitted.
EXTERNAL_FIELDS = {"category", "price_offer", "price_product", "ships_from", "avail_local", "weight", "dims", "rating"}


def b4(ev: Evidence, claims):
    return ["surface" if (c["field"] in EXTERNAL_FIELDS and _supported(ev, c, True)) else "refuse" for c in claims]
