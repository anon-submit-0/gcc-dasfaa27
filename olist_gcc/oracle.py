"""Independent ground-truth labeller.

Implements the contract semantics directly over pandas/NumPy structures built
from the raw Olist tables.  It shares no code with gate.py (no SQL, no
registry tables) so that agreement between the two is evidence that the
compiled gate implements the declared contract, not a tautology.

Each claim receives
  licensed  - the strongest outcome the contract permits
              (surface > caveat > degrade > refuse)
  cause     - the primary reason the licensed outcome is below `surface`
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from common import DAY, ELIGIBLE_DAYS, MIN_RATING_N, TAU_AVAIL, TAU_PRICE, TAU_RATING, Olist

REGISTERED = {"category", "price_offer", "price_product", "ships_from", "avail_local", "weight", "dims",
              "rating", "local_popularity", "rank_reason"}


class Oracle:
    def __init__(self, ol: Olist):
        self.cat = dict(zip(ol.products.product_id, ol.products.category))
        self.spec = {r.product_id: r for r in ol.products.itertuples(index=False)}
        self.sstate = dict(zip(ol.sellers.seller_id, ol.sellers.state))
        g = ol.offers.groupby(["product_id", "seller_id"])
        self.ps = {k: (v.observed_at.to_numpy(), v.price.to_numpy()) for k, v in g}
        self.sellers_of = defaultdict(list)
        for (p, s) in self.ps:
            self.sellers_of[p].append(s)
        gp = ol.offers.groupby("product_id").observed_at
        self.p_ts = {k: np.sort(v.to_numpy()) for k, v in gp}
        gr = ol.reviews.sort_values("created_at").groupby("product_id")
        self.rv = {k: (v.created_at.to_numpy(), np.cumsum(v.score.to_numpy())) for k, v in gr}

    def _latest(self, p, s, t):
        ts, pr = self.ps[(p, s)]
        i = np.searchsorted(ts, t, side="right") - 1
        return (None, None) if i < 0 else (int(ts[i]), float(pr[i]))

    def _eligible(self, p, t, req_cat):
        ts = self.p_ts.get(p)
        if ts is None or self.cat.get(p) != req_cat:
            return False
        i = np.searchsorted(ts, t, side="right") - 1
        return i >= 0 and ts[i] > t - ELIGIBLE_DAYS * DAY

    def label(self, c: dict, req_cat: str) -> tuple[str, str]:
        f, p, t, v = c["field"], c["product_id"], c["t"], c["value"]
        if f not in REGISTERED:
            return "refuse", "G1"
        if not self._eligible(p, t, req_cat):
            return "refuse", "G5"
        if f == "local_popularity":
            return "refuse", "G4"
        if f == "rank_reason":
            return "degrade", "internal"
        sp = self.spec[p]
        if f == "category":
            return ("caveat", "needs_caveat") if self.cat[p] == v else ("refuse", "VAL")
        if f == "weight":
            ok = not np.isnan(sp.weight_g) and int(sp.weight_g) == int(v)
            return ("caveat", "needs_caveat") if ok else ("refuse", "VAL")
        if f == "dims":
            d = (sp.length_cm, sp.height_cm, sp.width_cm)
            ok = not any(np.isnan(x) for x in d) and "x".join(str(int(x)) for x in d) == v
            return ("caveat", "needs_caveat") if ok else ("refuse", "VAL")
        if f == "ships_from":
            return ("caveat", "needs_caveat") if self.sstate.get(c["key"]) == v else ("refuse", "VAL")
        if f == "price_offer":
            ts, pr = self._latest(p, c["key"], t)
            if ts is None or abs(pr - float(v)) >= 0.005:
                return "refuse", "VAL"
            return ("surface", "ok") if t - ts <= TAU_PRICE * DAY else ("refuse", "G2")
        if f == "price_product":
            latest = [self._latest(p, s, t) for s in self.sellers_of[p]]
            latest = [x for x in latest if x[0] is not None]
            fresh = {round(pr, 2) for ts, pr in latest if t - ts <= TAU_PRICE * DAY}
            any_match = any(abs(pr - float(v)) < 0.005 for _, pr in latest)
            if not any_match:
                return "refuse", "VAL"
            if not fresh:
                return "refuse", "G2"
            if len(fresh) > 1:
                return "refuse", "G3"
            return ("surface", "ok") if abs(next(iter(fresh)) - float(v)) < 0.005 else ("refuse", "VAL")
        if f == "avail_local":
            if v != c["c_state"]:
                return "refuse", "G3"
            best = None
            for s in self.sellers_of[p]:
                if self.sstate.get(s) != v:
                    continue
                ts, _ = self._latest(p, s, t)
                if ts is not None and (best is None or ts > best):
                    best = ts
            if best is None:
                return "refuse", "VAL"
            return ("surface", "ok") if t - best <= TAU_AVAIL * DAY else ("refuse", "G2")
        if f == "rating":
            if p not in self.rv:
                return "refuse", "VAL"
            ts, cs = self.rv[p]
            i = np.searchsorted(ts, t, side="right") - 1
            if i < 0:
                return "refuse", "VAL"
            n = int(i + 1)
            avg = round(float(cs[i]) / n, 1)
            va, vn = v.split("|")
            if abs(float(va) - avg) > 1e-9 or int(vn) != n:
                return "refuse", "VAL"
            if n < MIN_RATING_N:
                return "caveat", "needs_caveat"
            if t - ts[i] > TAU_RATING * DAY:
                return "caveat", "G2"
            return "surface", "ok"
        return "refuse", "G1"
