"""GCC grounding gate compiled over a SQLite governed substrate.

The gate is driven by three additive contract tables (claim_registry,
evidence_binding, coverage_ledger).  Witness resolution is ordinary
parameterised SQL against the governed tables; every evaluated claim writes
exactly one coverage-ledger row.  Ablation flags disable one admissibility
conjunct at a time.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from common import DAY, Olist

OUTCOMES = ["refuse", "degrade", "caveat", "surface"]
RANK = {o: i for i, o in enumerate(OUTCOMES)}

REGISTRY = [
    # claim_field, claim_type, grain, external, surface_policy, sensitivity
    ("category", "identity", "product", 1, "caveat", "low"),
    ("price_offer", "physical", "offer", 1, "surface_if_fresh", "low"),
    ("price_product", "physical", "product", 1, "surface_if_fresh", "low"),
    ("ships_from", "identity", "offer", 1, "caveat", "low"),
    ("avail_local", "physical", "product_state", 1, "surface_if_fresh", "low"),
    ("weight", "spec", "product", 1, "caveat", "low"),
    ("dims", "spec", "product", 1, "caveat", "low"),
    ("rating", "aggregate", "product", 1, "surface_if_fresh", "low"),
    ("local_popularity", "aggregate", "product_city", 0, "refuse", "high"),
    ("rank_reason", "internal", "product", 0, "degrade", "low"),
]

BINDING = [
    # claim_field, evidence_table, value_column, key_columns, clock_column, tau_days, min_n
    ("category", "product", "category", "product_id", None, None, None),
    ("price_offer", "offer_obs", "price", "product_id,seller_id", "observed_at", 30, None),
    ("price_product", "offer_obs", "price", "product_id", "observed_at", 30, None),
    ("ships_from", "seller", "state", "seller_id", None, None, None),
    ("avail_local", "offer_obs JOIN seller", "state", "product_id,state", "observed_at", 60, None),
    ("weight", "product", "weight_g", "product_id", None, None, None),
    ("dims", "product", "length_cm,height_cm,width_cm", "product_id", None, None, None),
    ("rating", "review_obs", "score", "product_id", "created_at", 365, 5),
    ("local_popularity", "city_sales", "n", "product_id,city", "purchased_at", None, None),
    ("rank_reason", "offer_obs", "count", "product_id", "observed_at", 90, None),
]

DDL = """
CREATE TABLE product(product_id TEXT PRIMARY KEY, category TEXT, weight_g REAL, length_cm REAL, height_cm REAL, width_cm REAL);
CREATE TABLE seller(seller_id TEXT PRIMARY KEY, state TEXT, city TEXT);
CREATE TABLE offer_obs(product_id TEXT, seller_id TEXT, price REAL, observed_at INTEGER);
CREATE INDEX ix_offer_ps ON offer_obs(product_id, seller_id, observed_at);
CREATE INDEX ix_offer_p ON offer_obs(product_id, observed_at);
CREATE TABLE review_obs(product_id TEXT, score INTEGER, created_at INTEGER);
CREATE INDEX ix_rev ON review_obs(product_id, created_at);
CREATE TABLE city_sales(product_id TEXT, city TEXT, purchased_at INTEGER);
CREATE INDEX ix_cs ON city_sales(product_id, city, purchased_at);
CREATE TABLE funnel(snapshot_id TEXT, product_id TEXT, stage TEXT, PRIMARY KEY(snapshot_id, stage, product_id));
CREATE TABLE claim_registry(claim_field TEXT PRIMARY KEY, claim_type TEXT, grain TEXT, external INTEGER,
                            surface_policy TEXT, sensitivity TEXT);
CREATE TABLE evidence_binding(claim_field TEXT, evidence_table TEXT, value_column TEXT, key_columns TEXT,
                              clock_column TEXT, tau_days INTEGER, min_n INTEGER, PRIMARY KEY(claim_field, evidence_table));
CREATE TABLE coverage_ledger(run_id TEXT, snapshot_id TEXT, product_id TEXT, claim_field TEXT, claim_id TEXT,
                             disposition TEXT, reason TEXT, witness TEXT, clock_age_days REAL,
                             PRIMARY KEY(run_id, snapshot_id, product_id, claim_field));
"""


def build_db(path: str, ol: Olist, funnel_rows) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(DDL)
    con.executemany("INSERT INTO product VALUES (?,?,?,?,?,?)", ol.products.itertuples(index=False, name=None))
    con.executemany("INSERT INTO seller VALUES (?,?,?)", ol.sellers.itertuples(index=False, name=None))
    con.executemany("INSERT INTO offer_obs VALUES (?,?,?,?)",
                    ol.offers[["product_id", "seller_id", "price", "observed_at"]].itertuples(index=False, name=None))
    con.executemany("INSERT INTO review_obs VALUES (?,?,?)",
                    ol.reviews[["product_id", "score", "created_at"]].itertuples(index=False, name=None))
    con.executemany("INSERT INTO city_sales VALUES (?,?,?)",
                    ol.sales[["product_id", "city", "purchased_at"]].itertuples(index=False, name=None))
    con.executemany("INSERT OR IGNORE INTO funnel VALUES (?,?,?)", funnel_rows)
    con.executemany("INSERT INTO claim_registry VALUES (?,?,?,?,?,?)", REGISTRY)
    con.executemany("INSERT INTO evidence_binding VALUES (?,?,?,?,?,?,?)", BINDING)
    con.commit()
    return con


@dataclass
class Ablation:
    clock: bool = True     # caliber/freshness conjunct
    grain: bool = True     # key/grain conjunct
    policy: bool = True    # disclosure-policy conjunct
    lineage: bool = True   # assembly-lineage (funnel) conjunct
    name: str = "GCC"


def _eq_price(a: str, b: float) -> bool:
    return abs(float(a) - float(b)) < 0.005


class Gate:
    def __init__(self, con: sqlite3.Connection, ab: Ablation = Ablation(), run_id: str = "run"):
        self.con = con
        self.ab = ab
        self.run_id = run_id
        self.reg = {r[0]: r for r in con.execute("SELECT * FROM claim_registry")}
        self.bind = {r[0]: r for r in con.execute("SELECT * FROM evidence_binding")}
        self.cur = con.cursor()
        self._ledger = []

    # --- lineage (G5) -----------------------------------------------------
    def _lineage_ok(self, snap: str, p: str) -> bool:
        q = "SELECT COUNT(DISTINCT stage) FROM funnel WHERE snapshot_id=? AND product_id=?"
        return self.cur.execute(q, (snap, p)).fetchone()[0] == 3

    # --- witness resolution -----------------------------------------------
    def decide(self, c: dict) -> tuple[str, str, str, float | None]:
        f, t, p = c["field"], c["t"], c["product_id"]
        reg = self.reg.get(f)
        if reg is None:
            return "refuse", "G1_unregistered", "", None
        if self.ab.lineage and not self._lineage_ok(c["snapshot_id"], p):
            return "refuse", "G5_lineage", "", None
        _, ctype, grain, external, policy, sens = reg
        if self.ab.policy:
            if sens == "high":
                return "refuse", "G4_disclosure", "", None
            if not external:
                return "degrade", "internal_context", "", None
        b = self.bind[f]
        tau = b[5]
        cur = self.cur
        v = c["value"]
        if f == "category":
            r = cur.execute("SELECT category FROM product WHERE product_id=?", (p,)).fetchone()
            return ("caveat", "governed_unclocked", "product", None) if r and r[0] == v else ("refuse", "value_mismatch", "product", None)
        if f in ("weight",):
            r = cur.execute("SELECT weight_g FROM product WHERE product_id=?", (p,)).fetchone()
            ok = r and r[0] is not None and int(r[0]) == int(v)
            return ("caveat", "governed_unclocked", "product", None) if ok else ("refuse", "value_mismatch", "product", None)
        if f == "dims":
            r = cur.execute("SELECT length_cm,height_cm,width_cm FROM product WHERE product_id=?", (p,)).fetchone()
            ok = r and None not in r and "x".join(str(int(x)) for x in r) == v
            return ("caveat", "governed_unclocked", "product", None) if ok else ("refuse", "value_mismatch", "product", None)
        if f == "ships_from":
            r = cur.execute("SELECT state FROM seller WHERE seller_id=?", (c["key"],)).fetchone()
            return ("caveat", "governed_unclocked", "seller", None) if r and r[0] == v else ("refuse", "value_mismatch", "seller", None)
        if f == "price_offer":
            r = cur.execute("SELECT price, observed_at FROM offer_obs WHERE product_id=? AND seller_id=? AND observed_at<=? "
                            "ORDER BY observed_at DESC LIMIT 1", (p, c["key"], t)).fetchone()
            if not r:
                return "refuse", "no_witness", "offer", None
            age = (t - r[1]) / DAY
            if not _eq_price(v, r[0]):
                return "refuse", "value_mismatch", "offer", age
            if self.ab.clock and age > tau:
                return "refuse", "G2_stale", "offer", age
            return "surface", "clock_backed", "offer", age
        if f == "price_product":
            rows = cur.execute("SELECT seller_id, price, MAX(observed_at) FROM offer_obs WHERE product_id=? AND observed_at<=? "
                               "GROUP BY seller_id", (p, t)).fetchall()
            if not rows:
                return "refuse", "no_witness", "offer", None
            fresh = [r for r in rows if (t - r[2]) / DAY <= tau] if self.ab.clock else rows
            if not fresh:
                age = min((t - r[2]) / DAY for r in rows)
                return "refuse", "G2_stale", "offer", age
            age = min((t - r[2]) / DAY for r in fresh)
            if self.ab.grain:
                prices = {round(r[1], 2) for r in fresh}
                if len(prices) > 1:
                    return "refuse", "G3_grain", "offer", age
                ok = _eq_price(v, next(iter(prices)))
            else:
                ok = any(_eq_price(v, r[1]) for r in fresh)
            return ("surface", "clock_backed", "offer", age) if ok else ("refuse", "value_mismatch", "offer", age)
        if f == "avail_local":
            if self.ab.grain:
                r = cur.execute("SELECT MAX(o.observed_at) FROM offer_obs o JOIN seller s ON s.seller_id=o.seller_id "
                                "WHERE o.product_id=? AND s.state=? AND o.observed_at<=?", (p, v, t)).fetchone()
                if v != c["c_state"]:
                    return "refuse", "G3_grain", "offer", None
            else:  # grain-blind: any offer of the product licenses the regional claim
                r = cur.execute("SELECT MAX(observed_at) FROM offer_obs WHERE product_id=? AND observed_at<=?", (p, t)).fetchone()
            if not r or r[0] is None:
                return "refuse", "no_witness", "offer", None
            age = (t - r[0]) / DAY
            if self.ab.clock and age > tau:
                return "refuse", "G2_stale", "offer", age
            return "surface", "clock_backed", "offer", age
        if f == "rating":
            r = cur.execute("SELECT SUM(score), COUNT(*), MAX(created_at) FROM review_obs WHERE product_id=? AND created_at<=?",
                            (p, t)).fetchone()
            if not r or not r[1]:
                return "refuse", "no_witness", "review", None
            avg, n = round(r[0] / r[1], 1), r[1]
            va, vn = v.split("|")
            if abs(float(va) - avg) > 1e-9 or int(vn) != n:
                return "refuse", "value_mismatch", "review", None
            age = (t - r[2]) / DAY
            if n < b[6]:
                return "caveat", "small_denominator", "review", age
            if self.ab.clock and age > tau:
                return "caveat", "G2_stale", "review", age
            return "surface", "clock_backed", "review", age
        # policy-ablated branches: treat sensitive/internal fields as ordinary governed aggregates
        if f == "local_popularity":
            r = cur.execute("SELECT COUNT(*) FROM city_sales WHERE product_id=? AND city=? AND purchased_at<=?",
                            (p, c["key"], t)).fetchone()
            return ("surface", "governed_aggregate", "city_sales", None) if r[0] == int(v) else ("refuse", "value_mismatch", "city_sales", None)
        if f == "rank_reason":
            r = cur.execute("SELECT COUNT(*) FROM offer_obs WHERE product_id=? AND observed_at<=? AND observed_at>?",
                            (p, t, t - 90 * DAY)).fetchone()
            return ("surface", "governed_aggregate", "offer", None) if r[0] == int(v) else ("refuse", "value_mismatch", "offer", None)
        return "refuse", "no_binding", "", None

    def run(self, claims: list[dict], write_ledger: bool = True) -> list[str]:
        out = []
        led = []
        for c in claims:
            d, reason, wit, age = self.decide(c)
            out.append(d)
            if write_ledger:
                led.append((self.run_id, c["snapshot_id"], c["product_id"], c["field"], c["claim_id"], d, reason, wit, age))
            c.setdefault("_reason", {})[self.ab.name] = reason
        if write_ledger:
            self.con.executemany("INSERT OR REPLACE INTO coverage_ledger VALUES (?,?,?,?,?,?,?,?,?)", led)
            self.con.commit()
        return out
