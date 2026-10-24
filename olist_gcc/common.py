"""Shared loading utilities for the Olist-GCC benchmark.

The benchmark uses the public Olist Brazilian e-commerce dataset
(https://github.com/olist/work-at-olist-data, MIT license). Timestamps are
converted to integer epoch seconds.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400
OLIST_DIR = Path(os.environ.get("OLIST_DIR", Path(__file__).resolve().parent / "olist-data" / "datasets"))

# Contract caliber windows (days)
TAU_PRICE = 30
TAU_AVAIL = 60
TAU_RATING = 365
MIN_RATING_N = 5
ELIGIBLE_DAYS = 180
SALES_DAYS = 90
TOP_K = 5


def to_epoch(series: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(series)
    return ((dt - pd.Timestamp("1970-01-01")).dt.total_seconds()).astype("int64").to_numpy()


@dataclass
class Olist:
    products: pd.DataFrame      # product_id, category, weight_g, length_cm, height_cm, width_cm
    sellers: pd.DataFrame       # seller_id, state, city
    offers: pd.DataFrame        # product_id, seller_id, price, observed_at (one row per sold order item)
    reviews: pd.DataFrame       # product_id, score, created_at
    sales: pd.DataFrame         # product_id, customer_id, city, state, purchased_at
    customers: pd.DataFrame     # customer_unique_id, city, state, first_purchase


def load_olist() -> Olist:
    d = OLIST_DIR
    prod = pd.read_csv(d / "olist_products_dataset.csv")
    trans = pd.read_csv(d / "product_category_name_translation.csv", encoding="utf-8-sig")
    prod = prod.merge(trans, on="product_category_name", how="left")
    prod["category"] = prod["product_category_name_english"].fillna(prod["product_category_name"]).fillna("unknown")
    products = prod[["product_id", "category", "product_weight_g", "product_length_cm",
                     "product_height_cm", "product_width_cm"]].rename(columns={
        "product_weight_g": "weight_g", "product_length_cm": "length_cm",
        "product_height_cm": "height_cm", "product_width_cm": "width_cm"})

    sel = pd.read_csv(d / "olist_sellers_dataset.csv")
    sellers = sel.rename(columns={"seller_state": "state", "seller_city": "city"})[["seller_id", "state", "city"]]

    items = pd.read_csv(d / "olist_order_items_dataset.csv")
    orders = pd.read_csv(d / "olist_orders_dataset.csv")
    cust = pd.read_csv(d / "olist_customers_dataset.csv")
    orders = orders.merge(cust, on="customer_id")
    x = items.merge(orders[["order_id", "customer_unique_id", "customer_city", "customer_state",
                            "order_purchase_timestamp"]], on="order_id")
    x["ts"] = to_epoch(x["order_purchase_timestamp"])
    offers = x[["product_id", "seller_id", "price", "ts"]].rename(columns={"ts": "observed_at"})
    # one observation per (product, seller, instant); simultaneous rows keep the highest price
    offers = (offers.sort_values(["product_id", "seller_id", "observed_at", "price"])
              .drop_duplicates(["product_id", "seller_id", "observed_at"], keep="last")
              .reset_index(drop=True))

    sales = x[["product_id", "customer_unique_id", "customer_city", "customer_state", "ts"]].rename(columns={
        "customer_unique_id": "customer_id", "customer_city": "city", "customer_state": "state", "ts": "purchased_at"})

    rev = pd.read_csv(d / "olist_order_reviews_dataset.csv")
    ri = rev.merge(items[["order_id", "product_id"]].drop_duplicates(), on="order_id")
    ri["created_at"] = to_epoch(ri["review_creation_date"])
    reviews = ri[["product_id", "review_score", "created_at"]].rename(columns={"review_score": "score"})

    first = sales.groupby("customer_id").agg(city=("city", "first"), state=("state", "first"),
                                             first_purchase=("purchased_at", "min")).reset_index()
    return Olist(products, sellers, offers, reviews, sales, first)


def serving_times() -> list[int]:
    """Weekly serving instants (Sunday 12:00 UTC), 2018-01-07 .. 2018-08-26."""
    ts = pd.date_range("2018-01-07 12:00:00", "2018-08-26 12:00:00", freq="7D")
    return [int(t.value // 10**9) for t in ts]
