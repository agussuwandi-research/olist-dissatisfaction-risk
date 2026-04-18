"""
02_preprocessing.py
Filter, dedup review, buat target, temporal split.
Output: data_interim/reviews_dedup.csv, data_interim/orders_filtered.csv
        outputs/tables/02_*.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
RAW     = ROOT / "data_raw"
INTERIM = ROOT / "data_interim"
OUT     = ROOT / "outputs" / "tables"
INTERIM.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading orders dan reviews...")
DATE_COLS_ORDERS = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
orders = pd.read_csv(RAW / "olist_orders_dataset.csv",
                     parse_dates=DATE_COLS_ORDERS, low_memory=False)
reviews = pd.read_csv(RAW / "olist_order_reviews_dataset.csv",
                      parse_dates=["review_creation_date", "review_answer_timestamp"],
                      low_memory=False)

print(f"  orders raw   : {len(orders):,} rows")
print(f"  reviews raw  : {len(reviews):,} rows")

# ── Step 1: Filter delivered only ─────────────────────────────────────────────
orders = orders[orders["order_status"] == "delivered"].copy()
print(f"\nSetelah filter delivered: {len(orders):,} rows")

# ── Step 2: Temporal filter ────────────────────────────────────────────────────
# Buang 2016 dan tail 2018-09/10
orders["_ym"] = orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
EXCLUDE_YM = {"2018-09", "2018-10"}
orders = orders[
    (orders["order_purchase_timestamp"].dt.year != 2016) &
    (~orders["_ym"].isin(EXCLUDE_YM))
].copy()
orders.drop(columns=["_ym"], inplace=True)
print(f"Setelah temporal filter   : {len(orders):,} rows")
print(f"  Periode: {orders['order_purchase_timestamp'].min().date()} "
      f"s/d {orders['order_purchase_timestamp'].max().date()}")

# ── Step 3: Review dedup (keep last by review_creation_date) ──────────────────
reviews_sorted = reviews.sort_values(
    ["order_id", "review_creation_date", "review_answer_timestamp"],
    ascending=True, na_position="first"
)
reviews_dedup = reviews_sorted.drop_duplicates(subset=["order_id"], keep="last").copy()
print(f"\nReview dedup: {len(reviews):,} → {len(reviews_dedup):,} rows")
print(f"  Duplikat dihapus: {len(reviews) - len(reviews_dedup):,}")

# ── Step 4: Merge review ke orders ────────────────────────────────────────────
# Ambil kolom review yang diperlukan saja
review_cols = ["order_id", "review_score", "review_creation_date"]
df = orders.merge(
    reviews_dedup[review_cols],
    on="order_id",
    how="inner"  # drop orders tanpa review (~0.67% dari delivered)
).copy()
print(f"\nSetelah merge dengan review: {len(df):,} rows")
print(f"  Orders tanpa review (dropped): {len(orders) - len(df):,}")

# ── Step 5: Target variables ───────────────────────────────────────────────────
df["target_broad"]  = (df["review_score"] <= 3).astype(int)
df["target_severe"] = (df["review_score"] <= 2).astype(int)
print(f"\nBroad dissatisfaction rate  : {df['target_broad'].mean():.4f}")
print(f"Severe dissatisfaction rate : {df['target_severe'].mean():.4f}")

# ── Step 6: Temporal split ─────────────────────────────────────────────────────
ts = df["order_purchase_timestamp"]
train_mask = (ts >= "2017-01-01") & (ts < "2017-12-01")
val_mask   = (ts >= "2017-12-01") & (ts < "2018-03-01")
test_mask  = (ts >= "2018-03-01") & (ts < "2018-09-01")

df["split"] = np.select(
    [train_mask, val_mask, test_mask],
    ["train",    "validation", "test"],
    default="drop"
)
n_drop = (df["split"] == "drop").sum()
df = df[df["split"] != "drop"].copy()
print(f"\nSplit temporal (rows dropped: {n_drop}):")

split_summary = df.groupby("split").agg(
    n_orders           =("order_id",       "count"),
    broad_positive_rate=("target_broad",   "mean"),
    severe_positive_rate=("target_severe", "mean"),
    date_min           =("order_purchase_timestamp", "min"),
    date_max           =("order_purchase_timestamp", "max"),
).reset_index()
# Urutkan split
split_order = {"train": 0, "validation": 1, "test": 2}
split_summary["_ord"] = split_summary["split"].map(split_order)
split_summary = split_summary.sort_values("_ord").drop(columns=["_ord"])
print(split_summary[["split","n_orders","broad_positive_rate","severe_positive_rate"]].to_string(index=False))

# ── Step 7: Validasi ───────────────────────────────────────────────────────────
print("\n=== VALIDASI ===")
# Tidak boleh ada duplikat order_id
assert df["order_id"].is_unique, "ERROR: order_id duplikat di dataset final!"
print("  order_id unik              : OK")
# Target tidak boleh null
assert df["target_broad"].isnull().sum() == 0,  "ERROR: target_broad ada null!"
assert df["target_severe"].isnull().sum() == 0, "ERROR: target_severe ada null!"
print("  target_broad null          : 0  OK")
print("  target_severe null         : 0  OK")
# Split harus ada semua 3 nilai
assert set(df["split"].unique()) == {"train","validation","test"}, "ERROR: split tidak lengkap!"
print("  split values (3)           : OK")

# ── Save ───────────────────────────────────────────────────────────────────────
reviews_dedup.to_csv(INTERIM / "reviews_dedup.csv", index=False)
df.to_csv(INTERIM / "orders_filtered.csv", index=False)

overall = pd.DataFrame([{
    "n_orders_final":      len(df),
    "broad_positive_rate": round(df["target_broad"].mean(), 4),
    "severe_positive_rate":round(df["target_severe"].mean(), 4),
    "date_min":            df["order_purchase_timestamp"].min().date(),
    "date_max":            df["order_purchase_timestamp"].max().date(),
}])
overall.to_csv(OUT / "02_dataset_summary.csv", index=False)
split_summary.to_csv(OUT / "02_split_summary.csv", index=False)

print(f"\nSaved:")
print(f"  {INTERIM / 'reviews_dedup.csv'}")
print(f"  {INTERIM / 'orders_filtered.csv'}")
print(f"  {OUT / '02_dataset_summary.csv'}")
print(f"  {OUT / '02_split_summary.csv'}")
print("\nPREPROCESSING SELESAI.")
