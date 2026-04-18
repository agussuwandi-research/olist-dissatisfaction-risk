"""
01_data_audit.py
Audit semua tabel raw: ukuran, null, key uniqueness, join coverage, review duplication.
Output disimpan ke outputs/tables/.
"""

import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data_raw"
OUT  = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "orders":               "olist_orders_dataset.csv",
    "customers":            "olist_customers_dataset.csv",
    "items":                "olist_order_items_dataset.csv",
    "payments":             "olist_order_payments_dataset.csv",
    "reviews":              "olist_order_reviews_dataset.csv",
    "products":             "olist_products_dataset.csv",
    "sellers":              "olist_sellers_dataset.csv",
    "geo":                  "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# ── Load ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING TABLES")
print("=" * 60)
tables = {}
for name, fname in FILES.items():
    tables[name] = pd.read_csv(RAW / fname, low_memory=False)
    print(f"  {name:<25} {len(tables[name]):>8,} rows  {len(tables[name].columns)} cols")

# ── 1. Summary ────────────────────────────────────────────────────────────────
print("\n=== 1. DATASET SUMMARY ===")
rows = []
for name, df in tables.items():
    rows.append({
        "table":      name,
        "n_rows":     len(df),
        "n_cols":     len(df.columns),
        "null_cells": int(df.isnull().sum().sum()),
        "null_pct":   round(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 2),
        "columns":    ", ".join(df.columns.tolist()),
    })
summary_df = pd.DataFrame(rows)
print(summary_df[["table", "n_rows", "n_cols", "null_cells", "null_pct"]].to_string(index=False))
summary_df.to_csv(OUT / "01_table_summary.csv", index=False)

# ── 2. Null per kolom ─────────────────────────────────────────────────────────
print("\n=== 2. NULL REPORT (kolom dengan null > 0) ===")
null_rows = []
for name, df in tables.items():
    for col in df.columns:
        n = int(df[col].isnull().sum())
        if n > 0:
            null_rows.append({
                "table": name, "column": col,
                "n_null": n,
                "pct_null": round(n / len(df) * 100, 2),
            })
null_df = pd.DataFrame(null_rows)
print(null_df.to_string(index=False) if len(null_df) else "  Tidak ada null.")
null_df.to_csv(OUT / "01_null_report.csv", index=False)

# ── 3. Key uniqueness ─────────────────────────────────────────────────────────
print("\n=== 3. KEY UNIQUENESS ===")
# expected_unique=False → duplikat memang diharapkan (bukan tanda error)
key_checks = [
    ("orders",    "order_id",    True),
    ("customers", "customer_id", True),
    ("products",  "product_id",  True),
    ("sellers",   "seller_id",   True),
    ("reviews",   "review_id",   True),   # setiap review harus punya ID unik
    ("reviews",   "order_id",    False),  # 1 order bisa punya >1 review → wajar non-unique
]
key_rows = []
for tname, col, expect_unique in key_checks:
    df = tables[tname]
    n_total  = len(df)
    n_unique = df[col].nunique()
    is_unique = (n_total == n_unique)
    key_rows.append({"table": tname, "key": col, "n_rows": n_total,
                     "n_unique": n_unique, "is_unique": is_unique,
                     "expected_unique": expect_unique})
    if is_unique:
        flag = ""
    elif not expect_unique:
        flag = "  ← non-unique (expected: 1 order bisa punya >1 review)"
    else:
        flag = "  ← DUPLIKAT TIDAK TERDUGA — perlu investigasi"
    print(f"  {tname}.{col:<25} unique={is_unique}  ({n_unique:,}/{n_total:,}){flag}")
pd.DataFrame(key_rows).to_csv(OUT / "01_key_uniqueness.csv", index=False)

# ── 4. Review duplication detail ──────────────────────────────────────────────
print("\n=== 4. REVIEW DUPLICATION DETAIL ===")
rev = tables["reviews"]
dup_orders = rev.groupby("order_id").size()
n_dup = int((dup_orders > 1).sum())
dup_ids = dup_orders[dup_orders > 1].index
conflict = rev[rev["order_id"].isin(dup_ids)].groupby("order_id")["review_score"].nunique()
n_conflict = int((conflict > 1).sum())
print(f"  Order dengan >1 review row   : {n_dup:,}")
print(f"  Di antaranya konflik score   : {n_conflict:,}")
print(f"  Keputusan: keep last berdasarkan review_creation_date")
pd.DataFrame({
    "metric": ["orders_multiple_reviews", "orders_conflicting_scores"],
    "value":  [n_dup, n_conflict],
}).to_csv(OUT / "01_review_duplication_summary.csv", index=False)

# ── 5. Join coverage ──────────────────────────────────────────────────────────
print("\n=== 5. JOIN COVERAGE ===")
orders = tables["orders"]; items = tables["items"]
payments = tables["payments"]; customers = tables["customers"]
products = tables["products"]; sellers = tables["sellers"]

def coverage(left, lc, right, rc, label):
    lv = set(left[lc].dropna()); rv = set(right[rc].dropna())
    m  = lv & rv; pct = len(m) / len(lv) * 100 if lv else 0
    print(f"  {label:<45} {pct:.2f}%  ({len(m):,}/{len(lv):,})")
    return {"join": label, "n_left": len(lv), "n_matched": len(m), "coverage_pct": round(pct, 2)}

join_rows = [
    coverage(orders,   "customer_id", customers, "customer_id", "orders.customer_id → customers"),
    coverage(items,    "order_id",    orders,    "order_id",    "items.order_id → orders"),
    coverage(payments, "order_id",    orders,    "order_id",    "payments.order_id → orders"),
    coverage(rev,      "order_id",    orders,    "order_id",    "reviews.order_id → orders"),
    coverage(items,    "product_id",  products,  "product_id",  "items.product_id → products"),
    coverage(items,    "seller_id",   sellers,   "seller_id",   "items.seller_id → sellers"),
]
pd.DataFrame(join_rows).to_csv(OUT / "01_join_coverage.csv", index=False)

# ── 6. Order status ───────────────────────────────────────────────────────────
print("\n=== 6. ORDER STATUS DISTRIBUTION ===")
sd = orders["order_status"].value_counts().reset_index()
sd.columns = ["order_status", "count"]
sd["pct"] = (sd["count"] / len(orders) * 100).round(2)
print(sd.to_string(index=False))
sd.to_csv(OUT / "01_order_status_distribution.csv", index=False)

# ── 7. Temporal distribution ──────────────────────────────────────────────────
print("\n=== 7. MONTHLY ORDER COUNT (all statuses) ===")
orders_dt = pd.read_csv(RAW / FILES["orders"],
                         parse_dates=["order_purchase_timestamp"], low_memory=False)
orders_dt["ym"] = orders_dt["order_purchase_timestamp"].dt.to_period("M").astype(str)
monthly = orders_dt.groupby("ym").size().reset_index(name="n_orders")
print(monthly.to_string(index=False))
monthly.to_csv(OUT / "01_monthly_distribution.csv", index=False)

# ── 8. Review score ───────────────────────────────────────────────────────────
print("\n=== 8. REVIEW SCORE DISTRIBUTION ===")
sc = rev["review_score"].value_counts().sort_index().reset_index()
sc.columns = ["review_score", "count"]
sc["pct"] = (sc["count"] / len(rev) * 100).round(2)
print(sc.to_string(index=False))
sc.to_csv(OUT / "01_review_score_distribution.csv", index=False)

print("\n" + "=" * 60)
print("AUDIT SELESAI. Output di outputs/tables/")
print("=" * 60)
