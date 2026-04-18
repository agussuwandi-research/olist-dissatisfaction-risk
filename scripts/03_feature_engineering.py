"""
03_feature_engineering.py
Bangun semua fitur dari data_interim/orders_filtered.csv + raw tables.
Output:
  data_interim/order_level_features_pre.csv   — pre-fulfillment features
  data_interim/order_level_features_in.csv    — in-fulfillment features (tambahan)
  data_interim/explanatory_variables.csv      — post-outcome, hanya untuk analisis
  data_interim/clustering_features.csv        — subset numerik untuk K-Means
  outputs/tables/03_feature_contract.csv      — data dictionary
  outputs/tables/03_missing_value_report.csv
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
print("Loading tables...")
DATE_COLS = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
orders   = pd.read_csv(INTERIM / "orders_filtered.csv", parse_dates=DATE_COLS, low_memory=False)
items    = pd.read_csv(RAW / "olist_order_items_dataset.csv",   low_memory=False)
payments = pd.read_csv(RAW / "olist_order_payments_dataset.csv",low_memory=False)
products = pd.read_csv(RAW / "olist_products_dataset.csv",      low_memory=False)
sellers  = pd.read_csv(RAW / "olist_sellers_dataset.csv",       low_memory=False)
customers= pd.read_csv(RAW / "olist_customers_dataset.csv",     low_memory=False)
geo      = pd.read_csv(RAW / "olist_geolocation_dataset.csv",   low_memory=False)
cat_trans= pd.read_csv(RAW / "product_category_name_translation.csv", low_memory=False)

print(f"  orders_filtered : {len(orders):,}")

# ── Helper: Haversine ──────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = np.radians(lat1); phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# ── Step 1: Items — agregasi per order ────────────────────────────────────────
print("\nStep 1: Items aggregation...")
items_prod = items.merge(
    products[["product_id", "product_category_name",
              "product_weight_g", "product_length_cm",
              "product_height_cm", "product_width_cm"]],
    on="product_id", how="left"
)
items_prod["volume_cm3"] = (
    items_prod["product_length_cm"].fillna(0) *
    items_prod["product_height_cm"].fillna(0) *
    items_prod["product_width_cm"].fillna(0)
)

# Seller representative: seller dengan price tertinggi per order
# (Blueprint: seller_state dan distance_km mengikuti seller ini)
seller_rep = (
    items_prod
    .sort_values("price", ascending=False)
    .drop_duplicates("order_id", keep="first")
    [["order_id", "seller_id", "product_category_name"]]
    .rename(columns={
        "seller_id":               "seller_rep_id",
        "product_category_name":   "top_category_raw",
    })
)

items_agg = items_prod.groupby("order_id").agg(
    n_items            =("order_item_id",    "max"),
    n_unique_sellers   =("seller_id",        "nunique"),
    price_sum          =("price",            "sum"),
    freight_sum        =("freight_value",    "sum"),
    weight_sum         =("product_weight_g", "sum"),
    volume_sum         =("volume_cm3",       "sum"),
).reset_index()
items_agg["freight_to_price_ratio"] = (
    items_agg["freight_sum"] / (items_agg["price_sum"] + 0.01)
)
items_agg = items_agg.merge(seller_rep, on="order_id", how="left")

print(f"  items_agg rows: {len(items_agg):,}")

# ── Step 2: Category translation ───────────────────────────────────────────────
items_agg = items_agg.merge(
    cat_trans.rename(columns={
        "product_category_name":         "top_category_raw",
        "product_category_name_english": "top_category",
    }),
    on="top_category_raw", how="left"
)
items_agg["top_category"] = items_agg["top_category"].fillna("unknown")
items_agg.drop(columns=["top_category_raw"], inplace=True)

# ── Step 3: Payments — agregasi per order ─────────────────────────────────────
print("Step 3: Payments aggregation...")
pay_agg = payments.groupby("order_id").agg(
    payment_value_sum        =("payment_value",        "sum"),
    payment_installments_max =("payment_installments", "max"),
).reset_index()
# Modus payment type per order
pay_type = (
    payments.groupby("order_id")["payment_type"]
    .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "unknown")
    .reset_index()
    .rename(columns={"payment_type": "payment_type_main"})
)
pay_agg = pay_agg.merge(pay_type, on="order_id", how="left")
print(f"  pay_agg rows: {len(pay_agg):,}")

# ── Step 4: Geolocation centroid ───────────────────────────────────────────────
print("Step 4: Geolocation centroid...")
geo_centroid = (
    geo.groupby("geolocation_zip_code_prefix")
    .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"))
    .reset_index()
)
cust_geo = (
    customers[["customer_id", "customer_state", "customer_zip_code_prefix"]]
    .merge(
        geo_centroid.rename(columns={
            "geolocation_zip_code_prefix": "customer_zip_code_prefix",
            "lat": "cust_lat", "lng": "cust_lng"
        }),
        on="customer_zip_code_prefix", how="left"
    )
)
sell_geo = (
    sellers[["seller_id", "seller_state", "seller_zip_code_prefix"]]
    .merge(
        geo_centroid.rename(columns={
            "geolocation_zip_code_prefix": "seller_zip_code_prefix",
            "lat": "sell_lat", "lng": "sell_lng"
        }),
        on="seller_zip_code_prefix", how="left"
    )
)
print(f"  geo_centroid unique zips: {len(geo_centroid):,}")
print(f"  customer zip coverage  : {cust_geo['cust_lat'].notna().mean():.4f}")
print(f"  seller zip coverage    : {sell_geo['sell_lat'].notna().mean():.4f}")

# ── Step 5: Bangun master order-level table ────────────────────────────────────
print("\nStep 5: Build master table...")
master = orders.copy()

# Customer
master = master.merge(
    cust_geo[["customer_id", "customer_state", "cust_lat", "cust_lng"]],
    on="customer_id", how="left"
)

# Items + seller rep
master = master.merge(items_agg, on="order_id", how="left")

# Seller geo
master = master.merge(
    sell_geo[["seller_id", "seller_state", "sell_lat", "sell_lng"]],
    left_on="seller_rep_id", right_on="seller_id", how="left"
).drop(columns=["seller_id"])

# Payments
master = master.merge(pay_agg, on="order_id", how="left")

# ── Step 6: Derived features ───────────────────────────────────────────────────
print("Step 6: Derived features...")

# Pre-fulfillment temporal
master["estimated_delivery_days"] = (
    (master["order_estimated_delivery_date"] - master["order_purchase_timestamp"])
    .dt.total_seconds() / 86400
)
master["purchase_month"]     = master["order_purchase_timestamp"].dt.month
master["purchase_dayofweek"] = master["order_purchase_timestamp"].dt.dayofweek

# Geographic
master["same_state"] = (
    (master["customer_state"] == master["seller_state"]).astype(float)
)
geo_mask = master["cust_lat"].notna() & master["sell_lat"].notna()
master["distance_km"] = np.nan
master.loc[geo_mask, "distance_km"] = haversine_km(
    master.loc[geo_mask, "cust_lat"], master.loc[geo_mask, "cust_lng"],
    master.loc[geo_mask, "sell_lat"], master.loc[geo_mask, "sell_lng"],
)

# In-fulfillment (tambahan untuk Setting 2 & 3)
master["approval_lag_hrs"] = (
    (master["order_approved_at"] - master["order_purchase_timestamp"])
    .dt.total_seconds() / 3600
)
master["carrier_pickup_lag_hrs"] = (
    (master["order_delivered_carrier_date"] - master["order_approved_at"])
    .dt.total_seconds() / 3600
)

# Explanatory-only (POST-OUTCOME — tidak boleh masuk feature set model)
master["actual_delivery_days"] = (
    (master["order_delivered_customer_date"] - master["order_purchase_timestamp"])
    .dt.total_seconds() / 86400
)
master["is_late"] = (
    master["order_delivered_customer_date"] > master["order_estimated_delivery_date"]
).astype(float)
master["estimation_error"] = (
    master["actual_delivery_days"] - master["estimated_delivery_days"]
)
master["seller_phase_days"] = (
    (master["order_delivered_carrier_date"] - master["order_purchase_timestamp"])
    .dt.total_seconds() / 86400
)
master["logistics_phase_days"] = (
    (master["order_delivered_customer_date"] - master["order_delivered_carrier_date"])
    .dt.total_seconds() / 86400
)

print(f"  master rows: {len(master):,}")
print(f"  distance_km coverage: {master['distance_km'].notna().mean():.4f}")

# ── Step 7: Define kolom per kategori ─────────────────────────────────────────
ID_COLS = ["order_id", "split", "target_broad", "target_severe"]

PRE_FEAT = [
    "estimated_delivery_days",
    "purchase_month", "purchase_dayofweek",
    "customer_state", "seller_state", "same_state",
    "distance_km",
    "n_items", "n_unique_sellers",
    "price_sum", "freight_sum", "freight_to_price_ratio",
    "weight_sum", "volume_sum",
    "payment_value_sum", "payment_installments_max", "payment_type_main",
    "top_category",
]

IN_EXTRA = [
    "approval_lag_hrs",
    "carrier_pickup_lag_hrs",
]

EXPL_ONLY = [
    "actual_delivery_days", "is_late", "estimation_error",
    "seller_phase_days", "logistics_phase_days",
]

# ── Step 8: Missing value report ──────────────────────────────────────────────
print("\nStep 8: Missing value report...")
all_feat = PRE_FEAT + IN_EXTRA + EXPL_ONLY
mv_rows = []
for col in all_feat:
    if col in master.columns:
        n = int(master[col].isnull().sum())
        mv_rows.append({
            "column": col,
            "n_missing": n,
            "pct_missing": round(n / len(master) * 100, 2),
        })
mv_df = pd.DataFrame(mv_rows)
print(mv_df[mv_df["n_missing"] > 0].to_string(index=False))
mv_df.to_csv(OUT / "03_missing_value_report.csv", index=False)

# ── Step 9: Save feature tables ───────────────────────────────────────────────
print("\nStep 9: Save feature tables...")

pre_cols = ID_COLS + PRE_FEAT
in_cols  = ID_COLS + PRE_FEAT + IN_EXTRA
expl_cols = ["order_id", "split"] + EXPL_ONLY

# Filter kolom yang ada di master
pre_cols  = [c for c in pre_cols  if c in master.columns]
in_cols   = [c for c in in_cols   if c in master.columns]
expl_cols = [c for c in expl_cols if c in master.columns]

master[pre_cols].to_csv(INTERIM / "order_level_features_pre.csv", index=False)
master[in_cols].to_csv(INTERIM / "order_level_features_in.csv",   index=False)
master[expl_cols].to_csv(INTERIM / "explanatory_variables.csv",   index=False)

# Clustering features: subset numerik dari pre-fulfillment
CLUST_FEATS = [
    "order_id", "split",
    "freight_sum", "price_sum", "freight_to_price_ratio",
    "weight_sum", "volume_sum",
    "estimated_delivery_days", "distance_km",
    "n_items", "payment_installments_max", "same_state",
    # Tambahkan target untuk profiling nanti (tidak dipakai sebagai input K-Means)
    "target_broad", "target_severe",
]
clust_cols = [c for c in CLUST_FEATS if c in master.columns]
master[clust_cols].to_csv(INTERIM / "clustering_features.csv", index=False)

# ── Step 10: Feature contract ─────────────────────────────────────────────────
print("\nStep 10: Feature contract...")
contract = [
    # Pre-fulfillment
    {"feature": "estimated_delivery_days",    "definition": "Estimasi waktu delivery dari platform",
     "formula": "estimated_delivery_date - purchase_timestamp",
     "unit": "hari", "source": "orders", "aggregation": "per order",
     "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "purchase_month",             "definition": "Bulan pembelian",
     "formula": "purchase_timestamp.month", "unit": "1-12",
     "source": "orders", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "purchase_dayofweek",         "definition": "Hari dalam minggu (0=Senin)",
     "formula": "purchase_timestamp.dayofweek", "unit": "0-6",
     "source": "orders", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "customer_state",             "definition": "State pelanggan",
     "formula": "dari customers table", "unit": "kategorikal",
     "source": "customers", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "seller_state",               "definition": "State seller representative",
     "formula": "seller dengan price tertinggi per order", "unit": "kategorikal",
     "source": "sellers", "aggregation": "per order", "timing": "pre", "in_model": "ya",
     "note": "seller representative = seller dengan price item tertinggi per order"},
    {"feature": "same_state",                 "definition": "Flag customer dan seller satu state",
     "formula": "customer_state == seller_state → 1/0", "unit": "biner",
     "source": "customers + sellers", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "distance_km",                "definition": "Jarak Haversine customer-seller",
     "formula": "haversine(cust_centroid, sell_centroid)", "unit": "km",
     "source": "geolocation", "aggregation": "per order", "timing": "pre", "in_model": "ya",
     "note": "centroid = median lat/lng per zip prefix"},
    {"feature": "n_items",                    "definition": "Jumlah item dalam order",
     "formula": "max(order_item_id) per order_id", "unit": "count",
     "source": "order_items", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "n_unique_sellers",           "definition": "Jumlah seller unik dalam order",
     "formula": "nunique(seller_id) per order_id", "unit": "count",
     "source": "order_items", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "price_sum",                  "definition": "Total harga semua item",
     "formula": "sum(price) per order_id", "unit": "IDR",
     "source": "order_items", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "freight_sum",                "definition": "Total ongkos kirim semua item",
     "formula": "sum(freight_value) per order_id", "unit": "IDR",
     "source": "order_items", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "freight_to_price_ratio",     "definition": "Rasio ongkir terhadap harga",
     "formula": "freight_sum / (price_sum + 0.01)", "unit": "rasio",
     "source": "order_items", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "weight_sum",                 "definition": "Total berat semua produk",
     "formula": "sum(product_weight_g) per order_id", "unit": "gram",
     "source": "order_items + products", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "volume_sum",                 "definition": "Total volume semua produk",
     "formula": "sum(L*H*W) per order_id", "unit": "cm3",
     "source": "order_items + products", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "payment_value_sum",          "definition": "Total nilai pembayaran",
     "formula": "sum(payment_value) per order_id", "unit": "IDR",
     "source": "order_payments", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "payment_installments_max",   "definition": "Cicilan terbanyak dalam order",
     "formula": "max(payment_installments) per order_id", "unit": "count",
     "source": "order_payments", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "payment_type_main",          "definition": "Tipe pembayaran dominan",
     "formula": "modus payment_type per order_id", "unit": "kategorikal",
     "source": "order_payments", "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": ""},
    {"feature": "top_category",              "definition": "Kategori produk utama per order",
     "formula": "kategori dari item dengan price tertinggi per order", "unit": "kategorikal",
     "source": "order_items + products + category_translation",
     "aggregation": "per order", "timing": "pre", "in_model": "ya", "note": "dalam bahasa Inggris"},
    # In-fulfillment
    {"feature": "approval_lag_hrs",           "definition": "Waktu dari pembelian ke approval",
     "formula": "order_approved_at - order_purchase_timestamp", "unit": "jam",
     "source": "orders", "aggregation": "per order", "timing": "in", "in_model": "ya (S2,S3 only)",
     "note": "diketahui setelah order_approved_at"},
    {"feature": "carrier_pickup_lag_hrs",     "definition": "Waktu dari approval ke carrier pickup",
     "formula": "delivered_carrier_date - order_approved_at", "unit": "jam",
     "source": "orders", "aggregation": "per order", "timing": "in", "in_model": "ya (S2,S3 only)",
     "note": "diketahui setelah carrier_date; fitur terpenting S2"},
    # Explanatory only
    {"feature": "actual_delivery_days",       "definition": "Waktu actual pengiriman ke pelanggan",
     "formula": "delivered_customer_date - purchase_timestamp", "unit": "hari",
     "source": "orders", "aggregation": "per order", "timing": "post-outcome",
     "in_model": "TIDAK — leakage", "note": "hanya untuk explanatory analysis"},
    {"feature": "is_late",                    "definition": "Flag order terlambat dari estimasi",
     "formula": "delivered_customer_date > estimated_delivery_date → 1/0", "unit": "biner",
     "source": "orders", "aggregation": "per order", "timing": "post-outcome",
     "in_model": "TIDAK — leakage", "note": "hanya untuk explanatory analysis"},
    {"feature": "estimation_error",           "definition": "Selisih actual vs estimasi delivery",
     "formula": "actual_delivery_days - estimated_delivery_days", "unit": "hari",
     "source": "orders", "aggregation": "per order", "timing": "post-outcome",
     "in_model": "TIDAK — leakage", "note": "hanya untuk explanatory analysis; positif = terlambat"},
    {"feature": "seller_phase_days",          "definition": "Durasi fase seller (purchase → carrier)",
     "formula": "delivered_carrier_date - purchase_timestamp", "unit": "hari",
     "source": "orders", "aggregation": "per order", "timing": "post-outcome",
     "in_model": "TIDAK — leakage", "note": "untuk delivery chain decomposition"},
    {"feature": "logistics_phase_days",       "definition": "Durasi fase logistik (carrier → customer)",
     "formula": "delivered_customer_date - delivered_carrier_date", "unit": "hari",
     "source": "orders", "aggregation": "per order", "timing": "post-outcome",
     "in_model": "TIDAK — leakage", "note": "untuk delivery chain decomposition"},
]
contract_df = pd.DataFrame(contract)
contract_df.to_csv(OUT / "03_feature_contract.csv", index=False)
print(f"  Feature contract: {len(contract_df)} fitur terdokumentasi")

print(f"\nSaved:")
print(f"  {INTERIM / 'order_level_features_pre.csv'}  ({len(master):,} rows, {len(pre_cols)} cols)")
print(f"  {INTERIM / 'order_level_features_in.csv'}   ({len(master):,} rows, {len(in_cols)} cols)")
print(f"  {INTERIM / 'explanatory_variables.csv'}     ({len(master):,} rows)")
print(f"  {INTERIM / 'clustering_features.csv'}       ({len(master):,} rows)")
print(f"  {OUT / '03_feature_contract.csv'}")
print(f"  {OUT / '03_missing_value_report.csv'}")
print("\nFEATURE ENGINEERING SELESAI.")
