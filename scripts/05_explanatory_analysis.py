"""
05_explanatory_analysis.py
Analisis explanatory: expectation violation, delivery-chain decomposition,
category heterogeneity, geographic signals.
CATATAN: Semua variabel di sini adalah POST-OUTCOME — tidak dipakai di model prediktif.
Output: outputs/figures/05_*.png, outputs/tables/05_*.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import roc_auc_score

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
INTERIM= ROOT / "data_interim"
RAW    = ROOT / "data_raw"
OUT_T  = ROOT / "outputs" / "tables"
OUT_F  = ROOT / "outputs" / "figures"
OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading data...")
DATE_COLS = [
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_delivered_customer_date",
    "order_estimated_delivery_date",
]
orders = pd.read_csv(INTERIM / "orders_filtered.csv", parse_dates=DATE_COLS, low_memory=False)
expl   = pd.read_csv(INTERIM / "explanatory_variables.csv", low_memory=False)
pre    = pd.read_csv(INTERIM / "order_level_features_pre.csv", low_memory=False)
cat_trans = pd.read_csv(RAW / "product_category_name_translation.csv", low_memory=False)

# Gabung semua yang perlu.
# CATATAN: target_broad dan target_severe sudah ada di orders_filtered —
# JANGAN merge ulang dari pre, karena akan membuat suffix _x/_y → KeyError.
# Hanya ambil kolom deskriptif yang belum ada di orders_filtered.
df = orders.merge(expl, on=["order_id","split"], how="inner")
df = df.merge(
    pre[["order_id", "customer_state", "seller_state", "same_state", "top_category"]],
    on="order_id", how="left"
)

print(f"  Rows: {len(df):,}")
print(f"  Broad diss rate: {df['target_broad'].mean():.3f}")
print(f"  Severe diss rate: {df['target_severe'].mean():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# ANALISIS 1: Expectation Violation (estimation_error)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 1. EXPECTATION VIOLATION (estimation_error) ===")

df_ee = df.dropna(subset=["estimation_error","target_broad"]).copy()

# Statistik dasar
print(f"  Median estimation_error: {df_ee['estimation_error'].median():.2f} hari")
print(f"  % tiba lebih cepat dari janji: {(df_ee['estimation_error']<0).mean():.2%}")
print(f"  % tiba lebih lambat dari janji: {(df_ee['estimation_error']>0).mean():.2%}")

# Bin plot
df_ee["err_bin"] = pd.cut(
    df_ee["estimation_error"],
    bins=[-np.inf, -20, -15, -10, -5, -2, 0, 2, 5, 10, np.inf],
    labels=["<-20", "-20:-15", "-15:-10", "-10:-5", "-5:-2", "-2:0", "0:2", "2:5", "5:10", ">10"],
)
bin_stats = df_ee.groupby("err_bin", observed=True).agg(
    n=("target_broad","count"),
    diss_rate=("target_broad","mean"),
).reset_index()
bin_stats["diss_pct"] = (bin_stats["diss_rate"] * 100).round(1)
print("\nDissatisfaction rate per estimation_error bin:")
print(bin_stats.to_string(index=False))
bin_stats.to_csv(OUT_T / "05_estimation_error_bins.csv", index=False)

# Grafik
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["steelblue" if str(b).startswith("-") or b == "-2:0" else "tomato"
          for b in bin_stats["err_bin"]]
ax.bar(bin_stats["err_bin"].astype(str), bin_stats["diss_pct"], color=colors)
ax.set_xlabel("Estimation Error (hari): negatif = lebih cepat dari janji")
ax.set_ylabel("Broad Dissatisfaction Rate (%)")
ax.set_title("Dissatisfaction Rate per Expectation Violation Level\n"
             "(negatif = datang lebih cepat, positif = terlambat dari estimasi)")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
fig.savefig(OUT_F / "05_estimation_error_dissatisfaction.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 05_estimation_error_dissatisfaction.png")

# Asymmetry check
print("\n  Asymmetry check (loss-aversion style):")
asym_rows = []
for d in [2, 5, 10]:
    early_rate = df_ee[df_ee["estimation_error"] <= -d]["target_broad"].mean()
    late_rate  = df_ee[df_ee["estimation_error"] >=  d]["target_broad"].mean()
    ratio      = late_rate / early_rate if early_rate > 0 else float("inf")
    asym_rows.append({"delta_days": d, "early_diss_rate": round(early_rate,3),
                       "late_diss_rate": round(late_rate,3), "ratio": round(ratio,1)})
    print(f"    {d} hari lebih cepat: {early_rate:.3f}  |  "
          f"{d} hari terlambat: {late_rate:.3f}  (rasio: {ratio:.1f}x)")
pd.DataFrame(asym_rows).to_csv(OUT_T / "05_asymmetry_check.csv", index=False)

# Comparison: is_late (binary) vs estimation_error (continuous)
df_ee_nna = df_ee.dropna(subset=["is_late"])
auc_late = roc_auc_score(df_ee_nna["target_broad"], df_ee_nna["is_late"])
auc_error= roc_auc_score(df_ee_nna["target_broad"], df_ee_nna["estimation_error"])
print(f"\n  AUC — is_late (binary)         : {auc_late:.4f}")
print(f"  AUC — estimation_error (kontinu): {auc_error:.4f}")
pd.DataFrame([
    {"feature":"is_late","auc_vs_broad_diss": round(auc_late,4)},
    {"feature":"estimation_error","auc_vs_broad_diss": round(auc_error,4)},
]).to_csv(OUT_T / "05_late_vs_error_auc.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# ANALISIS 2: Delivery-Chain Decomposition
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 2. DELIVERY-CHAIN DECOMPOSITION ===")
df_dc = df.dropna(subset=["seller_phase_days","logistics_phase_days","target_broad"]).copy()

for phase in ["seller_phase_days", "logistics_phase_days"]:
    med = df_dc[phase].median()
    p90 = df_dc[phase].quantile(0.90)
    auc = roc_auc_score(df_dc["target_broad"], df_dc[phase])
    print(f"  {phase:<25} median={med:.2f}d  90p={p90:.2f}d  AUC={auc:.4f}")

# Quartile analysis
chain_rows = []
for phase in ["seller_phase_days", "logistics_phase_days"]:
    df_dc["_q"] = pd.qcut(df_dc[phase], 4, labels=["Q1","Q2","Q3","Q4"])
    q_stats = df_dc.groupby("_q", observed=True).agg(
        n=("target_broad","count"),
        diss_rate=("target_broad","mean")
    ).reset_index()
    q_stats["phase"] = phase
    q_stats.rename(columns={"_q":"quartile"}, inplace=True)
    chain_rows.append(q_stats)
chain_df = pd.concat(chain_rows, ignore_index=True)
print("\nDissatisfaction rate by delivery phase quartile:")
print(chain_df.to_string(index=False))
chain_df.to_csv(OUT_T / "05_delivery_chain_quartiles.csv", index=False)

# Violin plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, phase, label in zip(axes,
    ["seller_phase_days","logistics_phase_days"],
    ["Seller Phase (purchase→carrier)","Logistics Phase (carrier→customer)"]):
    data_neg = df_dc[df_dc["target_broad"]==0][phase].dropna()
    data_pos = df_dc[df_dc["target_broad"]==1][phase].dropna()
    vp = ax.violinplot([data_neg.clip(0,30), data_pos.clip(0,30)], showmedians=True)
    ax.set_xticks([1,2]); ax.set_xticklabels(["Satisfied (0)","Dissatisfied (1)"])
    ax.set_ylabel("Days"); ax.set_title(label)
    ax.set_ylim(0, 30)
plt.suptitle("Delivery Phase Distribution by Dissatisfaction")
plt.tight_layout()
fig.savefig(OUT_F / "05_delivery_chain_violin.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 05_delivery_chain_violin.png")

# ─────────────────────────────────────────────────────────────────────────────
# ANALISIS 3: Category Heterogeneity
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 3. CATEGORY HETEROGENEITY ===")
df_cat = df.dropna(subset=["top_category","target_broad"]).copy()
cat_stats = df_cat.groupby("top_category").agg(
    n=("target_broad","count"),
    diss_rate=("target_broad","mean"),
).reset_index()
cat_stats = cat_stats[cat_stats["n"] >= 100].sort_values("diss_rate", ascending=False)
cat_stats["diss_pct"] = (cat_stats["diss_rate"] * 100).round(1)
print("Top 10 highest dissatisfaction categories (≥100 orders):")
print(cat_stats.head(10).to_string(index=False))
print("Top 10 lowest dissatisfaction categories:")
print(cat_stats.tail(10).to_string(index=False))
cat_stats.to_csv(OUT_T / "05_category_dissatisfaction.csv", index=False)

# Bar chart
top15 = pd.concat([cat_stats.head(8), cat_stats.tail(7)])
fig, ax = plt.subplots(figsize=(10, 6))
colors = ["tomato" if r >= 0.25 else ("steelblue" if r <= 0.15 else "silver")
          for r in top15["diss_rate"]]
ax.barh(top15["top_category"], top15["diss_pct"], color=colors)
ax.axvline(df_cat["target_broad"].mean()*100, color="black", linestyle="--",
           label=f"Overall avg ({df_cat['target_broad'].mean()*100:.1f}%)")
ax.set_xlabel("Broad Dissatisfaction Rate (%)")
ax.set_title("Dissatisfaction Rate by Product Category\n(Top 8 worst + Bottom 7 best)")
ax.legend(); ax.invert_yaxis()
plt.tight_layout()
fig.savefig(OUT_F / "05_category_dissatisfaction.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 05_category_dissatisfaction.png")

# ─────────────────────────────────────────────────────────────────────────────
# ANALISIS 4: Geographic Signal
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== 4. GEOGRAPHIC SIGNAL ===")
df_geo = df.dropna(subset=["same_state","target_broad"]).copy()
geo_stats = df_geo.groupby("same_state").agg(
    n=("target_broad","count"),
    diss_rate=("target_broad","mean"),
    late_rate=("is_late","mean"),
).reset_index()
geo_stats["same_state_label"] = geo_stats["same_state"].map({1:"Same State", 0:"Cross State"})
print(geo_stats[["same_state_label","n","diss_rate","late_rate"]].to_string(index=False))
geo_stats.to_csv(OUT_T / "05_geographic_same_state.csv", index=False)

# Bar chart
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(geo_stats["same_state_label"], geo_stats["diss_rate"]*100,
       color=["steelblue","tomato"])
ax.set_ylabel("Broad Dissatisfaction Rate (%)")
ax.set_title("Dissatisfaction Rate: Same State vs Cross State")
plt.tight_layout()
fig.savefig(OUT_F / "05_geographic_same_state.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 05_geographic_same_state.png")

# State-level heterogeneity
df_st = df.dropna(subset=["customer_state","target_broad"]).copy()
state_stats = df_st.groupby("customer_state").agg(
    n=("target_broad","count"), diss_rate=("target_broad","mean"),
    late_rate=("is_late","mean"),
).reset_index()
state_stats = state_stats[state_stats["n"]>=200].sort_values("diss_rate",ascending=False)
state_stats.to_csv(OUT_T / "05_state_dissatisfaction.csv", index=False)
print("\nState-level dissatisfaction (top 5 worst, top 5 best):")
print(pd.concat([state_stats.head(5), state_stats.tail(5)]).to_string(index=False))

print("\nEXPLANATORY ANALYSIS SELESAI.")
