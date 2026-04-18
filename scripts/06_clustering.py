"""
06_clustering.py
K-Means risk segmentation pada pre-fulfillment features.
Urutan: redundancy check → RobustScaler → elbow + silhouette → stability check → profiling.
Output: outputs/tables/06_*.csv, outputs/figures/06_*.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.cluster     import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics     import silhouette_score

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data_interim"
OUT_T   = ROOT / "outputs" / "tables"
OUT_F   = ROOT / "outputs" / "figures"
OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print("Loading clustering features...")
clust_df = pd.read_csv(INTERIM / "clustering_features.csv", low_memory=False)
print(f"  Rows: {len(clust_df):,}  Cols: {len(clust_df.columns)}")
print(f"  Kolom: {list(clust_df.columns)}")

# Kandidat fitur numerik untuk K-Means (tanpa target dan identifiers)
CANDIDATE_FEATS = [
    "freight_sum", "price_sum", "freight_to_price_ratio",
    "weight_sum", "volume_sum",
    "estimated_delivery_days", "distance_km",
    "n_items", "payment_installments_max", "same_state",
]
CANDIDATE_FEATS = [c for c in CANDIDATE_FEATS if c in clust_df.columns]
print(f"\nKandidat fitur: {CANDIDATE_FEATS}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Redundancy Check (korelasi antar fitur)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 1: REDUNDANCY CHECK ===")
feat_data = clust_df[CANDIDATE_FEATS].copy()

# Impute numerik dengan median (clustering butuh data lengkap)
for col in CANDIDATE_FEATS:
    if feat_data[col].isnull().any():
        med = feat_data[col].median()
        feat_data[col] = feat_data[col].fillna(med)
        print(f"  Imputed {col} dengan median={med:.2f}")

corr_matrix = feat_data.corr().round(3)
corr_matrix.to_csv(OUT_T / "06_correlation_matrix.csv")
print("  Correlation matrix saved.")

# Identifikasi pasangan dengan korelasi > 0.85
high_corr = []
cols = corr_matrix.columns.tolist()
for i in range(len(cols)):
    for j in range(i+1, len(cols)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.85:
            high_corr.append({"feat1": cols[i], "feat2": cols[j], "correlation": r})
            print(f"  HIGH CORR: {cols[i]} ↔ {cols[j]} = {r:.3f}")

# Keputusan: drop payment_value_sum jika ada (tergantung kolom yang tersedia)
# Berdasarkan audit sebelumnya, price_sum dan payment_value_sum sangat berkorelasi
# weight_sum dan volume_sum bisa berkorelasi tinggi — pilih weight_sum
DROPPED = []
FINAL_FEATS = CANDIDATE_FEATS.copy()

# Drop volume_sum jika berkorelasi tinggi dengan weight_sum
if "volume_sum" in corr_matrix.columns and "weight_sum" in corr_matrix.columns:
    r_wv = abs(corr_matrix.loc["weight_sum", "volume_sum"])
    if r_wv > 0.85:
        FINAL_FEATS = [c for c in FINAL_FEATS if c != "volume_sum"]
        DROPPED.append(f"volume_sum (corr w/ weight_sum = {r_wv:.3f})")
        print(f"  → Dropped volume_sum (corr w/ weight_sum = {r_wv:.3f})")

print(f"\nFitur final untuk K-Means ({len(FINAL_FEATS)}): {FINAL_FEATS}")
if DROPPED: print(f"Dropped: {DROPPED}")

X_raw = feat_data[FINAL_FEATS].values

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Scaling dengan RobustScaler
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 2: ROBUSTSCALER ===")
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_raw)
print(f"  Scaled shape: {X_scaled.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Elbow + Silhouette untuk memilih K
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 3: ELBOW + SILHOUETTE (K=2..8) ===")
# Pakai sample 20K untuk efisiensi
N_SAMPLE = min(20000, len(X_scaled))
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(X_scaled), N_SAMPLE, replace=False)
X_sample = X_scaled[sample_idx]

inertias    = []
sil_scores  = []
K_RANGE     = range(2, 9)

for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_sample)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_sample, labels, sample_size=5000, random_state=42)
    sil_scores.append(sil)
    print(f"  K={k}  inertia={km.inertia_:.0f}  silhouette={sil:.4f}")

# Save elbow data
elbow_df = pd.DataFrame({"K": list(K_RANGE), "inertia": inertias, "silhouette": sil_scores})
elbow_df.to_csv(OUT_T / "06_elbow_silhouette.csv", index=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(list(K_RANGE), inertias, "o-", color="steelblue")
axes[0].set_xlabel("K"); axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Plot")
axes[1].plot(list(K_RANGE), sil_scores, "s-", color="tomato")
axes[1].set_xlabel("K"); axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Score vs K")
plt.suptitle("K Selection — K-Means Clustering")
plt.tight_layout()
fig.savefig(OUT_F / "06_elbow_silhouette.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 06_elbow_silhouette.png")

# Pilih K: silhouette terbaik di antara K=2..8 yang masih interpretable
# Default: K dengan silhouette tertinggi, prefer K yang lebih kecil jika selisih kecil
best_sil_idx = int(np.argmax(sil_scores))
K_BEST = list(K_RANGE)[best_sil_idx]
print(f"\n  Silhouette terbaik: K={K_BEST} (sil={sil_scores[best_sil_idx]:.4f})")
print(f"  *** K_FINAL dipilih: {K_BEST} ***")
print(f"  (Verifikasi interpretability di profil cluster setelah ini)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Stability Check
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== STEP 4: STABILITY CHECK (K={K_BEST}) ===")
SEEDS = [0, 7, 42, 99, 123]
profile_per_seed = []

for seed in SEEDS:
    km_s = KMeans(n_clusters=K_BEST, n_init=20, random_state=seed)
    labels_s = km_s.fit_predict(X_scaled)
    # Hitung dissatisfaction rate per cluster (sorted by diss_rate untuk alignment)
    clust_df_s = clust_df.copy()
    clust_df_s["cluster"] = labels_s
    rates = clust_df_s.groupby("cluster")["target_broad"].mean().sort_values().values
    profile_per_seed.append(rates)
    print(f"  seed={seed}: diss rates per cluster = {[round(r,3) for r in rates]}")

# Cek apakah urutan diss rate konsisten
stab_df = pd.DataFrame(profile_per_seed, columns=[f"cluster_{i}" for i in range(K_BEST)])
stab_std = stab_df.std()
print(f"\n  Std devs across seeds per cluster position: {stab_std.round(3).tolist()}")
if stab_std.max() < 0.03:
    print("  → STABLE: profil cluster konsisten lintas seed. Lanjut interpretasi.")
else:
    print("  → PERINGATAN: profil cluster cukup bervariasi. Pertimbangkan K lebih kecil.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Final K-Means (seed=42, full data)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n=== STEP 5: FINAL K-MEANS (K={K_BEST}, full data) ===")
km_final = KMeans(n_clusters=K_BEST, n_init=20, random_state=42)
labels_final = km_final.fit_predict(X_scaled)
clust_df["cluster"] = labels_final
print(f"  Cluster distribution:\n{pd.Series(labels_final).value_counts().sort_index().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: Cluster Profiling
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 6: CLUSTER PROFILING ===")

profile_cols = {
    "n_orders":         ("order_id", "count"),
    "broad_diss_rate":  ("target_broad", "mean"),
    "severe_diss_rate": ("target_severe", "mean"),
    "avg_freight_sum":  ("freight_sum", "mean"),
    "avg_price_sum":    ("price_sum", "mean"),
    "avg_freight_ratio":("freight_to_price_ratio","mean"),
    "avg_weight_sum":   ("weight_sum", "mean"),
    "avg_est_days":     ("estimated_delivery_days","mean"),
    "avg_distance_km":  ("distance_km","mean"),
    "avg_n_items":      ("n_items","mean"),
    "pct_same_state":   ("same_state","mean"),
}

agg_dict = {v[0]: (v[0], v[1]) for k, v in profile_cols.items() if v[0] in clust_df.columns}

# Tambahkan order_id count secara manual
agg_dict_clean = {}
for new_col, (src_col, func) in profile_cols.items():
    if src_col in clust_df.columns:
        agg_dict_clean[new_col] = pd.NamedAgg(column=src_col, aggfunc=func)

profile = clust_df.groupby("cluster").agg(**agg_dict_clean).reset_index()
profile = profile.sort_values("broad_diss_rate", ascending=False)

# Round
for col in profile.columns:
    if col not in ["cluster","n_orders"]:
        profile[col] = profile[col].round(3)

print(profile.to_string(index=False))
profile.to_csv(OUT_T / "06_cluster_profile.csv", index=False)

# Dominant category per cluster
if "top_category" in clust_df.columns:
    # Perlu merge — clust_df tidak punya top_category langsung
    pre_cat = pd.read_csv(ROOT / "data_interim" / "order_level_features_pre.csv",
                          usecols=["order_id","top_category"], low_memory=False)
    clust_with_cat = clust_df.merge(pre_cat, on="order_id", how="left")
    dom_cat = (clust_with_cat.groupby("cluster")["top_category"]
               .agg(lambda x: x.mode().iloc[0] if len(x.mode())>0 else "unknown")
               .reset_index().rename(columns={"top_category":"dominant_category"}))
    profile = profile.merge(dom_cat, on="cluster", how="left")
    print("\nDominant category per cluster:")
    print(profile[["cluster","broad_diss_rate","dominant_category"]].to_string(index=False))
    profile.to_csv(OUT_T / "06_cluster_profile.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Cluster Visualization
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== STEP 7: VISUALIZATION ===")

# Bar chart: dissatisfaction rate per cluster
fig, ax = plt.subplots(figsize=(7, 4))
colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(profile)))
ax.bar([f"Cluster {c}" for c in profile["cluster"]],
       profile["broad_diss_rate"] * 100, color=colors)
ax.set_ylabel("Broad Dissatisfaction Rate (%)")
ax.set_title("Dissatisfaction Rate per Cluster (sorted desc)")
plt.tight_layout()
fig.savefig(OUT_F / "06_cluster_diss_rate.png", dpi=120, bbox_inches="tight")
plt.close()

# Radar chart untuk profil cluster
from matplotlib.patches import FancyArrowPatch

radar_feats = ["avg_freight_ratio", "avg_weight_sum", "avg_est_days",
               "avg_distance_km", "pct_same_state", "avg_n_items"]
radar_feats = [c for c in radar_feats if c in profile.columns]

if len(radar_feats) >= 3:
    N = len(radar_feats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    cmap = plt.cm.tab10

    for idx, row in profile.iterrows():
        vals_raw = [row[f] for f in radar_feats]
        # Normalize per fitur (0-1 across clusters)
        mins = profile[radar_feats].min()
        maxs = profile[radar_feats].max()
        vals = [(v - mins[f]) / (maxs[f] - mins[f] + 1e-9)
                for v, f in zip(vals_raw, radar_feats)]
        vals += vals[:1]
        label = f"Cluster {int(row['cluster'])} (diss={row['broad_diss_rate']:.2f})"
        ax.plot(angles, vals, "o-", linewidth=2, label=label)
        ax.fill(angles, vals, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_feats, size=8)
    ax.set_yticks([0, 0.5, 1]); ax.set_ylim(0, 1)
    ax.set_title("Cluster Profile Radar Chart\n(normalized per feature)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_F / "06_cluster_radar.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 06_cluster_radar.png")

# Save cluster assignment
clust_df[["order_id","split","cluster","target_broad","target_severe"]].to_csv(
    ROOT / "data_interim" / "cluster_assignments.csv", index=False
)
print("  Saved: data_interim/cluster_assignments.csv")

print(f"\nSaved:")
print(f"  {OUT_T / '06_elbow_silhouette.csv'}")
print(f"  {OUT_T / '06_cluster_profile.csv'}")
print(f"  {OUT_T / '06_correlation_matrix.csv'}")
print(f"  {OUT_F / '06_elbow_silhouette.png'}")
print(f"  {OUT_F / '06_cluster_diss_rate.png'}")
print("\nCLUSTERING SELESAI.")
