"""
09_error_analysis.py
Analisis error model utama: S2 LR (in-fulfillment, broad dissatisfaction).
Fokus pada pola FN (false negative) dan FP (false positive) di test set.

Pertanyaan:
  1. Apakah FN didominasi review score 3 (mild dissatisfaction)?
  2. Kategori / wilayah mana yang paling banyak FN?
  3. Apakah FP punya karakteristik berbeda dari TP?
  4. Seberapa beda profil FN vs TP secara fitur?

Output: outputs/tables/09_*.csv, outputs/figures/09_*.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (confusion_matrix, classification_report,
                                   roc_auc_score, precision_recall_curve)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data_interim"
OUT_T   = ROOT / "outputs" / "tables"
OUT_F   = ROOT / "outputs" / "figures"
OUT_T.mkdir(parents=True, exist_ok=True)
OUT_F.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
META      = {"order_id", "split", "target_broad", "target_severe"}
BLACKLIST = {
    "order_delivered_customer_date", "actual_delivery_days",
    "is_late", "estimation_error", "seller_phase_days", "logistics_phase_days",
    "review_score", "review_comment_message", "review_comment_title",
    "review_creation_date", "review_answer_timestamp",
    "order_id", "customer_id", "customer_unique_id", "seller_id", "seller_rep_id",
    "split", "target_broad", "target_severe",
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_estimated_delivery_date",
}
CAT_COLS = ["customer_state", "seller_state", "payment_type_main", "top_category"]

def get_feat_cols(df):
    return [c for c in df.columns if c not in META and c not in BLACKLIST]

def encode_cat(tr, va, te, cats, top_n=15):
    for col in cats:
        if col not in tr.columns:
            continue
        top = tr[col].value_counts().head(top_n).index.tolist()
        for d in [tr, va, te]:
            d[col] = d[col].where(d[col].isin(top), "Other")
        combined = pd.get_dummies(
            pd.concat([tr[[col]], va[[col]], te[[col]]]), prefix=col
        )
        lens = [len(tr), len(va), len(te)]
        d_tr = combined.iloc[:lens[0]].reset_index(drop=True)
        d_va = combined.iloc[lens[0]:lens[0]+lens[1]].reset_index(drop=True)
        d_te = combined.iloc[lens[0]+lens[1]:].reset_index(drop=True)
        for d, dum in [(tr, d_tr), (va, d_va), (te, d_te)]:
            d.drop(columns=[col], inplace=True, errors="ignore")
            d.reset_index(drop=True, inplace=True)
            for c in dum.columns:
                d[c] = dum[c].values
    return tr, va, te

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data...")
inf  = pd.read_csv(INTERIM / "order_level_features_in.csv",  low_memory=False)
expl = pd.read_csv(INTERIM / "explanatory_variables.csv",    low_memory=False)
orders = pd.read_csv(INTERIM / "orders_filtered.csv",        low_memory=False)

# Tambahkan review_score ke inf untuk analisis error (bukan sebagai fitur model)
orders_rv = orders[["order_id", "review_score"]].copy()
inf = inf.merge(orders_rv, on="order_id", how="left")

print(f"  in-fulfillment table: {len(inf):,} rows")

# ── Preprocessing S2 LR ────────────────────────────────────────────────────────
FEAT_COLS = get_feat_cols(inf)  # review_score ada di inf tapi tidak di FEAT_COLS karena di BLACKLIST
NUM_COLS  = [c for c in FEAT_COLS if c not in CAT_COLS]

tr = inf[inf["split"] == "train"].reset_index(drop=True)
va = inf[inf["split"] == "validation"].reset_index(drop=True)
te = inf[inf["split"] == "test"].reset_index(drop=True)

# Simpan metadata test untuk analisis nanti
te_meta = te[["order_id", "target_broad", "target_severe", "review_score"]].copy()

Xtr = tr[FEAT_COLS].copy()
Xva = va[FEAT_COLS].copy()
Xte = te[FEAT_COLS].copy()

# Impute
for col in NUM_COLS:
    if col not in Xtr.columns: continue
    med = Xtr[col].median()
    for d in [Xtr, Xva, Xte]:
        d[col] = d[col].fillna(med)

for col in CAT_COLS:
    if col not in Xtr.columns: continue
    for d in [Xtr, Xva, Xte]:
        d[col] = d[col].fillna("Unknown")

Xtr, Xva, Xte = encode_cat(Xtr, Xva, Xte, CAT_COLS)
Xtr = Xtr.fillna(0).astype(float)
Xva = Xva.fillna(0).astype(float)
Xte = Xte.fillna(0).astype(float)

ytr = tr["target_broad"].values
yva = va["target_broad"].values
yte = te["target_broad"].values

# Scaling
num_in_feat = [c for c in NUM_COLS if c in Xtr.columns]
scaler = StandardScaler()
Xtr_lr = Xtr.copy(); Xva_lr = Xva.copy(); Xte_lr = Xte.copy()
Xtr_lr[num_in_feat] = scaler.fit_transform(Xtr[num_in_feat])
Xva_lr[num_in_feat] = scaler.transform(Xva[num_in_feat])
Xte_lr[num_in_feat] = scaler.transform(Xte[num_in_feat])

# ── Train LR ──────────────────────────────────────────────────────────────────
print("Training S2 LR...")
lr = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                         random_state=42, solver="lbfgs")
lr.fit(Xtr_lr, ytr)

proba_va = lr.predict_proba(Xva_lr)[:, 1]
proba_te = lr.predict_proba(Xte_lr)[:, 1]

# Threshold dari validation (recall >= 0.50)
prec_v, rec_v, thr_v = precision_recall_curve(yva, proba_va)
valid_thr = [(t, r, p) for t, r, p in zip(thr_v, rec_v[:-1], prec_v[:-1]) if r >= 0.50]
threshold = max(valid_thr, key=lambda x: x[0])[0] if valid_thr else 0.5
print(f"  Threshold (val, recall≥0.50): {threshold:.4f}")
print(f"  Test AUC: {roc_auc_score(yte, proba_te):.4f}")

# Prediksi di test
y_pred = (proba_te >= threshold).astype(int)
te_meta = te_meta.copy()
te_meta["proba"]  = proba_te
te_meta["y_pred"] = y_pred
te_meta["y_true"] = yte

# Error type
def error_type(row):
    if row["y_true"] == 1 and row["y_pred"] == 1: return "TP"
    if row["y_true"] == 1 and row["y_pred"] == 0: return "FN"
    if row["y_true"] == 0 and row["y_pred"] == 1: return "FP"
    return "TN"
te_meta["error_type"] = te_meta.apply(error_type, axis=1)

# Tambahkan fitur asli untuk profiling
te_feats_raw = te[FEAT_COLS + ["order_id"]].copy()
te_meta = te_meta.merge(te_feats_raw, on="order_id", how="left")

# Tambahkan explanatory vars untuk profiling
expl_te = expl[expl["split"] == "test"].copy()
te_meta = te_meta.merge(
    expl_te[["order_id", "estimation_error", "seller_phase_days", "logistics_phase_days"]],
    on="order_id", how="left"
)

# ── Overview: Confusion Matrix ────────────────────────────────────────────────
print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(yte, y_pred)
tn, fp, fn, tp = cm.ravel()
n_test = len(yte)
n_pos  = yte.sum()
print(f"  TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")
print(f"  Precision: {tp/(tp+fp):.3f}  Recall: {tp/(tp+fn):.3f}")
print(f"\n  Dari {n_pos:,} pelanggan dissatisfied di test:")
print(f"    → {tp:,} berhasil dideteksi ({tp/n_pos:.1%}) — TP")
print(f"    → {fn:,} terlewat ({fn/n_pos:.1%}) — FN (ini yang kita analisis)")
print(f"\n  Dari {fp+tn:,} pelanggan satisfied di test:")
print(f"    → {fp:,} di-flag salah ({fp/(fp+tn):.1%}) — FP (false alarm rate)")

cm_df = pd.DataFrame({"TP":[tp],"FP":[fp],"FN":[fn],"TN":[tn],
                        "precision":[round(tp/(tp+fp),4)],
                        "recall":[round(tp/(tp+fn),4)],
                        "threshold":[round(threshold,4)]})
cm_df.to_csv(OUT_T / "09_confusion_matrix.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# ANALISIS 1: FN vs TP — Apakah FN didominasi review score 3?
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== ANALISIS 1: REVIEW SCORE DISTRIBUTION — FN vs TP ===")

fn_df = te_meta[te_meta["error_type"] == "FN"].copy()
tp_df = te_meta[te_meta["error_type"] == "TP"].copy()
fp_df = te_meta[te_meta["error_type"] == "FP"].copy()

def score_dist(df, label):
    d = df["review_score"].value_counts(normalize=True).sort_index()
    d.name = label
    return d

fn_scores = score_dist(fn_df, "FN")
tp_scores = score_dist(tp_df, "TP")
score_comp = pd.DataFrame([fn_scores, tp_scores]).T.fillna(0).round(3)
print(score_comp.to_string())

# Pct score=3 in FN vs TP
fn_s3_pct = (fn_df["review_score"] == 3).mean()
tp_s3_pct = (tp_df["review_score"] == 3).mean()
fn_s1_pct = (fn_df["review_score"] == 1).mean()
tp_s1_pct = (tp_df["review_score"] == 1).mean()

print(f"\n  FN: score=3 adalah {fn_s3_pct:.1%} dari FN  |  score=1 adalah {fn_s1_pct:.1%}")
print(f"  TP: score=3 adalah {tp_s3_pct:.1%} dari TP  |  score=1 adalah {tp_s1_pct:.1%}")
print(f"\n  Interpretasi: FN didominasi mild dissatisfaction (score 3), "
      f"TP didominasi severe (score 1/2)")

score_comp.to_csv(OUT_T / "09_fn_tp_score_distribution.csv")

# Grafik
fig, ax = plt.subplots(figsize=(7, 4))
x = [1, 2, 3]
w = 0.35
ax.bar([i - w/2 for i in x], [tp_scores.get(i, 0)*100 for i in x],
       width=w, label="TP (detected)", color="steelblue")
ax.bar([i + w/2 for i in x], [fn_scores.get(i, 0)*100 for i in x],
       width=w, label="FN (missed)", color="tomato")
ax.set_xticks(x); ax.set_xticklabels(["Score 1\n(severe)", "Score 2\n(moderate)", "Score 3\n(mild)"])
ax.set_ylabel("Proportion within group (%)")
ax.set_title("Review Score Distribution: Detected (TP) vs Missed (FN)")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_F / "09_fn_tp_score_dist.png", dpi=120, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# ANALISIS 2: Estimation error — FN vs TP
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== ANALISIS 2: ESTIMATION ERROR — FN vs TP ===")
fn_err = fn_df["estimation_error"].dropna()
tp_err = tp_df["estimation_error"].dropna()

print(f"  FN estimation_error: median={fn_err.median():.2f}d  "
      f"90p={fn_err.quantile(0.9):.2f}d  mean={fn_err.mean():.2f}d")
print(f"  TP estimation_error: median={tp_err.median():.2f}d  "
      f"90p={tp_err.quantile(0.9):.2f}d  mean={tp_err.mean():.2f}d")

# FN yang estimation_error-nya negatif (datang lebih cepat dari estimasi)
# tapi tetap dissatisfied — ini yang paling susah dideteksi model
fn_neg_err = (fn_df["estimation_error"] < 0).mean()
tp_neg_err = (tp_df["estimation_error"] < 0).mean()
print(f"\n  FN dengan estimasi TIDAK terlambat (estimation_error < 0): {fn_neg_err:.1%}")
print(f"  TP dengan estimasi TIDAK terlambat (estimation_error < 0): {tp_neg_err:.1%}")
print(f"  → FN yang datang on-time tapi tetap kecewa: {fn_neg_err:.1%}")
print(f"    Ini menunjukkan ada sumber dissatisfaction di luar delivery timing")

err_summary = pd.DataFrame({
    "group":     ["FN", "TP", "FP"],
    "n":         [len(fn_df), len(tp_df), len(fp_df)],
    "median_err":[fn_err.median(), tp_err.median(),
                  fp_df["estimation_error"].dropna().median()],
    "pct_not_late": [fn_neg_err, tp_neg_err,
                     (fp_df["estimation_error"] < 0).mean()],
})
err_summary = err_summary.round(3)
print(f"\n{err_summary.to_string(index=False)}")
err_summary.to_csv(OUT_T / "09_error_estimation_error.csv", index=False)

# Violin plot
fig, ax = plt.subplots(figsize=(7, 4))
data_plot = [tp_err.clip(-30, 30).values, fn_err.clip(-30, 30).values,
             fp_df["estimation_error"].dropna().clip(-30, 30).values]
vp = ax.violinplot(data_plot, positions=[1, 2, 3], showmedians=True)
ax.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Expectation boundary")
ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["TP\n(detected diss)", "FN\n(missed diss)", "FP\n(false alarm)"])
ax.set_ylabel("Estimation Error (days, clipped ±30)")
ax.set_title("Estimation Error Distribution by Error Type")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_F / "09_error_estimation_error_violin.png", dpi=120, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# ANALISIS 3: FN rate per kategori produk
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== ANALISIS 3: FN RATE PER KATEGORI ===")
# FN rate = FN / (FN + TP) = proporsi dissatisfied yang terlewat per kategori

dissatisfied = te_meta[te_meta["y_true"] == 1].copy()
cat_err = dissatisfied.groupby("top_category").agg(
    n_diss   =("y_true",   "count"),
    n_fn     =("error_type", lambda x: (x == "FN").sum()),
    n_tp     =("error_type", lambda x: (x == "TP").sum()),
).reset_index()
cat_err["fn_rate"] = cat_err["n_fn"] / cat_err["n_diss"]
cat_err = cat_err[cat_err["n_diss"] >= 30].sort_values("fn_rate", ascending=False)

print("Top 10 kategori dengan FN rate tertinggi (>=30 dissatisfied orders):")
print(cat_err.head(10)[["top_category","n_diss","n_fn","fn_rate"]].round(3).to_string(index=False))
print("\nBottom 5 kategori dengan FN rate terendah:")
print(cat_err.tail(5)[["top_category","n_diss","n_fn","fn_rate"]].round(3).to_string(index=False))
cat_err.to_csv(OUT_T / "09_fn_rate_by_category.csv", index=False)

# Grafik top 10 + bottom 5
top_bot = pd.concat([cat_err.head(10), cat_err.tail(5)])
fig, ax = plt.subplots(figsize=(9, 5))
colors = ["tomato" if r > cat_err["fn_rate"].mean() else "steelblue"
          for r in top_bot["fn_rate"]]
ax.barh(top_bot["top_category"], top_bot["fn_rate"] * 100, color=colors)
ax.axvline(cat_err["fn_rate"].mean() * 100, color="black", linestyle="--",
           linewidth=1, label=f"Overall FN rate ({cat_err['fn_rate'].mean():.1%})")
ax.set_xlabel("FN Rate (% dissatisfied orders missed by model)")
ax.set_title("False Negative Rate by Product Category\n(red = above average miss rate)")
ax.invert_yaxis(); ax.legend(fontsize=8)
plt.tight_layout()
fig.savefig(OUT_F / "09_fn_rate_category.png", dpi=120, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# ANALISIS 4: FN rate per state
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== ANALISIS 4: FN RATE PER CUSTOMER STATE ===")
state_err = dissatisfied.groupby("customer_state").agg(
    n_diss=("y_true","count"),
    n_fn  =("error_type", lambda x: (x=="FN").sum()),
).reset_index()
state_err["fn_rate"] = state_err["n_fn"] / state_err["n_diss"]
state_err = state_err[state_err["n_diss"] >= 20].sort_values("fn_rate", ascending=False)
print(state_err.round(3).to_string(index=False))
state_err.to_csv(OUT_T / "09_fn_rate_by_state.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# ANALISIS 5: Profil fitur FN vs TP vs FP
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== ANALISIS 5: PROFIL FITUR — FN vs TP vs FP ===")
NUM_PROFILE = ["estimated_delivery_days", "freight_sum", "freight_to_price_ratio",
               "weight_sum", "distance_km", "n_items",
               "approval_lag_hrs", "carrier_pickup_lag_hrs",
               "estimation_error"]

profile_rows = []
for grp_name, grp_df in [("TP", tp_df), ("FN", fn_df), ("FP", fp_df)]:
    row = {"group": grp_name, "n": len(grp_df)}
    for col in NUM_PROFILE:
        if col in grp_df.columns:
            row[f"med_{col}"] = grp_df[col].median()
    profile_rows.append(row)

profile_df = pd.DataFrame(profile_rows)
# Rapikan kolom nama
profile_df.columns = [c.replace("med_", "") if c.startswith("med_") else c
                       for c in profile_df.columns]
print(profile_df.round(2).to_string(index=False))
profile_df.to_csv(OUT_T / "09_error_profile.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# ANALISIS 6: Probability score distribution
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== ANALISIS 6: PREDICTED PROBABILITY DISTRIBUTION ===")
for grp_name, grp_df in [("TP", tp_df), ("FN", fn_df), ("FP", fp_df), ("TN", te_meta[te_meta["error_type"]=="TN"])]:
    p = grp_df["proba"]
    print(f"  {grp_name}: median={p.median():.3f}  mean={p.mean():.3f}  "
          f"Q1={p.quantile(0.25):.3f}  Q3={p.quantile(0.75):.3f}")

# Grafik distribusi probabilitas
fig, ax = plt.subplots(figsize=(8, 4))
for grp_name, grp_df, color, ls in [
    ("TP", tp_df, "steelblue", "-"),
    ("FN", fn_df, "tomato", "--"),
    ("FP", fp_df, "orange", "-."),
]:
    grp_df["proba"].plot.hist(bins=40, density=True, alpha=0.4,
                               ax=ax, color=color, label=grp_name)
ax.axvline(threshold, color="black", linestyle=":", linewidth=1.5,
           label=f"Threshold ({threshold:.3f})")
ax.set_xlabel("Predicted Probability")
ax.set_title("Score Distribution by Error Type (S2 LR)")
ax.legend()
plt.tight_layout()
fig.savefig(OUT_F / "09_score_distribution.png", dpi=120, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# RINGKASAN UNTUK PAPER
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RINGKASAN ERROR ANALYSIS")
print("=" * 60)

fn_s3 = fn_s3_pct * 100
tp_s3 = tp_s3_pct * 100
fn_notlate = fn_neg_err * 100
overall_fn_rate = fn / (fn + tp) * 100

print(f"""
Model: S2 Logistic Regression (in-fulfillment, broad dissatisfaction)
Threshold: {threshold:.4f} (dari validation, target recall ≥ 0.50)

Confusion Matrix:
  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}
  Precision={tp/(tp+fp):.3f}  Recall={tp/(tp+fn):.3f}

Temuan utama:

1. FN didominasi mild dissatisfaction (score=3):
   - {fn_s3:.1f}% dari FN adalah review score 3
   - vs {tp_s3:.1f}% dari TP yang score 3
   → Model lebih mudah mendeteksi dissatisfaction berat (score 1-2)
     daripada ketidakpuasan ringan (score 3)

2. {fn_notlate:.1f}% dari FN tidak terlambat (estimation_error < 0):
   → Sebagian dissatisfaction terjadi bukan karena keterlambatan
     (mungkin karena kondisi barang, harapan produk, dll)
   → Sinyal fulfillment logistik tidak bisa menangkap semua sumber dissatisfaction

3. FN rate bervariasi antar kategori:
   - Kategori dengan FN rate tertinggi: {cat_err.iloc[0]['top_category']} ({cat_err.iloc[0]['fn_rate']:.1%})
   - Kategori dengan FN rate terendah: {cat_err.iloc[-1]['top_category']} ({cat_err.iloc[-1]['fn_rate']:.1%})
   → Model kurang efektif untuk kategori tertentu

4. FP memiliki estimated probability mendekati threshold:
   → False alarm bukan prediksi yang sangat yakin, melainkan kasus borderline
""")

pd.DataFrame([{
    "total_test": n_test, "n_positive": int(n_pos),
    "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    "precision": round(tp/(tp+fp), 4), "recall": round(tp/(tp+fn), 4),
    "fn_pct_score3": round(fn_s3_pct, 4),
    "fn_pct_not_late": round(fn_neg_err, 4),
    "threshold": round(threshold, 4),
}]).to_csv(OUT_T / "09_error_summary.csv", index=False)

print(f"\nOutput tersimpan di outputs/tables/ dan outputs/figures/")
print("ERROR ANALYSIS SELESAI.")
