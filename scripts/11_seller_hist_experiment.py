"""
11_seller_hist_experiment.py
[EKSPERIMEN] Seller historical features — expanding window + shift 1.

Tujuan: melihat apakah menambahkan track record historis seller
        meningkatkan performa model secara bermakna.

Ini adalah eksperimen terpisah. Hasilnya tidak otomatis masuk paper.
Keputusan penggunaan dibuat setelah melihat hasilnya.

Fitur yang ditambahkan (semua pre-fulfillment — diketahui saat order dibuat):
  - seller_hist_n_orders    : jumlah order prior seller ini
  - seller_hist_diss_rate   : historical broad dissatisfaction rate seller
  - seller_hist_appr_lag    : mean approval lag seller dari order-order prior
  - seller_is_new           : 1 jika hist_n_orders < 5 (cold start flag)

Leakage guard:
  - Expanding window di-shift 1 → order saat ini tidak diikutkan
  - Cold start imputation dari TRAIN set saja
  - Timestamp sort dilakukan sebelum cumulative computation

Output:
  outputs/tables/11_seller_hist_metrics.csv
  outputs/tables/11_seller_hist_delong.csv
  outputs/figures/11_seller_hist_comparison.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from scipy import stats as scipy_stats
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (roc_auc_score, average_precision_score,
                                   brier_score_loss, recall_score,
                                   precision_score, f1_score,
                                   precision_recall_curve)
import lightgbm as lgb

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
RAW     = ROOT / "data_raw"
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
CAT_COLS    = ["customer_state", "seller_state", "payment_type_main", "top_category"]
NEW_FEATS   = ["seller_hist_n_orders", "seller_hist_diss_rate",
               "seller_hist_appr_lag", "seller_is_new"]
COLD_START_N = 5   # seller dengan < N order prior dianggap "baru"

# ── Helpers ────────────────────────────────────────────────────────────────────
def get_feat_cols(df):
    return [c for c in df.columns if c not in META and c not in BLACKLIST]

def encode_cat(tr, va, te, cats, top_n=15):
    for col in cats:
        if col not in tr.columns: continue
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

def get_threshold(y_val, proba_val, target_recall=0.50):
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    valid = [(t, r, p) for t, r, p in zip(thr, rec[:-1], prec[:-1])
             if r >= target_recall]
    return max(valid, key=lambda x: x[0])[0] if valid else 0.5

def eval_model(name, y_true, proba, y_val, proba_val):
    thr    = get_threshold(y_val, proba_val)
    y_pred = (proba >= thr).astype(int)
    n      = len(y_true); total_pos = y_true.sum()
    row = {
        "model":     name,
        "roc_auc":   round(roc_auc_score(y_true, proba), 4),
        "pr_auc":    round(average_precision_score(y_true, proba), 4),
        "brier":     round(brier_score_loss(y_true, proba), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "threshold": round(thr, 4),
    }
    for pct in [5, 10, 20, 30]:
        k = int(n * pct / 100)
        top_idx = np.argsort(proba)[::-1][:k]
        row[f"top{pct}pct"] = round(y_true[top_idx].sum() / total_pos, 4)
    return row

# ── DeLong (disederhanakan untuk perbandingan dua model) ──────────────────────
def delong_2model(y_true, proba_a, proba_b, chunk=2000):
    y = np.asarray(y_true, int)
    a = np.asarray(proba_a, float)
    b = np.asarray(proba_b, float)
    n1 = y.sum(); n0 = len(y) - n1
    pos = a[y==1]; neg = a[y==0]
    V10a = np.array([((pos[i:i+chunk,None] > neg[None,:]).mean(1) +
                      0.5*(pos[i:i+chunk,None] == neg[None,:]).mean(1)).mean()
                     for i in range(0, n1, chunk)])
    # rebuild properly
    def sc(scores, pos_s, neg_s, ch=chunk):
        n1_ = len(pos_s); n0_ = len(neg_s)
        V10 = np.empty(n1_)
        for i in range(0, n1_, ch):
            b_ = pos_s[i:i+ch, None]
            V10[i:i+ch] = (b_ > neg_s).mean(1) + 0.5*(b_ == neg_s).mean(1)
        V01 = np.empty(n0_)
        for j in range(0, n0_, ch):
            b_ = neg_s[j:j+ch, None]
            V01[j:j+ch] = (pos_s > b_).mean(1) + 0.5*(pos_s == b_).mean(1)
        return V10, V01

    pos_a = a[y==1]; neg_a = a[y==0]
    pos_b = b[y==1]; neg_b = b[y==0]
    V10_a, V01_a = sc(a, pos_a, neg_a)
    V10_b, V01_b = sc(b, pos_b, neg_b)
    auc_a = V10_a.mean(); auc_b = V10_b.mean()
    var_a  = np.var(V10_a,ddof=1)/n1 + np.var(V01_a,ddof=1)/n0
    var_b  = np.var(V10_b,ddof=1)/n1 + np.var(V01_b,ddof=1)/n0
    cov_ab = np.cov(V10_a,V10_b,ddof=1)[0,1]/n1 + np.cov(V01_a,V01_b,ddof=1)[0,1]/n0
    var_d  = var_a + var_b - 2*cov_ab
    if var_d <= 0: return auc_a, auc_b, 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var_d)
    p = float(2*(1 - scipy_stats.norm.cdf(abs(z))))
    return float(auc_a), float(auc_b), float(z), p

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Bangun seller historical features
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("[EKSPERIMEN] Seller Historical Features")
print("=" * 60)

print("\nStep 1: Membangun seller historical features...")

DATE_COLS = ["order_purchase_timestamp","order_approved_at",
             "order_delivered_carrier_date","order_delivered_customer_date",
             "order_estimated_delivery_date"]
orders = pd.read_csv(INTERIM / "orders_filtered.csv",
                     parse_dates=DATE_COLS, low_memory=False)
items  = pd.read_csv(RAW / "olist_order_items_dataset.csv", low_memory=False)

# Seller representative: seller dengan price tertinggi per order
seller_rep = (items.sort_values("price", ascending=False)
              .drop_duplicates("order_id", keep="first")
              [["order_id", "seller_id"]])

# Approval lag dari kolom yang ada di orders
orders["approval_lag_hrs"] = (
    (orders["order_approved_at"] - orders["order_purchase_timestamp"])
    .dt.total_seconds() / 3600
)

# Gabung seller_id
df_hist = orders[["order_id","split","order_purchase_timestamp",
                   "target_broad","approval_lag_hrs"]].copy()
df_hist = df_hist.merge(seller_rep, on="order_id", how="left")

print(f"  Orders: {len(df_hist):,}  |  Sellers unik: {df_hist['seller_id'].nunique():,}")
print(f"  Seller NaN: {df_hist['seller_id'].isna().sum()}")

# Sort berdasarkan timestamp — KRITIS untuk expanding window yang benar
df_hist = df_hist.sort_values("order_purchase_timestamp").reset_index(drop=True)

# Expanding window per seller (shift 1 — tidak ikutkan order saat ini)
print("  Menghitung expanding window per seller...")
df_hist["seller_hist_n_orders"] = df_hist.groupby("seller_id").cumcount()

df_hist["seller_hist_diss_rate"] = (
    df_hist.groupby("seller_id")["target_broad"]
    .transform(lambda x: x.cumsum().shift(1))
    /
    df_hist.groupby("seller_id").cumcount().replace(0, np.nan)
)

df_hist["seller_hist_appr_lag"] = (
    df_hist.groupby("seller_id")["approval_lag_hrs"]
    .transform(lambda x: x.expanding().mean().shift(1))
)

df_hist["seller_is_new"] = (
    df_hist["seller_hist_n_orders"] < COLD_START_N
).astype(float)

# Verifikasi: baris pertama per seller harus punya hist_n=0 dan diss_rate=NaN
n_first_orders = (df_hist["seller_hist_n_orders"] == 0).sum()
n_nan_diss     = df_hist["seller_hist_diss_rate"].isna().sum()
print(f"  First-order rows (hist_n=0): {n_first_orders:,}")
print(f"  NaN diss_rate (sebelum impute): {n_nan_diss:,}")
print(f"  seller_is_new (hist_n<{COLD_START_N}): {df_hist['seller_is_new'].sum():,.0f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Cold start imputation — dari TRAIN set saja
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 2: Cold start imputation dari train set...")

# Hanya pakai seller yang sudah punya histori di train (hist_n >= COLD_START_N)
train_mature = df_hist[
    (df_hist["split"] == "train") &
    (df_hist["seller_hist_n_orders"] >= COLD_START_N) &
    df_hist["seller_hist_diss_rate"].notna()
]
fill_diss = train_mature["seller_hist_diss_rate"].mean()
fill_lag  = train_mature["seller_hist_appr_lag"].dropna().mean()

print(f"  Fill value seller_hist_diss_rate : {fill_diss:.4f}")
print(f"  Fill value seller_hist_appr_lag  : {fill_lag:.2f} hrs")

df_hist["seller_hist_diss_rate"] = df_hist["seller_hist_diss_rate"].fillna(fill_diss)
df_hist["seller_hist_appr_lag"]  = df_hist["seller_hist_appr_lag"].fillna(fill_lag)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Merge ke in-fulfillment feature table
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 3: Merge ke in-fulfillment features...")

inf = pd.read_csv(INTERIM / "order_level_features_in.csv", low_memory=False)
seller_hist_cols = ["order_id"] + NEW_FEATS
inf_extended = inf.merge(
    df_hist[seller_hist_cols], on="order_id", how="left"
)

# Verifikasi: tidak boleh ada NaN di fitur baru (setelah impute)
for feat in NEW_FEATS:
    n_nan = inf_extended[feat].isna().sum()
    if n_nan > 0:
        print(f"  WARNING: {feat} masih punya {n_nan} NaN — diisi median")
        inf_extended[feat] = inf_extended[feat].fillna(inf_extended[feat].median())
    else:
        print(f"  {feat}: OK (0 NaN)")

# Distribution check
print(f"\n  seller_hist_n_orders: median={inf_extended['seller_hist_n_orders'].median():.0f}  "
      f"90p={inf_extended['seller_hist_n_orders'].quantile(0.9):.0f}")
print(f"  seller_hist_diss_rate: mean={inf_extended['seller_hist_diss_rate'].mean():.3f}  "
      f"std={inf_extended['seller_hist_diss_rate'].std():.3f}")
print(f"  seller_is_new: {inf_extended['seller_is_new'].mean():.2%} dari orders")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Preprocessing splits
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 4: Preprocessing...")

# Tanpa seller hist (baseline)
FEAT_BASE = [c for c in inf.columns if c not in META and c not in BLACKLIST]
# Dengan seller hist
FEAT_EXT  = FEAT_BASE + NEW_FEATS
NUM_BASE  = [c for c in FEAT_BASE if c not in CAT_COLS]
NUM_EXT   = [c for c in FEAT_EXT  if c not in CAT_COLS]

def make_splits(df, feat_cols, num_cols):
    tr = df[df["split"]=="train"].reset_index(drop=True)
    va = df[df["split"]=="validation"].reset_index(drop=True)
    te = df[df["split"]=="test"].reset_index(drop=True)
    Xtr = tr[feat_cols].copy()
    Xva = va[feat_cols].copy()
    Xte = te[feat_cols].copy()
    for col in num_cols:
        if col not in Xtr.columns: continue
        med = Xtr[col].median()
        for d in [Xtr,Xva,Xte]: d[col] = d[col].fillna(med)
    for col in CAT_COLS:
        if col not in Xtr.columns: continue
        for d in [Xtr,Xva,Xte]: d[col] = d[col].fillna("Unknown")
    Xtr, Xva, Xte = encode_cat(Xtr, Xva, Xte, CAT_COLS)
    Xtr = Xtr.fillna(0).astype(float)
    Xva = Xva.fillna(0).astype(float)
    Xte = Xte.fillna(0).astype(float)
    ytr = tr["target_broad"].values
    yva = va["target_broad"].values
    yte = te["target_broad"].values
    return Xtr, Xva, Xte, ytr, yva, yte

# Splits baseline
Xtr_b, Xva_b, Xte_b, ytr, yva, yte = make_splits(inf, FEAT_BASE, NUM_BASE)
# Splits extended
Xtr_e, Xva_e, Xte_e, _,   _,   _   = make_splits(inf_extended, FEAT_EXT, NUM_EXT)

print(f"  Baseline features : {Xtr_b.shape[1]}")
print(f"  Extended features : {Xtr_e.shape[1]}")

# Load best params dari Optuna (dari script 10)
try:
    bp_df = pd.read_csv(OUT_T / "10_best_params.csv")
    bp = bp_df.iloc[0].to_dict()
    n_est = int(bp.pop("n_estimators", 475))
    bp.pop("best_val_auc", None)
    for p in ["num_leaves","max_depth","min_child_samples"]:
        if p in bp: bp[p] = int(bp[p])
    print(f"  Optuna best params loaded (n_est={n_est})")
except FileNotFoundError:
    print("  10_best_params.csv tidak ditemukan — pakai params manual")
    n_est = 475
    bp = {"learning_rate":0.025,"num_leaves":16,"max_depth":6,
          "min_child_samples":93,"subsample":0.853,"colsample_bytree":0.533,
          "reg_alpha":0.004,"reg_lambda":0.010}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Training & Evaluation
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 5: Training models...")
results = []
probas  = {}   # simpan untuk DeLong

# LR scaling helper
def lr_scale(Xtr, Xva, Xte, num_cols):
    num_in = [c for c in num_cols if c in Xtr.columns]
    sc = StandardScaler()
    Xtr_lr = Xtr.copy(); Xva_lr = Xva.copy(); Xte_lr = Xte.copy()
    Xtr_lr[num_in] = sc.fit_transform(Xtr[num_in])
    Xva_lr[num_in] = sc.transform(Xva[num_in])
    Xte_lr[num_in] = sc.transform(Xte[num_in])
    return Xtr_lr, Xva_lr, Xte_lr

# ── LR baseline (no seller hist) ──────────────────────────────────────────────
print("  [1/4] LR baseline...")
Xtr_lr, Xva_lr, Xte_lr = lr_scale(Xtr_b, Xva_b, Xte_b, NUM_BASE)
lr_base = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                              random_state=42, solver="lbfgs")
lr_base.fit(Xtr_lr, ytr)
p_lr_va = lr_base.predict_proba(Xva_lr)[:,1]
p_lr_te = lr_base.predict_proba(Xte_lr)[:,1]
r = eval_model("LR (baseline, no seller hist)", yte, p_lr_te, yva, p_lr_va)
results.append(r); probas["LR_base"] = p_lr_te
print(f"    AUC={r['roc_auc']}  PR-AUC={r['pr_auc']}  Top10%={r['top10pct']}")

# ── LR + seller hist ──────────────────────────────────────────────────────────
print("  [2/4] LR + seller hist...")
Xtr_lre, Xva_lre, Xte_lre = lr_scale(Xtr_e, Xva_e, Xte_e, NUM_EXT)
lr_ext = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                             random_state=42, solver="lbfgs")
lr_ext.fit(Xtr_lre, ytr)
p_lre_va = lr_ext.predict_proba(Xva_lre)[:,1]
p_lre_te = lr_ext.predict_proba(Xte_lre)[:,1]
r = eval_model("LR + seller hist", yte, p_lre_te, yva, p_lre_va)
results.append(r); probas["LR_ext"] = p_lre_te
print(f"    AUC={r['roc_auc']}  PR-AUC={r['pr_auc']}  Top10%={r['top10pct']}")

# ── LGBM tuned baseline ───────────────────────────────────────────────────────
print("  [3/4] LGBM tuned baseline (no seller hist)...")
lgbm_base = lgb.LGBMClassifier(n_estimators=n_est, **bp,
                                 class_weight="balanced",
                                 random_state=42, n_jobs=-1, verbose=-1)
lgbm_base.fit(Xtr_b, ytr,
              eval_set=[(Xva_b, yva)],
              callbacks=[lgb.early_stopping(50,verbose=False),
                         lgb.log_evaluation(-1)])
p_lgb_va = lgbm_base.predict_proba(Xva_b)[:,1]
p_lgb_te = lgbm_base.predict_proba(Xte_b)[:,1]
r = eval_model("LGBM tuned (no seller hist)", yte, p_lgb_te, yva, p_lgb_va)
results.append(r); probas["LGBM_base"] = p_lgb_te
print(f"    AUC={r['roc_auc']}  PR-AUC={r['pr_auc']}  Top10%={r['top10pct']}")

# ── LGBM tuned + seller hist ──────────────────────────────────────────────────
print("  [4/4] LGBM tuned + seller hist...")
lgbm_ext = lgb.LGBMClassifier(n_estimators=n_est, **bp,
                                class_weight="balanced",
                                random_state=42, n_jobs=-1, verbose=-1)
lgbm_ext.fit(Xtr_e, ytr,
             eval_set=[(Xva_e, yva)],
             callbacks=[lgb.early_stopping(50,verbose=False),
                        lgb.log_evaluation(-1)])
p_lgbe_va = lgbm_ext.predict_proba(Xva_e)[:,1]
p_lgbe_te = lgbm_ext.predict_proba(Xte_e)[:,1]
r = eval_model("LGBM tuned + seller hist", yte, p_lgbe_te, yva, p_lgbe_va)
results.append(r); probas["LGBM_ext"] = p_lgbe_te
print(f"    AUC={r['roc_auc']}  PR-AUC={r['pr_auc']}  Top10%={r['top10pct']}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: DeLong test — apakah gain signifikan?
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 6: DeLong test...")

delong_rows = []
comparisons = [
    ("LR: base vs +seller",   "LR_ext",   "LR_base"),
    ("LGBM: base vs +seller", "LGBM_ext", "LGBM_base"),
]
for label, ka, kb in comparisons:
    auc_a, auc_b, z, p = delong_2model(yte, probas[ka], probas[kb])
    sig = "***" if p<0.001 else ("**" if p<0.01 else ("*" if p<0.05 else "ns"))
    delta = auc_a - auc_b
    print(f"  {label}: Δ={delta:+.4f}  z={z:.3f}  p={p:.4f}  {sig}")
    delong_rows.append({"comparison":label,"auc_with":round(auc_a,4),
                        "auc_without":round(auc_b,4),"delta":round(delta,4),
                        "z":round(z,3),"p_value":round(p,4),"sig":sig})

delong_df = pd.DataFrame(delong_rows)
delong_df.to_csv(OUT_T / "11_seller_hist_delong.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: Seller hist feature importance (LGBM)
# ─────────────────────────────────────────────────────────────────────────────
print("\nStep 7: Feature importance seller hist di LGBM...")
feat_names = Xtr_e.columns.tolist()
fi = pd.Series(lgbm_ext.feature_importances_, index=feat_names).sort_values(ascending=False)
print("  Top 10 features (LGBM + seller hist):")
for fname, fval in fi.head(10).items():
    marker = " ←" if fname in NEW_FEATS else ""
    print(f"    {fname:<35} {fval:.0f}{marker}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8: Output dan interpretasi
# ─────────────────────────────────────────────────────────────────────────────
comp_df = pd.DataFrame(results)
col_order = ["model","roc_auc","pr_auc","recall","precision","f1",
             "top5pct","top10pct","top20pct","top30pct","threshold"]
col_order = [c for c in col_order if c in comp_df.columns]

print("\n" + "=" * 65)
print("HASIL EKSPERIMEN — Seller Historical Features")
print("=" * 65)
print(comp_df[col_order].to_string(index=False))
comp_df[col_order].to_csv(OUT_T / "11_seller_hist_metrics.csv", index=False)

# Bar chart
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
models_short = ["LR\n(base)", "LR\n+seller", "LGBM\n(base)", "LGBM\n+seller"]
colors       = ["#4C72B0","#55A868","#C44E52","#DD8452"]

for ax, metric, title, ylim in zip(
    axes,
    ["roc_auc","pr_auc","top10pct"],
    ["ROC-AUC","PR-AUC","Top-10% Capture"],
    [(0.58,0.68),(0.28,0.38),(0.17,0.24)]
):
    vals = comp_df[metric].values
    bars = ax.bar(models_short, vals, color=colors)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.001,
                f"{v:.4f}", ha="center", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)

plt.suptitle("[EKSPERIMEN] Dampak Seller Historical Features\n"
             "(train-only, fair comparison)", fontsize=11)
plt.tight_layout()
fig.savefig(OUT_F / "11_seller_hist_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# Interpretasi otomatis
print("\n=== INTERPRETASI ===")
lr_gain   = probas["LR_ext"];   lr_base_  = probas["LR_base"]
lgbm_gain = probas["LGBM_ext"]; lgbm_base_= probas["LGBM_base"]
lr_delta   = roc_auc_score(yte, lr_gain) - roc_auc_score(yte, lr_base_)
lgbm_delta = roc_auc_score(yte, lgbm_gain) - roc_auc_score(yte, lgbm_base_)
lr_sig     = delong_df[delong_df["comparison"].str.startswith("LR")]["sig"].values[0]
lgbm_sig   = delong_df[delong_df["comparison"].str.startswith("LGBM")]["sig"].values[0]

for model, delta, sig in [("LR", lr_delta, lr_sig), ("LGBM", lgbm_delta, lgbm_sig)]:
    if sig in ["***","**","*"] and delta >= 0.005:
        verdict = f"SIGNIFIKAN dan SUBSTANTIF → layak dipertimbangkan untuk paper"
    elif sig in ["***","**","*"]:
        verdict = f"signifikan secara statistik tapi gain kecil ({delta:+.4f})"
    else:
        verdict = f"tidak signifikan (ns) → seller hist tidak menambah nilai"
    print(f"  {model} + seller hist: Δ={delta:+.4f}  {sig}  → {verdict}")

print(f"\nSaved: {OUT_T / '11_seller_hist_metrics.csv'}")
print(f"       {OUT_T / '11_seller_hist_delong.csv'}")
print("\nEKSPERIMEN SELESAI.")
