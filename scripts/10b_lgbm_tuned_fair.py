"""
10b_lgbm_tuned_fair.py
Rerun LightGBM tuned dengan best params dari 10_lgbm_tuning.py,
tapi dilatih hanya pada TRAIN SET (bukan train+val) agar threshold
bisa dipilih secara benar dari validation yang benar-benar held-out.

Tanpa ini, perbandingan recall/F1 dengan LR tidak valid.

Baca best params dari: outputs/tables/10_best_params.csv
Output: outputs/tables/10b_fair_comparison.csv
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
from sklearn.metrics       import (roc_auc_score, average_precision_score,
                                   brier_score_loss, recall_score,
                                   precision_score, f1_score,
                                   precision_recall_curve)
import lightgbm as lgb

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

def eval_full(model_name, y_true, proba, y_val, proba_val):
    thr = get_threshold(y_val, proba_val)
    y_pred = (proba >= thr).astype(int)
    row = {
        "model":     model_name,
        "roc_auc":   round(roc_auc_score(y_true, proba), 4),
        "pr_auc":    round(average_precision_score(y_true, proba), 4),
        "brier":     round(brier_score_loss(y_true, proba), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "threshold": round(thr, 4),
    }
    # Top-k
    n = len(y_true); total_pos = y_true.sum()
    for pct in [5, 10, 20, 30]:
        k = int(n * pct / 100)
        top_idx = np.argsort(proba)[::-1][:k]
        row[f"top{pct}pct"] = round(y_true[top_idx].sum() / total_pos, 4)
    return row

# ── Load & preprocess ──────────────────────────────────────────────────────────
print("Loading data...")
inf = pd.read_csv(INTERIM / "order_level_features_in.csv", low_memory=False)
FEAT_COLS = get_feat_cols(inf)
NUM_COLS  = [c for c in FEAT_COLS if c not in CAT_COLS]

tr = inf[inf["split"] == "train"].reset_index(drop=True)
va = inf[inf["split"] == "validation"].reset_index(drop=True)
te = inf[inf["split"] == "test"].reset_index(drop=True)

Xtr = tr[FEAT_COLS].copy(); Xva = va[FEAT_COLS].copy(); Xte = te[FEAT_COLS].copy()
for col in NUM_COLS:
    if col not in Xtr.columns: continue
    med = Xtr[col].median()
    for d in [Xtr, Xva, Xte]: d[col] = d[col].fillna(med)
for col in CAT_COLS:
    if col not in Xtr.columns: continue
    for d in [Xtr, Xva, Xte]: d[col] = d[col].fillna("Unknown")
Xtr, Xva, Xte = encode_cat(Xtr, Xva, Xte, CAT_COLS)
Xtr = Xtr.fillna(0).astype(float)
Xva = Xva.fillna(0).astype(float)
Xte = Xte.fillna(0).astype(float)

ytr = tr["target_broad"].values
yva = va["target_broad"].values
yte = te["target_broad"].values

# ── Load best params dari Optuna ───────────────────────────────────────────────
print("Loading best params from Optuna...")
best_params_df = pd.read_csv(OUT_T / "10_best_params.csv")
best_params = best_params_df.iloc[0].to_dict()

# Pisahkan n_estimators dari params lain
n_est = int(best_params.pop("n_estimators", 500))
best_val_auc = best_params.pop("best_val_auc", None)
# Pastikan tipe data benar
int_params = ["num_leaves", "max_depth", "min_child_samples"]
for p in int_params:
    if p in best_params:
        best_params[p] = int(best_params[p])

print(f"  n_estimators: {n_est}")
print(f"  params: {best_params}")

results = []

# ── 1. LR (benchmark, train only) ─────────────────────────────────────────────
print("\n--- LR (train only) ---")
num_in_feat = [c for c in NUM_COLS if c in Xtr.columns]
scaler = StandardScaler()
Xtr_lr = Xtr.copy(); Xva_lr = Xva.copy(); Xte_lr = Xte.copy()
Xtr_lr[num_in_feat] = scaler.fit_transform(Xtr[num_in_feat])
Xva_lr[num_in_feat] = scaler.transform(Xva[num_in_feat])
Xte_lr[num_in_feat] = scaler.transform(Xte[num_in_feat])

lr = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                         random_state=42, solver="lbfgs")
lr.fit(Xtr_lr, ytr)
p_lr_va = lr.predict_proba(Xva_lr)[:, 1]
p_lr_te = lr.predict_proba(Xte_lr)[:, 1]
r = eval_full("LR", yte, p_lr_te, yva, p_lr_va)
results.append(r)
print(f"  AUC={r['roc_auc']}  PR-AUC={r['pr_auc']}  "
      f"Recall={r['recall']}  F1={r['f1']}  threshold={r['threshold']}")

# ── 2. LightGBM Default (train only, untuk referensi) ─────────────────────────
print("\n--- LightGBM Default (train only) ---")
lgbm_def = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
)
lgbm_def.fit(Xtr, ytr)
p_def_va = lgbm_def.predict_proba(Xva)[:, 1]
p_def_te = lgbm_def.predict_proba(Xte)[:, 1]
r = eval_full("LightGBM (default)", yte, p_def_te, yva, p_def_va)
results.append(r)
print(f"  AUC={r['roc_auc']}  PR-AUC={r['pr_auc']}  "
      f"Recall={r['recall']}  F1={r['f1']}  threshold={r['threshold']}")

# ── 3. LightGBM Tuned (train only — fair comparison) ──────────────────────────
print("\n--- LightGBM Tuned (train only, FAIR) ---")
lgbm_tuned = lgb.LGBMClassifier(
    n_estimators=n_est,
    **best_params,
    class_weight="balanced",
    random_state=42, n_jobs=-1, verbose=-1
)
lgbm_tuned.fit(Xtr, ytr,
               eval_set=[(Xva, yva)],
               callbacks=[lgb.early_stopping(50, verbose=False),
                          lgb.log_evaluation(-1)])
p_tuned_va = lgbm_tuned.predict_proba(Xva)[:, 1]
p_tuned_te = lgbm_tuned.predict_proba(Xte)[:, 1]
r = eval_full("LightGBM (tuned, train-only)", yte, p_tuned_te, yva, p_tuned_va)
results.append(r)
print(f"  AUC={r['roc_auc']}  PR-AUC={r['pr_auc']}  "
      f"Recall={r['recall']}  F1={r['f1']}  threshold={r['threshold']}")

# ── Tabel final ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PERBANDINGAN FAIR (semua model dilatih pada train set saja)")
print("=" * 65)

comp_df = pd.DataFrame(results)
col_order = ["model","roc_auc","pr_auc","recall","precision","f1",
             "brier","top5pct","top10pct","top20pct","top30pct","threshold"]
col_order = [c for c in col_order if c in comp_df.columns]
print(comp_df[col_order].to_string(index=False))

comp_df[col_order].to_csv(OUT_T / "10b_fair_comparison.csv", index=False)

# ── Bar chart perbandingan AUC ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

models  = comp_df["model"].tolist()
colors  = ["steelblue", "gray", "tomato"]

axes[0].bar(models, comp_df["roc_auc"], color=colors)
axes[0].set_ylim(0.55, 0.70)
axes[0].set_title("ROC-AUC (fair comparison)")
axes[0].set_ylabel("AUC")
for i, v in enumerate(comp_df["roc_auc"]):
    axes[0].text(i, v + 0.001, str(v), ha="center", fontsize=9)
axes[0].tick_params(axis="x", rotation=15)

axes[1].bar(models, comp_df["top10pct"], color=colors)
axes[1].set_ylim(0.15, 0.25)
axes[1].set_title("Top-10% Capture Rate (fair comparison)")
axes[1].set_ylabel("Recall at top 10%")
for i, v in enumerate(comp_df["top10pct"]):
    axes[1].text(i, v + 0.001, str(v), ha="center", fontsize=9)
axes[1].tick_params(axis="x", rotation=15)

plt.suptitle("Fair Model Comparison — S2 In-Fulfillment, Broad Dissatisfaction")
plt.tight_layout()
fig.savefig(OUT_F / "10b_fair_comparison.png", dpi=120, bbox_inches="tight")
plt.close()

# ── Interpretasi ───────────────────────────────────────────────────────────────
print("\n=== INTERPRETASI FINAL ===")
lr_auc     = comp_df[comp_df["model"]=="LR"]["roc_auc"].values[0]
tuned_auc  = comp_df[comp_df["model"].str.contains("tuned")]["roc_auc"].values[0]
tuned_rec  = comp_df[comp_df["model"].str.contains("tuned")]["recall"].values[0]
lr_rec     = comp_df[comp_df["model"]=="LR"]["recall"].values[0]
gap        = tuned_auc - lr_auc

print(f"  LightGBM tuned vs LR : Δ AUC = {gap:+.4f}")
if gap >= 0.01:
    print(f"\n  → LightGBM tuned melampaui LR (Δ={gap:.4f}).")
    print(f"  → Namun recall LightGBM tuned ({tuned_rec:.3f}) vs LR ({lr_rec:.3f})")
    print(f"    perlu dipertimbangkan sesuai kebutuhan operasional.")
    print(f"\n  REKOMENDASI PAPER:")
    print(f"  Laporkan keduanya: LR untuk interpretability,")
    print(f"  LightGBM tuned untuk AUC maksimal.")
    print(f"  Gunakan top-k capture sebagai metrik operasional utama.")
elif gap >= 0.003:
    print(f"\n  → Gain kecil tapi positif (+{gap:.4f}).")
    print(f"  → Perbedaan tidak substansial. LR tetap direkomendasikan")
    print(f"    sebagai primary model karena interpretability lebih baik.")
else:
    print(f"\n  → Tidak ada perbedaan substantif ({gap:+.4f}).")
    print(f"  → LR adalah primary model yang tepat.")
    print(f"  → Best params Optuna (num_leaves=16) menunjukkan bahwa")
    print(f"    LightGBM optimal pun menyerupai model sederhana —")
    print(f"    konsisten dengan LR yang kompetitif.")

print(f"\nSaved: {OUT_T / '10b_fair_comparison.csv'}")
print("FAIR COMPARISON SELESAI.")
