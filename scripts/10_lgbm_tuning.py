"""
10_lgbm_tuning.py
Tuning hyperparameter LightGBM dengan Optuna untuk S2 (in-fulfillment, broad).
Perbandingan: LightGBM default vs LightGBM tuned vs LR (baseline terbaik).

Output: outputs/tables/10_tuning_results.csv
        outputs/tables/10_tuning_comparison.csv
        outputs/figures/10_optuna_history.png
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
                                   brier_score_loss, precision_recall_curve)
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# ── Load ───────────────────────────────────────────────────────────────────────
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

print(f"  Train: {len(ytr):,}  Val: {len(yva):,}  Test: {len(yte):,}")
print(f"  Features: {len(FEAT_COLS)}")

# ── Helper: threshold + metrics ────────────────────────────────────────────────
def get_threshold(y_val, proba_val, target_recall=0.50):
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    valid = [(t, r, p) for t, r, p in zip(thr, rec[:-1], prec[:-1]) if r >= target_recall]
    return max(valid, key=lambda x: x[0])[0] if valid else 0.5

def eval_model(y_true, proba, threshold):
    from sklearn.metrics import recall_score, precision_score, f1_score
    y_pred = (proba >= threshold).astype(int)
    return {
        "roc_auc":  round(roc_auc_score(y_true, proba), 4),
        "pr_auc":   round(average_precision_score(y_true, proba), 4),
        "brier":    round(brier_score_loss(y_true, proba), 4),
        "recall":   round(recall_score(y_true, y_pred, zero_division=0), 4),
        "precision":round(precision_score(y_true, y_pred, zero_division=0), 4),
        "f1":       round(f1_score(y_true, y_pred, zero_division=0), 4),
        "threshold":round(threshold, 4),
    }

def topk_recall(y_true, proba, pct):
    k = int(len(y_true) * pct / 100)
    top_idx = np.argsort(proba)[::-1][:k]
    return round(y_true[top_idx].sum() / y_true.sum(), 4) if y_true.sum() > 0 else 0

# ── Baseline: LR (best model so far) ──────────────────────────────────────────
print("\n--- Baseline: Logistic Regression ---")
num_in_feat = [c for c in NUM_COLS if c in Xtr.columns]
scaler = StandardScaler()
Xtr_lr = Xtr.copy(); Xva_lr = Xva.copy(); Xte_lr = Xte.copy()
Xtr_lr[num_in_feat] = scaler.fit_transform(Xtr[num_in_feat])
Xva_lr[num_in_feat] = scaler.transform(Xva[num_in_feat])
Xte_lr[num_in_feat] = scaler.transform(Xte[num_in_feat])

lr = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                         random_state=42, solver="lbfgs")
lr.fit(Xtr_lr, ytr)
proba_lr_va = lr.predict_proba(Xva_lr)[:, 1]
proba_lr_te = lr.predict_proba(Xte_lr)[:, 1]
thr_lr = get_threshold(yva, proba_lr_va)
metrics_lr = eval_model(yte, proba_lr_te, thr_lr)
metrics_lr["model"] = "LR (baseline)"
print(f"  AUC={metrics_lr['roc_auc']}  PR-AUC={metrics_lr['pr_auc']}  "
      f"Recall={metrics_lr['recall']}  F1={metrics_lr['f1']}")

# ── LightGBM Default (dari script 04) ─────────────────────────────────────────
print("\n--- LightGBM Default ---")
lgbm_default = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
)
lgbm_default.fit(Xtr, ytr)
proba_def_va = lgbm_default.predict_proba(Xva)[:, 1]
proba_def_te = lgbm_default.predict_proba(Xte)[:, 1]
thr_def = get_threshold(yva, proba_def_va)
metrics_def = eval_model(yte, proba_def_te, thr_def)
metrics_def["model"] = "LightGBM (default)"
print(f"  AUC={metrics_def['roc_auc']}  PR-AUC={metrics_def['pr_auc']}  "
      f"Recall={metrics_def['recall']}  F1={metrics_def['f1']}")

# ── Optuna Tuning ─────────────────────────────────────────────────────────────
print("\n--- Optuna Tuning (50 trials) ---")
N_TRIALS = 50

def objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 200, 1500),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 16, 128),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 100),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "class_weight": "balanced",
        "random_state":     42,
        "n_jobs":          -1,
        "verbose":         -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(Xtr, ytr,
              eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(-1)])
    proba_va = model.predict_proba(Xva)[:, 1]
    return roc_auc_score(yva, proba_va)

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best_params = study.best_params
best_val_auc = study.best_value
print(f"  Best val AUC: {best_val_auc:.4f}")
print(f"  Best params: {best_params}")

# Optuna history plot
trial_aucs = [t.value for t in study.trials if t.value is not None]
best_so_far = [max(trial_aucs[:i+1]) for i in range(len(trial_aucs))]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1, len(trial_aucs)+1), trial_aucs, "o", alpha=0.4,
        color="gray", markersize=3, label="Trial AUC (val)")
ax.plot(range(1, len(best_so_far)+1), best_so_far, "-",
        color="steelblue", linewidth=2, label="Best so far")
ax.axhline(metrics_lr["roc_auc"], color="tomato", linestyle="--",
           linewidth=1.5, label=f"LR baseline ({metrics_lr['roc_auc']})")
ax.set_xlabel("Trial"); ax.set_ylabel("Validation AUC")
ax.set_title("Optuna Tuning History — LightGBM (S2 in-fulfillment)")
ax.legend(); plt.tight_layout()
fig.savefig(OUT_F / "10_optuna_history.png", dpi=120, bbox_inches="tight")
plt.close()

# ── LightGBM Tuned: retrain dengan best params di train+val ───────────────────
print("\n--- LightGBM Tuned: final training ---")
# Gabung train+val untuk final model (test tetap untouched)
Xtr_full = pd.concat([Xtr, Xva]).reset_index(drop=True)
ytr_full  = np.concatenate([ytr, yva])

best_params_clean = {k: v for k, v in best_params.items()
                     if k != "n_estimators"}
lgbm_tuned = lgb.LGBMClassifier(
    **best_params_clean,
    n_estimators=best_params.get("n_estimators", 1000),
    class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1
)
lgbm_tuned.fit(Xtr_full, ytr_full)
proba_tuned_te = lgbm_tuned.predict_proba(Xte)[:, 1]

# Threshold dari validation saja (bukan full)
proba_tuned_va = lgbm_tuned.predict_proba(Xva)[:, 1]
thr_tuned = get_threshold(yva, proba_tuned_va)
metrics_tuned = eval_model(yte, proba_tuned_te, thr_tuned)
metrics_tuned["model"] = "LightGBM (tuned)"
print(f"  AUC={metrics_tuned['roc_auc']}  PR-AUC={metrics_tuned['pr_auc']}  "
      f"Recall={metrics_tuned['recall']}  F1={metrics_tuned['f1']}")

# ── Tabel perbandingan ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PERBANDINGAN FINAL: LR vs LGBM Default vs LGBM Tuned")
print("=" * 60)

comparison = []
for m in [metrics_lr, metrics_def, metrics_tuned]:
    row = dict(m)
    # Top-k
    if "LR" in m["model"]:
        proba_ref = proba_lr_te
    elif "default" in m["model"]:
        proba_ref = proba_def_te
    else:
        proba_ref = proba_tuned_te
    for pct in [5, 10, 20, 30]:
        row[f"top{pct}pct"] = topk_recall(yte, proba_ref, pct)
    comparison.append(row)

comp_df = pd.DataFrame(comparison)
col_order = ["model","roc_auc","pr_auc","recall","precision","f1",
             "brier","top5pct","top10pct","top20pct","top30pct","threshold"]
col_order = [c for c in col_order if c in comp_df.columns]
print(comp_df[col_order].to_string(index=False))

comp_df[col_order].to_csv(OUT_T / "10_tuning_comparison.csv", index=False)

# Best params table
pd.DataFrame([{
    "best_val_auc": round(best_val_auc, 4),
    **best_params
}]).to_csv(OUT_T / "10_best_params.csv", index=False)

# ── Interpretasi ───────────────────────────────────────────────────────────────
print("\n=== INTERPRETASI ===")
gain_tuned = metrics_tuned["roc_auc"] - metrics_def["roc_auc"]
gap_vs_lr  = metrics_tuned["roc_auc"] - metrics_lr["roc_auc"]
print(f"  LightGBM tuned vs default : Δ AUC = {gain_tuned:+.4f}")
print(f"  LightGBM tuned vs LR      : Δ AUC = {gap_vs_lr:+.4f}")

if gap_vs_lr >= 0.005:
    print(f"\n  → LightGBM tuned melampaui LR secara substantif (+{gap_vs_lr:.4f}).")
    print(f"    LightGBM tuned menjadi primary model yang direkomendasikan.")
elif gap_vs_lr >= 0:
    print(f"\n  → LightGBM tuned sedikit lebih tinggi dari LR (+{gap_vs_lr:.4f}),")
    print(f"    tapi perbedaannya tidak substantif.")
    print(f"    LR tetap direkomendasikan sebagai primary model karena:")
    print(f"    (1) performa setara, (2) jauh lebih interpretatif.")
else:
    print(f"\n  → LightGBM tuned masih di bawah LR ({gap_vs_lr:.4f}).")
    print(f"    Ini konfirmasi kuat bahwa hubungan fitur bersifat linear.")
    print(f"    LR adalah primary model yang tepat.")

print(f"\nSaved: {OUT_T / '10_tuning_comparison.csv'}")
print("LGBM TUNING SELESAI.")
