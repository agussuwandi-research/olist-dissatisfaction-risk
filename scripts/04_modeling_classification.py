"""
04_modeling_classification.py
Training semua skenario klasifikasi, evaluasi lengkap, SHAP, intervention simulation.
Output: outputs/tables/04_*.csv, outputs/figures/04_*.png, outputs/models/
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.dummy          import DummyClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.preprocessing  import StandardScaler, RobustScaler, label_binarize
from sklearn.calibration    import CalibratedClassifierCV, calibration_curve
from sklearn.metrics        import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    brier_score_loss, confusion_matrix,
    roc_curve, precision_recall_curve,
)
import lightgbm as lgb
import shap

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
INTERIM    = ROOT / "data_interim"
OUT_T      = ROOT / "outputs" / "tables"
OUT_F      = ROOT / "outputs" / "figures"
OUT_M      = ROOT / "outputs" / "models"
for p in [OUT_T, OUT_F, OUT_M]: p.mkdir(parents=True, exist_ok=True)

# ── Leakage blacklist ──────────────────────────────────────────────────────────
LEAKAGE_BLACKLIST = {
    "order_delivered_customer_date", "actual_delivery_days",
    "is_late", "estimation_error", "seller_phase_days", "logistics_phase_days",
    "review_score", "review_comment_message", "review_comment_title",
    "review_creation_date", "review_answer_timestamp",
    "order_id", "customer_id", "customer_unique_id", "seller_id", "seller_rep_id",
    "split", "target_broad", "target_severe",
    # timestamp kolom
    "order_purchase_timestamp", "order_approved_at",
    "order_delivered_carrier_date", "order_estimated_delivery_date",
}

def assert_no_leakage(df, name="feature_matrix"):
    violations = [c for c in LEAKAGE_BLACKLIST if c in df.columns]
    assert len(violations) == 0, f"LEAKAGE di {name}: {violations}"
    print(f"  Leakage check [{name}]: PASSED ({len(df.columns)} kolom OK)")

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading feature tables...")
pre = pd.read_csv(INTERIM / "order_level_features_pre.csv", low_memory=False)
inf = pd.read_csv(INTERIM / "order_level_features_in.csv",  low_memory=False)
print(f"  pre-fulfillment : {len(pre):,} rows, {len(pre.columns)} cols")
print(f"  in-fulfillment  : {len(inf):,} rows, {len(inf.columns)} cols")

# ── Feature sets ───────────────────────────────────────────────────────────────
META_COLS = {"order_id", "split", "target_broad", "target_severe"}

def get_feature_cols(df):
    return [c for c in df.columns if c not in META_COLS and c not in LEAKAGE_BLACKLIST]

PRE_FEATS = get_feature_cols(pre)
IN_FEATS  = get_feature_cols(inf)

# Categorical dan numeric splits
CAT_COLS = ["customer_state", "seller_state", "payment_type_main", "top_category"]
NUM_COLS_PRE = [c for c in PRE_FEATS if c not in CAT_COLS]
NUM_COLS_IN  = [c for c in IN_FEATS  if c not in CAT_COLS]

print(f"\n  Pre-fulfillment features ({len(PRE_FEATS)}): {PRE_FEATS}")
print(f"  In-fulfillment  features ({len(IN_FEATS)}): {IN_FEATS}")

# ── Preprocessing pipeline per split ─────────────────────────────────────────
def encode_categoricals(train_df, val_df, test_df, cat_cols, top_n=15):
    """One-hot encode categorical columns. Fit mapping on train only."""
    dfs = {"train": train_df.copy(), "val": val_df.copy(), "test": test_df.copy()}
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        top_vals = train_df[col].value_counts().head(top_n).index.tolist()
        for key in dfs:
            dfs[key][col] = dfs[key][col].where(dfs[key][col].isin(top_vals), "Other")
        dummies = pd.get_dummies(
            pd.concat([dfs["train"][[col]], dfs["val"][[col]], dfs["test"][[col]]]),
            prefix=col, drop_first=False
        )
        split_lens = [len(dfs["train"]), len(dfs["val"]), len(dfs["test"])]
        d_train = dummies.iloc[:split_lens[0]].reset_index(drop=True)
        d_val   = dummies.iloc[split_lens[0]:split_lens[0]+split_lens[1]].reset_index(drop=True)
        d_test  = dummies.iloc[split_lens[0]+split_lens[1]:].reset_index(drop=True)
        for key, d in zip(["train","val","test"], [d_train, d_val, d_test]):
            dfs[key] = dfs[key].drop(columns=[col]).reset_index(drop=True)
            dfs[key] = pd.concat([dfs[key], d], axis=1)
    return dfs["train"], dfs["val"], dfs["test"]

def prepare_splits(df, target_col, feat_cols, cat_cols, num_cols):
    """Buat X_train, X_val, X_test yang siap dipakai."""
    train = df[df["split"] == "train"].reset_index(drop=True)
    val   = df[df["split"] == "validation"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)

    # Ambil feature kolom saja
    tr_x = train[feat_cols].copy()
    va_x = val[feat_cols].copy()
    te_x = test[feat_cols].copy()

    # Impute numerik: median dari train
    for col in num_cols:
        if col not in tr_x.columns: continue
        med = tr_x[col].median()
        tr_x[col] = tr_x[col].fillna(med)
        va_x[col] = va_x[col].fillna(med) if col in va_x.columns else va_x[col]
        te_x[col] = te_x[col].fillna(med) if col in te_x.columns else te_x[col]

    # Impute kategorikal: "Unknown"
    for col in cat_cols:
        if col not in tr_x.columns: continue
        tr_x[col] = tr_x[col].fillna("Unknown")
        va_x[col] = va_x[col].fillna("Unknown") if col in va_x.columns else va_x[col]
        te_x[col] = te_x[col].fillna("Unknown") if col in te_x.columns else te_x[col]

    # Encode
    tr_x, va_x, te_x = encode_categoricals(tr_x, va_x, te_x, cat_cols)

    y_tr = train[target_col].values
    y_va = val[target_col].values
    y_te = test[target_col].values

    return tr_x, va_x, te_x, y_tr, y_va, y_te

# ── Threshold selection dari validation ───────────────────────────────────────
def select_threshold_from_val(y_val, proba_val, target_recall=0.50):
    """Pilih threshold tertinggi yang memenuhi target recall dari validation set."""
    precisions, recalls, thresholds = precision_recall_curve(y_val, proba_val)
    # threshold index: len(thresholds) = len(precisions)-1
    valid = [(t, r, p) for t, r, p in zip(thresholds, recalls[:-1], precisions[:-1])
             if r >= target_recall]
    if valid:
        return max(valid, key=lambda x: x[0])[0]  # threshold tertinggi yang masih OK
    return 0.5  # fallback

# ── Metrik ─────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, proba, threshold, scenario, model):
    """scenario dan model disimpan terpisah agar Tabel 4 bersih."""
    y_pred = (proba >= threshold).astype(int)
    return {
        "scenario":    scenario,
        "model":       model,
        "roc_auc":     round(roc_auc_score(y_true, proba), 4),
        "pr_auc":      round(average_precision_score(y_true, proba), 4),
        "brier_score": round(brier_score_loss(y_true, proba), 4),
        "precision":   round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":      round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":          round(f1_score(y_true, y_pred, zero_division=0), 4),
        "threshold":   round(threshold, 4),
    }

def compute_topk(y_true, proba, scenario, model):
    """Top-k dissatisfaction capture rate."""
    n = len(y_true)
    total_pos = y_true.sum()
    row = {"scenario": scenario, "model": model}
    for pct in [5, 10, 20, 30]:
        k = int(n * pct / 100)
        top_idx = np.argsort(proba)[::-1][:k]
        caught = y_true[top_idx].sum()
        row[f"top_{pct}pct_recall"] = round(caught / total_pos, 4) if total_pos > 0 else 0
    return row

# ── Scenarios definition ───────────────────────────────────────────────────────
SCENARIOS = [
    {"name": "S1_pre_broad",  "df": pre, "feat": PRE_FEATS, "target": "target_broad",
     "num": NUM_COLS_PRE, "cat": CAT_COLS},
    {"name": "S2_in_broad",   "df": inf, "feat": IN_FEATS,  "target": "target_broad",
     "num": NUM_COLS_IN,  "cat": CAT_COLS},
    {"name": "S3_in_severe",  "df": inf, "feat": IN_FEATS,  "target": "target_severe",
     "num": NUM_COLS_IN,  "cat": CAT_COLS},
    # S4 opsional: {"name":"S4_pre_severe","df":pre,"feat":PRE_FEATS,"target":"target_severe",...}
]

MODELS = {
    "Dummy":    lambda: DummyClassifier(strategy="stratified", random_state=42),
    "LR":       lambda: LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                                            random_state=42, solver="lbfgs"),
    "RF":       lambda: RandomForestClassifier(n_estimators=300, max_depth=8,
                                                class_weight="balanced",
                                                random_state=42, n_jobs=-1),
    "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                             num_leaves=31, max_depth=6,
                                             class_weight="balanced",
                                             random_state=42, n_jobs=-1,
                                             verbose=-1),
}

# ── Main training loop ─────────────────────────────────────────────────────────
all_metrics = []
all_topk    = []
thresholds_log = {}
roc_fig, roc_axes = plt.subplots(1, len(SCENARIOS), figsize=(6*len(SCENARIOS), 5))
pr_fig,  pr_axes  = plt.subplots(1, len(SCENARIOS), figsize=(6*len(SCENARIOS), 5))

for si, sc in enumerate(SCENARIOS):
    sname   = sc["name"]
    target  = sc["target"]
    feat_df = sc["df"].dropna(subset=[target]).reset_index(drop=True)
    feat_cols = [c for c in sc["feat"] if c in feat_df.columns]

    print(f"\n{'='*60}")
    print(f"SCENARIO: {sname}  target={target}  feats={len(feat_cols)}")
    print(f"{'='*60}")

    # Prepare splits
    tr_x, va_x, te_x, y_tr, y_va, y_te = prepare_splits(
        feat_df, target, feat_cols, sc["cat"], sc["num"]
    )

    # Leakage audit
    assert_no_leakage(tr_x, f"{sname} X_train")
    assert_no_leakage(va_x, f"{sname} X_val")
    assert_no_leakage(te_x, f"{sname} X_test")

    print(f"  Train: {len(y_tr):,}  pos={y_tr.mean():.3f}")
    print(f"  Val  : {len(y_va):,}  pos={y_va.mean():.3f}")
    print(f"  Test : {len(y_te):,}  pos={y_te.mean():.3f}")

    # LR scaling (fit on train only)
    scaler = StandardScaler()
    num_in_feat = [c for c in sc["num"] if c in tr_x.columns]
    tr_x_lr = tr_x.copy(); va_x_lr = va_x.copy(); te_x_lr = te_x.copy()
    if num_in_feat:
        tr_x_lr[num_in_feat] = scaler.fit_transform(tr_x[num_in_feat])
        va_x_lr[num_in_feat] = scaler.transform(va_x[num_in_feat])
        te_x_lr[num_in_feat] = scaler.transform(te_x[num_in_feat])

    roc_ax = roc_axes[si] if len(SCENARIOS) > 1 else roc_axes
    pr_ax  = pr_axes[si]  if len(SCENARIOS) > 1 else pr_axes
    roc_ax.set_title(sname); roc_ax.set_xlabel("FPR"); roc_ax.set_ylabel("TPR")
    pr_ax.set_title(sname);  pr_ax.set_xlabel("Recall"); pr_ax.set_ylabel("Precision")

    thresholds_log[sname] = {}

    for mname, model_fn in MODELS.items():
        print(f"\n  Training {mname}...")
        model = model_fn()

        x_tr = tr_x_lr if mname == "LR" else tr_x
        x_va = va_x_lr if mname == "LR" else va_x
        x_te = te_x_lr if mname == "LR" else te_x

        model.fit(x_tr, y_tr)

        proba_va = model.predict_proba(x_va)[:, 1]
        proba_te = model.predict_proba(x_te)[:, 1]

        # Threshold dari validation
        thr = select_threshold_from_val(y_va, proba_va, target_recall=0.50)
        thresholds_log[sname][mname] = float(thr)
        print(f"    Threshold (val, recall≥0.50): {thr:.4f}")

        # Metrik di test
        metrics = compute_metrics(y_te, proba_te, thr, sname, mname)
        all_metrics.append(metrics)

        # Top-k
        topk = compute_topk(y_te, proba_te, sname, mname)
        all_topk.append(topk)

        print(f"    Test AUC={metrics['roc_auc']}  PR-AUC={metrics['pr_auc']}  "
              f"Recall={metrics['recall']}  F1={metrics['f1']}")

        # ROC & PR curves
        if mname != "Dummy":
            fpr, tpr, _ = roc_curve(y_te, proba_te)
            roc_ax.plot(fpr, tpr, label=f"{mname} (AUC={metrics['roc_auc']})")
            prec_c, rec_c, _ = precision_recall_curve(y_te, proba_te)
            pr_ax.plot(rec_c, prec_c, label=f"{mname} (AP={metrics['pr_auc']})")

        # SHAP untuk LightGBM (hanya S2 sebagai primary model)
        if mname == "LightGBM" and sname == "S2_in_broad":
            print(f"    Computing SHAP for {sname}/{mname}...")
            explainer = shap.TreeExplainer(model)
            shap_sample = x_te.sample(min(2000, len(x_te)), random_state=42)
            shap_vals = explainer.shap_values(shap_sample)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # kelas positif
            shap.summary_plot(shap_vals, shap_sample, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(OUT_F / f"04_shap_{sname}.png", dpi=120, bbox_inches="tight")
            plt.close()
            print(f"    SHAP saved: 04_shap_{sname}.png")

        # LR koefisien (S1)
        if mname == "LR" and sname == "S1_pre_broad":
            coef_df = pd.DataFrame({
                "feature": x_tr.columns.tolist(),
                "coef":    model.coef_[0],
            }).sort_values("coef", key=abs, ascending=False).head(20)
            coef_df.to_csv(OUT_T / f"04_lr_coef_{sname}.csv", index=False)

            fig, ax = plt.subplots(figsize=(8, 6))
            coef_df_plot = coef_df.sort_values("coef")
            ax.barh(coef_df_plot["feature"], coef_df_plot["coef"],
                    color=["steelblue" if v > 0 else "tomato" for v in coef_df_plot["coef"]])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"LR Coefficients — {sname}"); ax.set_xlabel("Coefficient")
            plt.tight_layout()
            plt.savefig(OUT_F / f"04_lr_coef_{sname}.png", dpi=120, bbox_inches="tight")
            plt.close()

    # Finalize ROC / PR plots
    roc_ax.plot([0,1],[0,1],"k--",linewidth=0.8)
    roc_ax.legend(fontsize=7); roc_ax.set_xlim(0,1); roc_ax.set_ylim(0,1)
    pr_ax.legend(fontsize=7)

# ── Calibration (LightGBM, semua skenario) ────────────────────────────────────
print("\n=== CALIBRATION CHECK ===")
cal_fig, cal_axes = plt.subplots(1, len(SCENARIOS), figsize=(6*len(SCENARIOS), 5))
brier_rows = []
for si, sc in enumerate(SCENARIOS):
    sname  = sc["name"]
    target = sc["target"]
    feat_df = sc["df"].dropna(subset=[target]).reset_index(drop=True)
    feat_cols = [c for c in sc["feat"] if c in feat_df.columns]
    tr_x, va_x, te_x, y_tr, y_va, y_te = prepare_splits(
        feat_df, target, feat_cols, sc["cat"], sc["num"]
    )
    lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                max_depth=6, class_weight="balanced",
                                random_state=42, n_jobs=-1, verbose=-1)
    lgbm.fit(tr_x, y_tr)
    proba_te = lgbm.predict_proba(te_x)[:, 1]

    frac_pos, mean_pred = calibration_curve(y_te, proba_te, n_bins=10)
    ax = cal_axes[si] if len(SCENARIOS) > 1 else cal_axes
    ax.plot(mean_pred, frac_pos, "s-", label="LightGBM")
    ax.plot([0,1],[0,1],"k--", linewidth=0.8, label="Perfect")
    ax.set_title(f"Calibration — {sname}")
    ax.set_xlabel("Mean predicted prob"); ax.set_ylabel("Fraction of positives")
    ax.legend(fontsize=7)

    bs = brier_score_loss(y_te, proba_te)
    brier_rows.append({"scenario": sname, "brier_score": round(bs, 4)})
    print(f"  {sname}: Brier Score = {bs:.4f}")

cal_fig.suptitle("Calibration Curves — LightGBM")
plt.tight_layout()
cal_fig.savefig(OUT_F / "04_calibration_curves.png", dpi=120, bbox_inches="tight")
plt.close()

# ── Save ROC / PR figures ──────────────────────────────────────────────────────
roc_fig.suptitle("ROC Curves per Scenario")
roc_fig.tight_layout()
roc_fig.savefig(OUT_F / "04_roc_curves.png", dpi=120, bbox_inches="tight")
plt.close()

pr_fig.suptitle("Precision-Recall Curves per Scenario")
pr_fig.tight_layout()
pr_fig.savefig(OUT_F / "04_pr_curves.png", dpi=120, bbox_inches="tight")
plt.close()

# ── Cumulative Gain (LightGBM S2 primary) ─────────────────────────────────────
print("\n=== CUMULATIVE GAIN CHART ===")
sc = SCENARIOS[1]  # S2
feat_df = sc["df"].dropna(subset=[sc["target"]]).reset_index(drop=True)
feat_cols = [c for c in sc["feat"] if c in feat_df.columns]
tr_x, va_x, te_x, y_tr, y_va, y_te = prepare_splits(
    feat_df, sc["target"], feat_cols, sc["cat"], sc["num"]
)
lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                            max_depth=6, class_weight="balanced",
                            random_state=42, n_jobs=-1, verbose=-1)
lgbm.fit(tr_x, y_tr)
proba_te = lgbm.predict_proba(te_x)[:, 1]
sorted_idx = np.argsort(proba_te)[::-1]
y_sorted   = y_te[sorted_idx]
cum_pos    = np.cumsum(y_sorted) / y_te.sum()
x_pct      = np.arange(1, len(y_te)+1) / len(y_te)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_pct, cum_pos, label="LightGBM S2", color="steelblue")
ax.plot([0,1],[0,1],"k--", linewidth=0.8, label="Random baseline")
ax.set_xlabel("Fraction of orders flagged")
ax.set_ylabel("Fraction of dissatisfied orders captured")
ax.set_title("Cumulative Gain — S2 In-fulfillment Broad")
ax.legend(); ax.set_xlim(0,1); ax.set_ylim(0,1)
plt.tight_layout()
fig.savefig(OUT_F / "04_cumulative_gain_S2.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: 04_cumulative_gain_S2.png")

# ── Save tables ────────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)
col_order  = ["scenario","model","roc_auc","pr_auc","recall","precision","f1","brier_score","threshold"]
col_order  = [c for c in col_order if c in metrics_df.columns]
metrics_df[col_order].to_csv(OUT_T / "04_classification_metrics.csv", index=False)

topk_df = pd.DataFrame(all_topk)
topk_df.to_csv(OUT_T / "04_topk_capture.csv", index=False)

pd.DataFrame(brier_rows).to_csv(OUT_T / "04_brier_scores.csv", index=False)

with open(OUT_T / "04_thresholds.json", "w") as f:
    json.dump(thresholds_log, f, indent=2)

print("\n=== HASIL UTAMA ===")
print(metrics_df[col_order].to_string(index=False))

print("\n=== TOP-K CAPTURE ===")
print(topk_df.to_string(index=False))

print(f"\nSemua output di:")
print(f"  {OUT_T}")
print(f"  {OUT_F}")
print("\nMODELING SELESAI.")
