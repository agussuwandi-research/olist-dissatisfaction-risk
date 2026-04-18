"""
08_delong_test.py
Uji signifikansi statistik perbedaan AUC antara skenario S1 (pre-fulfillment)
dan S2 (in-fulfillment) menggunakan metode DeLong (1988).

Comparisons utama (untuk RQ2):
  - LR   S1 vs LR   S2
  - RF   S1 vs RF   S2
  - LGBM S1 vs LGBM S2

Comparisons sekunder (untuk diskusi model):
  - LR S2 vs RF S2
  - LR S2 vs LGBM S2

Reference:
  DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988).
  Comparing the areas under two or more correlated receiver operating
  characteristic curves: a nonparametric approach.
  Biometrics, 44(3), 837-845.

Output: outputs/tables/08_delong_results.csv
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import roc_auc_score
import lightgbm as lgb

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
INTERIM = ROOT / "data_interim"
OUT_T   = ROOT / "outputs" / "tables"
OUT_T.mkdir(parents=True, exist_ok=True)

# ── DeLong implementation ──────────────────────────────────────────────────────
def _structural_components(y_true, scores, chunk=2000):
    """
    Hitung V10 dan V01 (structural components) secara chunked
    agar tidak OOM untuk dataset besar.
    
    V10[i] = P(score positif ke-i > skor negatif acak)
    V01[j] = P(skor positif acak > skor negatif ke-j)
    """
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    n1, n0 = len(pos), len(neg)

    V10 = np.empty(n1)
    for i in range(0, n1, chunk):
        b = pos[i:i+chunk, None]           # (batch, 1)
        gt = b > neg[None, :]              # (batch, n0)
        eq = b == neg[None, :]
        V10[i:i+chunk] = gt.mean(1) + 0.5 * eq.mean(1)

    V01 = np.empty(n0)
    for j in range(0, n0, chunk):
        b = neg[j:j+chunk, None]           # (batch, 1)
        lt = pos[None, :] > b              # (batch, n1)
        eq = pos[None, :] == b
        V01[j:j+chunk] = lt.mean(1) + 0.5 * eq.mean(1)

    return V10, V01


def delong_test(y_true, proba_a, proba_b):
    """
    DeLong test untuk membandingkan dua AUC yang berkorelasi
    (keduanya diukur pada test set yang sama).

    H0: AUC_a == AUC_b
    Returns: auc_a, auc_b, z_stat, p_value, ci95_lo, ci95_hi
    """
    y = np.asarray(y_true, dtype=int)
    a = np.asarray(proba_a, dtype=float)
    b = np.asarray(proba_b, dtype=float)

    n1 = y.sum()
    n0 = len(y) - n1

    V10_a, V01_a = _structural_components(y, a)
    V10_b, V01_b = _structural_components(y, b)

    auc_a = float(V10_a.mean())
    auc_b = float(V10_b.mean())

    # Variance-covariance matrix
    var_a  = np.var(V10_a, ddof=1) / n1 + np.var(V01_a, ddof=1) / n0
    var_b  = np.var(V10_b, ddof=1) / n1 + np.var(V01_b, ddof=1) / n0
    cov_ab = (np.cov(V10_a, V10_b, ddof=1)[0, 1] / n1 +
              np.cov(V01_a, V01_b, ddof=1)[0, 1] / n0)

    var_diff = var_a + var_b - 2 * cov_ab
    if var_diff <= 0:
        return auc_a, auc_b, 0.0, 1.0, auc_a - auc_b, auc_a - auc_b

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))

    # 95% CI pada selisih AUC
    se_diff = np.sqrt(var_diff)
    ci_lo   = (auc_a - auc_b) - 1.96 * se_diff
    ci_hi   = (auc_a - auc_b) + 1.96 * se_diff

    return auc_a, auc_b, float(z), p, float(ci_lo), float(ci_hi)


# ── Preprocessing helpers (disalin ringkas dari script 04) ─────────────────────
META  = {"order_id", "split", "target_broad", "target_severe"}
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
            pd.concat([tr[[col]], va[[col]], te[[col]]]),
            prefix=col
        )
        lens = [len(tr), len(va), len(te)]
        d_tr = combined.iloc[:lens[0]].reset_index(drop=True)
        d_va = combined.iloc[lens[0]:lens[0]+lens[1]].reset_index(drop=True)
        d_te = combined.iloc[lens[0]+lens[1]:].reset_index(drop=True)
        for d, dum in [(tr, d_tr), (va, d_va), (te, d_te)]:
            for c in d.columns:
                if c == col:
                    d.drop(columns=[col], inplace=True)
                    break
            d.reset_index(drop=True, inplace=True)
            for c in dum.columns:
                d[c] = dum[c].values
    return tr, va, te

def prepare(df, target, feat_cols, num_cols):
    tr = df[df["split"] == "train"].reset_index(drop=True)
    va = df[df["split"] == "validation"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)

    Xtr = tr[feat_cols].copy()
    Xva = va[feat_cols].copy()
    Xte = te[feat_cols].copy()

    for col in num_cols:
        if col not in Xtr.columns:
            continue
        med = Xtr[col].median()
        Xtr[col] = Xtr[col].fillna(med)
        Xva[col] = Xva[col].fillna(med)
        Xte[col] = Xte[col].fillna(med)

    for col in CAT_COLS:
        if col not in Xtr.columns:
            continue
        Xtr[col] = Xtr[col].fillna("Unknown")
        Xva[col] = Xva[col].fillna("Unknown")
        Xte[col] = Xte[col].fillna("Unknown")

    Xtr, Xva, Xte = encode_cat(Xtr, Xva, Xte, CAT_COLS)
    Xtr = Xtr.fillna(0).astype(float)
    Xva = Xva.fillna(0).astype(float)
    Xte = Xte.fillna(0).astype(float)

    return (Xtr, Xva, Xte,
            tr[target].values, va[target].values, te[target].values)


def get_test_probas(df, target, feat_cols, num_cols):
    """Latih LR, RF, LGBM; kembalikan probabilitas di test set."""
    Xtr, Xva, Xte, ytr, yva, yte = prepare(df, target, feat_cols, num_cols)

    num_in_feat = [c for c in num_cols if c in Xtr.columns]

    # Scaling untuk LR
    scaler = StandardScaler()
    Xtr_lr = Xtr.copy(); Xva_lr = Xva.copy(); Xte_lr = Xte.copy()
    if num_in_feat:
        Xtr_lr[num_in_feat] = scaler.fit_transform(Xtr[num_in_feat])
        Xva_lr[num_in_feat] = scaler.transform(Xva[num_in_feat])
        Xte_lr[num_in_feat] = scaler.transform(Xte[num_in_feat])

    models = {
        "LR": LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced",
                                  random_state=42, solver="lbfgs"),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=8,
                                      class_weight="balanced",
                                      random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                         num_leaves=31, max_depth=6,
                                         class_weight="balanced",
                                         random_state=42, n_jobs=-1, verbose=-1),
    }

    probas = {}
    for mname, model in models.items():
        print(f"    Training {mname}...", end=" ", flush=True)
        Xtr_m = Xtr_lr if mname == "LR" else Xtr
        Xte_m = Xte_lr if mname == "LR" else Xte
        model.fit(Xtr_m, ytr)
        probas[mname] = model.predict_proba(Xte_m)[:, 1]
        auc = roc_auc_score(yte, probas[mname])
        print(f"AUC={auc:.4f}")

    return probas, yte


# ── Main ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print("DELONG TEST — S1 vs S2 Significance Analysis")
print("=" * 60)

# Load feature tables
print("\nLoading feature tables...")
pre = pd.read_csv(INTERIM / "order_level_features_pre.csv", low_memory=False)
inf = pd.read_csv(INTERIM / "order_level_features_in.csv",  low_memory=False)

PRE_FEATS = get_feat_cols(pre)
IN_FEATS  = get_feat_cols(inf)
NUM_PRE   = [c for c in PRE_FEATS if c not in CAT_COLS]
NUM_IN    = [c for c in IN_FEATS  if c not in CAT_COLS]

print(f"  Pre-fulfillment features : {len(PRE_FEATS)}")
print(f"  In-fulfillment features  : {len(IN_FEATS)}")

# Train dan dapatkan prediksi test
print("\n--- S1: Pre-fulfillment (broad) ---")
probas_s1, y_te = get_test_probas(pre, "target_broad", PRE_FEATS, NUM_PRE)

print("\n--- S2: In-fulfillment (broad) ---")
probas_s2, _    = get_test_probas(inf, "target_broad", IN_FEATS,  NUM_IN)

print(f"\nTest set: {len(y_te):,} orders  |  pos rate: {y_te.mean():.3f}")

# ── Jalankan DeLong tests ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RUNNING DELONG TESTS")
print("=" * 60)

comparisons = [
    # Nama, proba_A, proba_B, label_A, label_B
    ("RQ2: LR   S1 vs S2",   probas_s1["LR"],       probas_s2["LR"],       "LR_S1",    "LR_S2"),
    ("RQ2: RF   S1 vs S2",   probas_s1["RF"],       probas_s2["RF"],       "RF_S1",    "RF_S2"),
    ("RQ2: LGBM S1 vs S2",   probas_s1["LightGBM"], probas_s2["LightGBM"],"LGBM_S1",  "LGBM_S2"),
    ("Disc: LR vs RF (S2)",   probas_s2["LR"],       probas_s2["RF"],       "LR_S2",    "RF_S2"),
    ("Disc: LR vs LGBM (S2)", probas_s2["LR"],       probas_s2["LightGBM"],"LR_S2",    "LGBM_S2"),
]

rows = []
for label, pa, pb, na, nb in comparisons:
    print(f"\n  {label}...", end=" ", flush=True)
    auc_a, auc_b, z, p, ci_lo, ci_hi = delong_test(y_te, pa, pb)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"AUC_A={auc_a:.4f}  AUC_B={auc_b:.4f}  "
          f"Δ={auc_a-auc_b:+.4f}  z={z:.3f}  p={p:.4f}  {sig}")
    rows.append({
        "comparison":   label,
        "model_A":      na,
        "model_B":      nb,
        "auc_A":        round(auc_a, 4),
        "auc_B":        round(auc_b, 4),
        "delta_auc":    round(auc_a - auc_b, 4),
        "z_stat":       round(z, 4),
        "p_value":      round(p, 4),
        "ci95_lo":      round(ci_lo, 4),
        "ci95_hi":      round(ci_hi, 4),
        "significance": sig,
    })

# ── Ringkasan ──────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(rows)
results_df.to_csv(OUT_T / "08_delong_results.csv", index=False)

print("\n" + "=" * 60)
print("HASIL AKHIR")
print("=" * 60)
print(f"\nKeterangan: *** p<0.001 | ** p<0.01 | * p<0.05 | ns = tidak signifikan\n")

display_cols = ["comparison","auc_A","auc_B","delta_auc","p_value","significance"]
print(results_df[display_cols].to_string(index=False))

print(f"\n\nINTERPRETASI:")
rq2_rows = results_df[results_df["comparison"].str.startswith("RQ2")]
all_sig   = all(rq2_rows["p_value"] < 0.05)
any_sig   = any(rq2_rows["p_value"] < 0.05)

if all_sig:
    print("  → Gain S2 vs S1 signifikan secara statistik untuk SEMUA model.")
    print("    Klaim RQ2 dapat dipertahankan dengan kuat.")
elif any_sig:
    print("  → Gain S2 vs S1 signifikan untuk sebagian model.")
    print("    Klaim RQ2 perlu dikalibrasi — sebutkan model mana yang signifikan.")
else:
    print("  → Gain S2 vs S1 TIDAK signifikan secara statistik.")
    print("    Klaim RQ2 perlu direvisi: sinyal in-fulfillment memberikan")
    print("    peningkatan deskriptif tapi belum terbukti signifikan.")

disc_rows = results_df[results_df["comparison"].str.startswith("Disc")]
for _, r in disc_rows.iterrows():
    sig_str = "signifikan" if r["p_value"] < 0.05 else "TIDAK signifikan"
    print(f"\n  → {r['comparison']}: perbedaan {sig_str} "
          f"(Δ={r['delta_auc']:+.4f}, p={r['p_value']:.4f})")

print(f"\nSaved: {OUT_T / '08_delong_results.csv'}")
print("\nDELONG TEST SELESAI.")
