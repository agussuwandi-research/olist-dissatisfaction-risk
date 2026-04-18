"""
07_reporting_assets.py
Konsolidasi semua output menjadi tabel laporan siap-tempel.
Cek konsistensi angka antar tabel.
Output: outputs/tables/07_*.csv, outputs/tables/07_report_summary.txt
"""

import pandas as pd
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent
OUT_T = ROOT / "outputs" / "tables"
OUT_T.mkdir(parents=True, exist_ok=True)

lines = []
def log(s=""):
    print(s)
    lines.append(s)

log("=" * 60)
log("REPORT ASSETS — Konsolidasi Output Final")
log("=" * 60)

# ── Tabel 1: Dataset Summary ───────────────────────────────────────────────────
log("\n=== TABEL 1: DATASET SUMMARY ===")
try:
    t1 = pd.read_csv(OUT_T / "02_dataset_summary.csv")
    log(t1.to_string(index=False))
except FileNotFoundError:
    log("  [BELUM ADA] Jalankan 02_preprocessing.py terlebih dahulu.")

# ── Tabel 2: Split Temporal ────────────────────────────────────────────────────
log("\n=== TABEL 2: SPLIT TEMPORAL ===")
try:
    t2 = pd.read_csv(OUT_T / "02_split_summary.csv")
    order = {"train":0,"validation":1,"test":2}
    t2["_ord"] = t2["split"].map(order)
    t2 = t2.sort_values("_ord").drop(columns=["_ord"])
    log(t2.to_string(index=False))
    t2.to_csv(OUT_T / "07_table2_split_summary.csv", index=False)
except FileNotFoundError:
    log("  [BELUM ADA] Jalankan 02_preprocessing.py terlebih dahulu.")

# ── Tabel 3: Feature Contract (summary) ───────────────────────────────────────
log("\n=== TABEL 3: FEATURE CONTRACT (summary) ===")
try:
    t3 = pd.read_csv(OUT_T / "03_feature_contract.csv")
    log(f"  Total fitur terdokumentasi: {len(t3)}")
    log(f"  Fitur yang masuk model: {(t3['in_model'].str.startswith('ya')).sum()}")
    log(f"  Explanatory-only (TIDAK masuk model): {(t3['in_model'].str.startswith('TIDAK')).sum()}")
    log("\n  In-model features:")
    log(t3[t3["in_model"].str.startswith("ya")][["feature","timing","unit"]].to_string(index=False))
    log("\n  Explanatory-only features:")
    log(t3[t3["in_model"].str.startswith("TIDAK")][["feature","timing","note"]].to_string(index=False))
    t3.to_csv(OUT_T / "07_table3_feature_contract.csv", index=False)
except FileNotFoundError:
    log("  [BELUM ADA] Jalankan 03_feature_engineering.py terlebih dahulu.")

# ── Tabel 4: Hasil Klasifikasi ─────────────────────────────────────────────────
log("\n=== TABEL 4: HASIL KLASIFIKASI ===")
try:
    t4 = pd.read_csv(OUT_T / "04_classification_metrics.csv")
    display_cols = ["scenario","model","roc_auc","pr_auc","recall","precision","f1","brier_score"]
    display_cols = [c for c in display_cols if c in t4.columns]
    log(t4[display_cols].to_string(index=False))
    t4[display_cols].to_csv(OUT_T / "07_table4_classification_metrics.csv", index=False)
except FileNotFoundError:
    log("  [BELUM ADA] Jalankan 04_modeling_classification.py terlebih dahulu.")

# ── Tabel 5: Top-K Capture ─────────────────────────────────────────────────────
log("\n=== TABEL 5: TOP-K CAPTURE (Intervention Simulation) ===")
try:
    t5 = pd.read_csv(OUT_T / "04_topk_capture.csv")
    log(t5.to_string(index=False))
    t5.to_csv(OUT_T / "07_table5_topk_capture.csv", index=False)
except FileNotFoundError:
    log("  [BELUM ADA] Jalankan 04_modeling_classification.py terlebih dahulu.")

# ── Tabel 6: Cluster Profile ───────────────────────────────────────────────────
log("\n=== TABEL 6: CLUSTER PROFILE ===")
try:
    t6 = pd.read_csv(OUT_T / "06_cluster_profile.csv")
    log(t6.to_string(index=False))
    t6.to_csv(OUT_T / "07_table6_cluster_profile.csv", index=False)
except FileNotFoundError:
    log("  [BELUM ADA] Jalankan 06_clustering.py terlebih dahulu.")

# ── Consistency Check ──────────────────────────────────────────────────────────
log("\n=== CONSISTENCY CHECK ===")
checks_passed = 0; checks_failed = 0

def check(cond, msg_ok, msg_fail):
    global checks_passed, checks_failed
    if cond:
        log(f"  ✓ {msg_ok}"); checks_passed += 1
    else:
        log(f"  ✗ {msg_fail}"); checks_failed += 1

try:
    t2 = pd.read_csv(OUT_T / "02_split_summary.csv")
    check(len(t2) == 3, "Split: 3 partisi ada", "Split: tidak ditemukan 3 partisi")
    check(t2["n_orders"].sum() > 50000, "Split: total orders > 50K", "Split: total orders terlalu kecil")
except: log("  ? Split summary tidak ditemukan")

try:
    t4 = pd.read_csv(OUT_T / "04_classification_metrics.csv")
    n_scenarios = t4["scenario"].nunique()
    check(n_scenarios >= 3, f"Scenarios: {n_scenarios} skenario ada", f"Scenarios: hanya {n_scenarios} skenario")
    check(all(t4["roc_auc"] >= 0.5), "AUC: semua >= 0.5", "AUC: ada yang < 0.5 (model lebih buruk dari random?)")
    check("LightGBM" in t4["model"].values, "LightGBM: ada di hasil", "LightGBM: tidak ditemukan di hasil")
    check("LR" in t4["model"].values, "LR: ada di hasil", "LR: tidak ditemukan di hasil")
except: log("  ? Classification metrics tidak ditemukan")

try:
    t3 = pd.read_csv(OUT_T / "03_feature_contract.csv")
    leaked = t3[(t3["in_model"].str.startswith("TIDAK")) & (t3["timing"] == "post-outcome")]
    check(len(leaked) >= 5, f"Feature contract: {len(leaked)} post-outcome vars terdokumentasi",
          "Feature contract: post-outcome vars belum lengkap")
    all_leaked_excluded = all(t3[t3["in_model"].str.startswith("TIDAK")]["timing"] == "post-outcome")
    check(all_leaked_excluded, "Leakage: semua TIDAK-masuk-model adalah post-outcome",
          "Leakage: ada variabel yang ditandai tidak masuk model tapi bukan post-outcome — perlu cek")
except: log("  ? Feature contract tidak ditemukan")

log(f"\nHasil check: {checks_passed} passed, {checks_failed} failed")

# ── Output list ────────────────────────────────────────────────────────────────
log("\n=== OUTPUT FILES ===")
log(f"  Tables:  {OUT_T}")
log(f"  Figures: {ROOT / 'outputs' / 'figures'}")

out_files = sorted(ROOT.glob("outputs/tables/*.csv")) + sorted(ROOT.glob("outputs/figures/*.png"))
for f in out_files:
    log(f"  {f.relative_to(ROOT)}")

# Save report
with open(OUT_T / "07_report_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

log("\nREPORTING ASSETS SELESAI.")
