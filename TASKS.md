# Task Tracker — Proyek Olist

Legenda: [ ] belum | [~] sedang | [x] selesai

> Status disinkronkan dengan kondisi folder aktual (1 Juli 2026).
> Daftar temuan audit & perbaikan: lihat `AUDIT_FINDINGS.md`.

---

## A. Setup Proyek
- [x] Buat struktur folder proyek
- [x] Pindahkan semua raw CSV ke `data_raw/`
- [x] Simpan `olist_blueprint_final.md` di root project
- [x] Buat conda env `olist`
- [x] Install package utama (termasuk shap, lightgbm, optuna, streamlit)
- [x] Register Jupyter kernel `Python (olist)`
- [x] Export `environment.yml` → lihat juga `environment_clean.yml` (versi portabel)
- [x] Buat `.gitignore`

---

## B. Dokumentasi Dasar
- [x] Buat `README.md`
- [x] Buat `DECISIONS.md`
- [x] Buat `TASKS.md`
- [x] Isi `feature_contract` (otomatis oleh `03_feature_engineering.py`) → 25 fitur
- [x] Outline & draf RTM1 (`report/rtm1/`)
- [~] RTM2 / RTM3 (materi tercakup di paper draft)
- [ ] RTM4 — **jatuh tempo 6 Juli 2026** (presentasi + knowledge-based system model terbaik)

---

## C. Data Audit
- [x] Buat `scripts/01_data_audit.py`
- [x] Jalankan & simpan output `01_*` (semua ada di `outputs/tables/`)

## D. Preprocessing Dasar
- [x] Buat `scripts/02_preprocessing.py`
- [x] Jalankan & simpan `reviews_dedup.csv`, `orders_filtered.csv`, `02_*`
- [x] Verifikasi: order_id unik, target tidak null, 3 split ada

## E. Feature Engineering
- [x] Buat `scripts/03_feature_engineering.py`
- [x] Jalankan & simpan feature tables + `03_feature_contract.csv`, `03_missing_value_report.csv`

## F. Quality Gate (di script 04)
- [x] `assert_no_leakage()` passed untuk X_train, X_val, X_test
- [x] Semua blacklist kolom tidak ada di feature matrix

## G. Modeling Klasifikasi
- [x] Buat `scripts/04_modeling_classification.py`
- [x] Jalankan: S1, S2, S3 × LR + RF + LightGBM + Dummy
- [x] Threshold dipilih dari validation set
- [x] Output: `04_*` (metrics, topk, thresholds, brier, ROC/PR/calibration/SHAP/gain/coef)

## H. Explanatory Analysis
- [x] Buat `scripts/05_explanatory_analysis.py`
- [x] Jalankan: estimation error + asymmetry, is_late vs estimation_error, delivery chain, kategori, state, same-state
- [x] Output: semua `05_*`

## I. Clustering
- [x] Buat `scripts/06_clustering.py`
- [x] Jalankan: redundancy → RobustScaler → elbow+silhouette → stability → K-Means → profiling
- [x] Output: `06_*`, `cluster_assignments.csv`
- [ ] **Tindak lanjut audit:** pemilihan K masih murni max-silhouette → K=2 (trivial). Pertimbangkan K=3–4 + interpretability (lihat B3 di AUDIT_FINDINGS.md)

## J. Reporting Assets
- [x] Buat `scripts/07_reporting_assets.py`
- [x] Jalankan → `07_table2..6_*`, `07_report_summary.txt`
- [ ] **Jalankan ulang `07`** setelah script 08–11 (summary saat ini belum memuat DeLong/tuning/error/seller-hist)

---

## K. Analisis Lanjutan (ekstensi)
- [x] `scripts/08_delong_test.py` — uji signifikansi AUC S1 vs S2 (RQ2) → `08_delong_results.csv`
- [x] `scripts/09_error_analysis.py` — analisis FN/FP model S2 → `09_*`
- [x] `scripts/10_lgbm_tuning.py` — tuning Optuna LightGBM → `10_*`
- [x] `scripts/10b_lgbm_tuned_fair.py` — perbandingan fair (train-only) → `10b_fair_comparison.csv` **(hasil kanonik)**
- [x] `scripts/11_seller_hist_experiment.py` — eksperimen fitur histori seller → `11_*`

## L. Dashboard & Deployment
- [x] Buat `dashboard.py` (Streamlit, 9 tab)
- [x] Deploy ke Streamlit Community Cloud
- [x] Publikasi repo GitHub (`agussuwandi-research/olist-dissatisfaction-risk`)
- [ ] **Tindak lanjut audit:** keluarkan `data_raw`/`data_interim` dari repo publik; tab error-analysis baca `09_*.csv` (jangan hardcode)

---

## M. Paper (RTM & UAS)
- [x] Paper draft: Title, Author, Abstract, Keyword, Introduction, Related Works, Methodology
- [ ] **Result and Discussion** — belum ada (UAS, jatuh tempo 27 Juli 2026)
- [ ] **Conclusion** — belum ada
- [ ] Perbaiki klaim di paper: GPU-accelerated training (kode CPU), DummyClassifier (`stratified`, bukan `most_frequent`)
- [ ] Klarifikasi peran author vs pembimbing (Andi Sunyoto, Robert Marco)
- [ ] Cek Turnitin < 20%

---

## N. Ringkasan Kondisi
- Seluruh pipeline (01–11 + 10b) sudah dibuat & dijalankan; output lengkap di `outputs/`.
- Dashboard live & repo publik sudah ada.
- **Fokus berikutnya:** (1) rapikan sesuai `AUDIT_FINDINGS.md`; (2) RTM#4 (6 Juli) knowledge-based system model terbaik; (3) UAS (27 Juli) lengkapi Result/Discussion/Conclusion paper.

---

## Urutan Eksekusi Pipeline
```bash
cd scripts/
python 01_data_audit.py
python 02_preprocessing.py
python 03_feature_engineering.py
python 04_modeling_classification.py
python 05_explanatory_analysis.py
python 06_clustering.py
python 08_delong_test.py
python 09_error_analysis.py
python 10_lgbm_tuning.py
python 10b_lgbm_tuned_fair.py
python 11_seller_hist_experiment.py
python 07_reporting_assets.py   # jalankan paling akhir
```
