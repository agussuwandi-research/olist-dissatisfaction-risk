# Task Tracker — Proyek Olist

Legenda: [ ] belum | [~] sedang | [x] selesai

---

## A. Setup Proyek
- [x] Buat struktur folder proyek
- [x] Pindahkan semua raw CSV ke `data_raw/`
- [x] Simpan `olist_blueprint_final.md` di root project
- [x] Buat conda env `olist`
- [x] Install package utama (termasuk shap, lightgbm)
- [x] Register Jupyter kernel `Python (olist)`
- [x] Export `environment.yml` → lihat juga `environment_clean.yml` (versi portabel)
- [x] Buat `.gitignore`

---

## B. Dokumentasi Dasar
- [x] Buat `README.md`
- [x] Buat `DECISIONS.md`
- [x] Buat `TASKS.md`
- [ ] Isi `feature_contract` (dikerjakan otomatis oleh `03_feature_engineering.py`)
- [ ] Siapkan outline RTM1
- [ ] Siapkan outline RTM2
- [ ] Siapkan outline RTM3
- [ ] Siapkan outline RTM4

---

## C. Data Audit
- [x] Buat `scripts/01_data_audit.py`
- [ ] Jalankan dan simpan output: `01_table_summary.csv`, `01_null_report.csv`,
      `01_key_uniqueness.csv`, `01_join_coverage.csv`,
      `01_review_duplication_summary.csv`, `01_monthly_distribution.csv`,
      `01_order_status_distribution.csv`, `01_review_score_distribution.csv`

---

## D. Preprocessing Dasar
- [x] Buat `scripts/02_preprocessing.py`
- [ ] Jalankan dan simpan: `reviews_dedup.csv`, `orders_filtered.csv`,
      `02_dataset_summary.csv`, `02_split_summary.csv`
- [ ] Verifikasi: order_id unik, target tidak null, 3 split ada

---

## E. Feature Engineering
- [x] Buat `scripts/03_feature_engineering.py`
- [ ] Jalankan dan simpan: `order_level_features_pre.csv`, `order_level_features_in.csv`,
      `explanatory_variables.csv`, `clustering_features.csv`,
      `03_feature_contract.csv`, `03_missing_value_report.csv`

---

## F. Quality Gate (otomatis di script 04)
- [ ] `assert_no_leakage()` passed untuk X_train, X_val, X_test
- [ ] Semua blacklist kolom tidak ada di feature matrix

---

## G. Modeling Klasifikasi
- [x] Buat `scripts/04_modeling_classification.py`
- [ ] Jalankan: S1, S2, S3 × LR + RF + LightGBM + Dummy
- [ ] Threshold dipilih dari validation set
- [ ] Output: `04_classification_metrics.csv`, `04_topk_capture.csv`,
      `04_thresholds.json`, `04_brier_scores.csv`,
      `04_roc_curves.png`, `04_pr_curves.png`, `04_calibration_curves.png`,
      `04_cumulative_gain_S2.png`, `04_shap_S2_in_broad.png`, `04_lr_coef_S1_pre_broad.png`

---

## H. Explanatory Analysis
- [x] Buat `scripts/05_explanatory_analysis.py`
- [ ] Jalankan: estimation error bins + asymmetry, AUC is_late vs estimation_error,
      delivery chain quartiles, category rank, state-level, same-state vs cross-state
- [ ] Output: semua `05_*.csv` dan `05_*.png`

---

## I. Clustering
- [x] Buat `scripts/06_clustering.py`
- [ ] Jalankan: redundancy check → RobustScaler → elbow+silhouette →
      stability check (5 seeds) → K-Means final → profiling → visualization
- [ ] Output: `06_elbow_silhouette.csv`, `06_cluster_profile.csv`,
      `06_correlation_matrix.csv`, `cluster_assignments.csv`,
      `06_elbow_silhouette.png`, `06_cluster_diss_rate.png`, `06_cluster_radar.png`

---

## J. Reporting Assets
- [x] Buat `scripts/07_reporting_assets.py`
- [ ] Jalankan setelah semua script selesai
- [ ] Output: `07_table2..6_*.csv`, `07_report_summary.txt`
- [ ] Verifikasi consistency check: semua passed

---

## K. RTM dan Paper
### RTM1: [ ] Latar belakang, RQ, tinjauan paper, rencana metode
### RTM2: [ ] Dataset, preprocessing, split, feature engineering, leakage boundary
### RTM3: [ ] 9 model (3 algo × 3 skenario), evaluasi, clustering formal, profil cluster
### RTM4: [ ] Pembahasan, gap S1 vs S2, expectation violation, delivery chain,
              cluster interpretation, intervensi, keterbatasan

### Paper Draft
- [ ] Introduction | Data & Methods | Results | Discussion | Limitations | Conclusion

---

## L. Urutan Eksekusi

```bash
cd scripts/
python 01_data_audit.py
python 02_preprocessing.py
python 03_feature_engineering.py
python 04_modeling_classification.py
python 05_explanatory_analysis.py
python 06_clustering.py
python 07_reporting_assets.py
```

**Aturan:** Jangan jalankan 04–06 sebelum 03 selesai. Jalankan 07 terakhir.

---

## M. Catatan Progres

### Update terakhir
- Semua 7 script sudah tersedia di `scripts/`
- Script 01 dan 02 sudah diperbaiki: path relatif (pathlib), validasi lebih ketat
- Script 03–07 dibuat baru: feature engineering lengkap, full modeling pipeline,
  explanatory analysis, clustering dengan stability check, reporting consolidation
- `environment_clean.yml` ditambahkan: versi minimal portabel (hapus prefix Windows)
- TASKS.md diupdate: status script sudah benar ([x] untuk yang sudah ada)
