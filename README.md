# Proyek Olist — Customer Dissatisfaction Risk Modeling

## Judul Penelitian
**Predicting Customer Dissatisfaction Before Delivery: Explainable Risk Scoring from Fulfillment and Logistics Signals in E-Commerce**

## Ringkasan
Proyek ini membangun sistem **risk scoring dissatisfaction pelanggan sebelum outcome terjadi** pada dataset Olist (Brazilian e-commerce). Pendekatan utama: klasifikasi berbasis sinyal fulfillment dan logistik, dengan analisis explanatory tentang mekanisme dissatisfaction, dan translasi hasil ke prioritas intervensi operasional.

Proyek ini punya dua lapisan:
1. **Core**: klasifikasi dissatisfaction risk (pre-fulfillment dan in-fulfillment)
2. **Extension**: clustering untuk risk segmentation

Desain ini **LOCKED**. Semua keputusan metodologis ada di `DECISIONS.md`.

---

## Research Questions

| RQ | Pertanyaan |
|---|---|
| **RQ1** | Sejauh mana customer dissatisfaction dapat diprediksi menggunakan sinyal yang tersedia sebelum outcome? |
| **RQ2** | Seberapa besar tambahan nilai prediktif dari sinyal in-fulfillment dibanding hanya sinyal pre-fulfillment? |
| **RQ3** | Apakah expectation violation dan delivery-chain friction lebih informatif dibanding binary late-delivery flag? |
| **RQ4** | Seberapa efektif risk score dalam memprioritaskan order yang berpotensi berujung pada dissatisfaction? |

---

## Target

| Variabel | Definisi | Peran |
|---|---|---|
| `target_broad` | `review_score <= 3` → 1, else 0 | Primary |
| `target_severe` | `review_score <= 2` → 1, else 0 | Secondary |

---

## Skenario Modeling

| Skenario | Setting | Target | Status |
|---|---|---|---|
| S1 | Pre-fulfillment | Broad | Wajib |
| S2 | In-fulfillment | Broad | Wajib (primary model) |
| S3 | In-fulfillment | Severe | Wajib |
| S4 | Pre-fulfillment | Severe | Opsional |

---

## Algoritma

### Klasifikasi (tiap skenario)
| Algoritma | Peran |
|---|---|
| Logistic Regression | Interpretable baseline |
| Random Forest | Ensemble baseline |
| LightGBM | Main predictive model (SHAP analysis) |
| DummyClassifier | Baseline konteks evaluasi saja — bukan algoritma utama |

### Clustering (extension)
- K-Means + RobustScaler
- Tujuan: risk segmentation untuk intervention prioritization

---

## Keputusan Penting yang Dikunci

### Review dedup
Urutkan berdasarkan `review_creation_date`, keep **last** per `order_id`. Alasan: mencerminkan keputusan final pelanggan.

### Seller representative
Gunakan seller dengan **price tertinggi** per order sebagai seller representative. `seller_state` dan `distance_km` mengikuti seller ini.

### Threshold policy
Threshold untuk confusion matrix dan operational rule **dipilih dari validation set** — tidak dioptimalkan ulang di test set.

### Temporal split
| Split | Periode |
|---|---|
| Train | 2017-01 s/d 2017-11 |
| Validation | 2017-12 s/d 2018-02 |
| Test | 2018-03 s/d 2018-08 |

Data 2016 dan tail 2018-09/10 dibuang.

### Leakage boundary
Variabel berikut **tidak boleh masuk feature set model** — hanya untuk explanatory analysis:

```
estimation_error, is_late, actual_delivery_days,
seller_phase_days, logistics_phase_days,
order_delivered_customer_date, review_score,
review_comment_message, review_comment_title,
review_creation_date, review_answer_timestamp,
order_id, customer_id, customer_unique_id, seller_id
```

Leakage audit dijalankan otomatis di `04_modeling_classification.py` via `assert_no_leakage()`.

### Calibration
Calibration curve dan Brier score wajib untuk LightGBM. Post-hoc calibration (Platt/isotonic) opsional — hanya jika curve terlihat buruk, hanya pakai validation set.

---

## Fitur

### Pre-fulfillment (Setting S1, S4)
Diketahui saat `order_purchase_timestamp`:
`estimated_delivery_days`, `purchase_month`, `purchase_dayofweek`, `customer_state`, `seller_state`, `same_state`, `distance_km`, `n_items`, `n_unique_sellers`, `price_sum`, `freight_sum`, `freight_to_price_ratio`, `weight_sum`, `volume_sum`, `payment_value_sum`, `payment_installments_max`, `payment_type_main`, `top_category`

### In-fulfillment tambahan (Setting S2, S3)
Diketahui setelah proses fulfillment berjalan, sebelum delivery:
`approval_lag_hrs`, `carrier_pickup_lag_hrs`

Lihat `outputs/tables/03_feature_contract.csv` untuk data dictionary lengkap.

---

## Urutan Eksekusi

```bash
cd scripts/
python 01_data_audit.py
python 02_preprocessing.py
python 03_feature_engineering.py
python 04_modeling_classification.py   # paling lama
python 05_explanatory_analysis.py
python 06_clustering.py
python 07_reporting_assets.py
```

Jangan jalankan 04–07 sebelum 03 selesai. Jalankan 07 paling terakhir.

---

## Struktur Folder

```
olist_project/
├── data_raw/                          ← raw CSV (tidak di-commit ke git)
│   ├── olist_orders_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   └── product_category_name_translation.csv
│
├── data_interim/                      ← output preprocessing & feature eng (tidak di-commit)
│   ├── reviews_dedup.csv
│   ├── orders_filtered.csv
│   ├── order_level_features_pre.csv
│   ├── order_level_features_in.csv
│   ├── explanatory_variables.csv
│   ├── clustering_features.csv
│   └── cluster_assignments.csv
│
├── scripts/                           ← semua kode Python
│   ├── 01_data_audit.py
│   ├── 02_preprocessing.py
│   ├── 03_feature_engineering.py
│   ├── 04_modeling_classification.py
│   ├── 05_explanatory_analysis.py
│   ├── 06_clustering.py
│   └── 07_reporting_assets.py
│
├── outputs/
│   ├── tables/                        ← CSV hasil: metrics, feature contract, dll
│   ├── figures/                       ← grafik: ROC, PR, SHAP, calibration, dll
│   ├── models/                        ← model tersimpan (tidak di-commit)
│   └── metrics/
│
├── report/
│   ├── rtm1/
│   ├── rtm2/
│   ├── rtm3/
│   ├── rtm4/
│   └── paper_draft/
│
├── notebooks/                         ← opsional untuk eksplorasi
├── README.md                          ← dokumen ini
├── DECISIONS.md                       ← semua keputusan metodologis yang locked
├── TASKS.md                           ← progress tracker
├── olist_blueprint_final.md           ← blueprint eksekusi lengkap
├── environment.yml                    ← conda env lengkap (Windows-generated)
└── environment_clean.yml              ← conda env minimal, portabel (gunakan ini)
```

---

## Setup Environment

```bash
# Buat environment dari file minimal
conda env create -f environment_clean.yml

# Aktifkan
conda activate olist

# Register Jupyter kernel
python -m ipykernel install --user --name olist --display-name "Python (olist)"
```

---

## Output Utama yang Dihasilkan

| File | Isi |
|---|---|
| `03_feature_contract.csv` | Data dictionary 25 fitur |
| `04_classification_metrics.csv` | AUC, PR-AUC, Recall, F1, Brier semua skenario |
| `04_topk_capture.csv` | Intervention simulation (top 5/10/20/30%) |
| `04_thresholds.json` | Threshold terpilih per skenario dari validation |
| `06_cluster_profile.csv` | Profil tiap cluster (n, diss rate, karakteristik) |
| `07_report_summary.txt` | Ringkasan konsistensi semua output |

Grafik: ROC curves, PR curves, calibration plots, SHAP summary, cumulative gain, cluster radar.
