# Blueprint Eksekusi Final — Proyek Olist
> **Status:** LOCKED — v2. Perubahan dari v1: refinement teknis implementasi (A–H). Core desain tidak berubah.

---

## Hal yang Tidak Boleh Berubah

Bagian berikut adalah **fixed** dan tidak dapat diubah tanpa alasan yang sangat kuat:

- Target: broad dissatisfaction (≤3) primary, severe (≤2) secondary
- Setting: pre-fulfillment dan in-fulfillment
- Algoritma: Logistic Regression, Random Forest, LightGBM
- Split: Train 2017-01/11 | Val 2017-12/2018-02 | Test 2018-03/08
- `estimation_error`, `is_late`, `actual_delivery_days` hanya untuk explanatory analysis
- Clustering: K-Means + RobustScaler
- RQ: 4 RQ, tanpa RQ5 terpisah
- Review duplication: keep-last berdasarkan `review_creation_date`

---

## 1. Tujuan Kerja

### Inti Utama
Klasifikasi customer dissatisfaction risk sebelum outcome, dengan 3 skenario utama:

| Skenario | Setting | Target |
|---|---|---|
| S1 | Pre-fulfillment | Broad dissatisfaction (≤3) |
| S2 | In-fulfillment | Broad dissatisfaction (≤3) |
| S3 | In-fulfillment | Severe dissatisfaction (≤2) |
| S4 *(opsional)* | Pre-fulfillment | Severe dissatisfaction (≤2) — kerjakan jika waktu cukup |

### Lapisan Tambahan
Clustering untuk risk segmentation: bukan pusat paper, dipakai untuk memperkaya intervention prioritization, diposisikan formal di capstone report.

---

## 2. Struktur Folder Kerja

```
olist_project/
├── data_raw/
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
├── data_interim/
│   ├── reviews_dedup.csv
│   ├── orders_filtered.csv
│   ├── order_level_features_pre.csv
│   ├── order_level_features_in.csv
│   └── clustering_features.csv
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling_classification.ipynb
│   ├── 05_explanatory_analysis.ipynb
│   ├── 06_clustering.ipynb
│   └── 07_reporting_tables_figures.ipynb
│
├── outputs/
│   ├── tables/
│   ├── figures/
│   ├── models/
│   └── metrics/
│
└── report/
    ├── rtm1/
    ├── rtm2/
    ├── rtm3/
    ├── rtm4/
    └── paper_draft/
```

---

## 3. Urutan Kerja

**Fase A — Bangun Dataset Order-Level Final.** Jangan loncat ke modeling sebelum ini bersih.

- **A1:** Baca semua tabel, cek kunci join dan coverage
- **A2:** Filter `order_status == 'delivered'`, buang 2016, buang tail 2018-09 dan 2018-10
- **A3:** Dedup review — sort by `review_creation_date`, keep last per `order_id` → simpan sebagai `reviews_dedup.csv`
- **A4:** Bangun satu tabel order-level — semua fitur dikumpulkan ke level `order_id` sebelum split

---

## 4. Feature Contract / Data Dictionary

> **[DELIVERABLE WAJIB]** Buat sebagai artefak terpisah — sheet pertama di Excel atau tabel di notebook 03. Harus lengkap sebelum notebook 04 dijalankan. Tujuan: mencegah definisi berubah di tengah jalan, mencegah salah tafsir leakage, memudahkan audit.

Untuk setiap fitur, dokumentasikan kolom berikut:

| Kolom | Keterangan |
|---|---|
| Nama fitur | Nama kolom final di dataset |
| Definisi | Penjelasan singkat apa yang diukur |
| Rumus / cara hitung | Formula atau operasi yang dipakai |
| Satuan | Hari, jam, km, gram, cm³, IDR, dsb. |
| Sumber tabel | Tabel asal sebelum agregasi |
| Level agregasi | Per order, per item, dll. |
| Pre/In-fulfillment | Kapan fitur ini bisa diketahui |
| Masuk model? | Ya / Tidak |
| Alasan jika tidak | Khusus fitur explanatory-only |

---

## 5. Feature Engineering

### 5.1 Fitur Pre-Fulfillment
*(Diketahui saat `order_purchase_timestamp`)*

**Dari orders:**

| Fitur | Definisi | Satuan |
|---|---|---|
| `purchase_month` | Bulan dari `order_purchase_timestamp` | 1–12 |
| `purchase_dayofweek` | Hari dalam minggu | 0–6 |
| `estimated_delivery_days` | `estimated_delivery_date − purchase_timestamp` | Hari |

**Dari customers & sellers:**

| Fitur | Definisi |
|---|---|
| `customer_state` | State customer |
| `seller_state` | State dari seller representative (lihat aturan di bawah) |
| `same_state` | 1 jika `customer_state == seller_state`, else 0 |

> **Seller Representative Rule (LOCKED):** Gunakan seller dengan nilai `price` tertinggi per order sebagai seller representative. `seller_state` dan `distance_km` mengikuti seller ini. Lebih defensible dari "seller pertama" karena mencerminkan kontribusi ekonomi dominan, dan lebih stabil dari "item terbanyak" yang bisa tied.

**Dari order_items** *(agregasi per order_id)*:

| Fitur | Definisi | Satuan |
|---|---|---|
| `n_items` | `max(order_item_id)` per order | Count |
| `n_unique_sellers` | `nunique(seller_id)` per order | Count |
| `price_sum` | `sum(price)` | IDR |
| `freight_sum` | `sum(freight_value)` | IDR |
| `freight_to_price_ratio` | `freight_sum / (price_sum + 0.01)` | Rasio |
| `weight_sum` | `sum(product_weight_g)` | Gram |
| `volume_sum` | `sum(length × height × width)` | cm³ |

**Dari order_payments** *(agregasi per order_id)*:

| Fitur | Definisi |
|---|---|
| `payment_value_sum` | `sum(payment_value)` |
| `payment_installments_max` | `max(payment_installments)` |
| `payment_type_main` | Modus `payment_type` per order |

**Dari products** *(join ke items)*:

| Fitur | Definisi |
|---|---|
| `top_category` | Kategori produk dari item dengan `price` tertinggi per order |

**Dari geolocation** *(zip prefix centroid)*:

| Fitur | Definisi | Satuan |
|---|---|---|
| `distance_km` | Haversine antara centroid zip customer dan seller representative | km |

> Centroid = median lat/lng per zip prefix. Coverage ~99.7% untuk customer dan seller.

### 5.2 Fitur Tambahan In-Fulfillment
*(Hanya untuk Setting 2 dan 3)*

| Fitur | Definisi | Satuan |
|---|---|---|
| `approval_lag_hrs` | `order_approved_at − order_purchase_timestamp` | Jam |
| `carrier_pickup_lag_hrs` | `order_delivered_carrier_date − order_approved_at` | Jam |

> Definisi ini tidak boleh diubah. `carrier_pickup_lag_hrs` adalah fitur terpenting dalam in-fulfillment model. Jika definisinya berubah, perbandingan S1 vs S2 tidak valid.

---

## 6. Target

| Variabel | Definisi | Dipakai di |
|---|---|---|
| `target_broad` | 1 jika `review_score <= 3`, else 0 | S1, S2, S4 |
| `target_severe` | 1 jika `review_score <= 2`, else 0 | S3, S4 |

---

## 7. Variabel Explanatory-Only
*(Tidak boleh masuk ke feature set model apapun)*

| Variabel | Definisi |
|---|---|
| `actual_delivery_days` | `delivered_customer_date − purchase_timestamp` |
| `is_late` | 1 jika `delivered_customer_date > estimated_delivery_date` |
| `estimation_error` | `actual_delivery_days − estimated_delivery_days` |
| `seller_phase_days` | `delivered_carrier_date − purchase_timestamp` |
| `logistics_phase_days` | `delivered_customer_date − delivered_carrier_date` |

---

## 8. Quality Gate: Leakage Audit Wajib

> **[QUALITY GATE — jalankan sebelum notebook 04, sebelum `.fit()` apapun]**

```python
LEAKAGE_BLACKLIST = [
    # Post-outcome delivery
    'order_delivered_customer_date',
    'actual_delivery_days',
    'is_late',
    'estimation_error',
    'seller_phase_days',
    'logistics_phase_days',
    # Target / review
    'review_score',
    'review_comment_message',
    'review_comment_title',
    'review_creation_date',
    'review_answer_timestamp',
    # Identifiers — bukan fitur model
    'order_id',
    'customer_id',
    'customer_unique_id',
    'seller_id',
]

def assert_no_leakage(feature_df, blacklist=LEAKAGE_BLACKLIST):
    violations = [col for col in blacklist if col in feature_df.columns]
    assert len(violations) == 0, f"LEAKAGE DETECTED: {violations}"
    print("Leakage check passed.")
```

Jalankan untuk `X_train`, `X_val`, dan `X_test` masing-masing.

---

## 9. Temporal Split

| Split | Periode | Tujuan |
|---|---|---|
| **Train** | 2017-01 s/d 2017-11 | Pelatihan model |
| **Validation** | 2017-12 s/d 2018-02 | Hyperparameter tuning + threshold selection |
| **Test** | 2018-03 s/d 2018-08 | Final evaluation — sentuh sekali di akhir |

Wajib dilaporkan: jumlah order, broad positive rate, dan severe positive rate per split.

> Validation set memiliki positive rate lebih tinggi (~24.7%) karena peak season — akui sebagai temuan temporal heterogeneity, bukan disembunyikan.

---

## 10. Threshold Policy

> **[RULE WAJIB]**

1. Threshold untuk confusion matrix dan operational rule **ditentukan dari validation set**
2. Test set hanya dipakai untuk final reporting dengan threshold yang sudah dikunci dari validation
3. Threshold **tidak boleh dioptimalkan ulang di test set**

Cara memilih threshold dari validation: tentukan target recall minimum (mis. ≥ 0.50), pilih threshold tertinggi yang masih memenuhi target, catat eksplisit sebelum membuka test set.

---

## 11. Preprocessing

### 11.1 Missing Value Handling

| Tipe Fitur | Strategi |
|---|---|
| Numerik kontinu | Median imputation — dihitung dari **train set saja** |
| Kategorikal | `"Unknown"` atau modus dari train set |
| Binary | 0 jika absence berarti false/tidak ada |

### 11.2 Encoding
- `top_category`, `customer_state`, `seller_state`: top-N (mis. top-15) lalu `"Other"`, kemudian one-hot
- `payment_type_main`: one-hot (4 kategori)
- Untuk LightGBM: label encoding langsung dapat dipakai

### 11.3 Scaling

| Konteks | Metode |
|---|---|
| Logistic Regression | StandardScaler pada fitur numerik |
| Random Forest | Tidak wajib |
| LightGBM | Tidak wajib |
| K-Means | **RobustScaler wajib** |

---

## 12. Eksperimen Klasifikasi

### Algoritma

| Algoritma | Peran di paper | Peran di tugas |
|---|---|---|
| Logistic Regression | Interpretable baseline | Algoritma 1 |
| Random Forest | Ensemble baseline | Algoritma 2 |
| LightGBM | Main model, SHAP analysis | Algoritma 3 |
| **Dummy Classifier** | Konteks evaluasi saja — bukan algoritma formal | Baris referensi di tabel metrik |

> **Dummy Classifier:** `DummyClassifier(strategy='stratified')` — jalankan untuk mendapatkan baseline PR-AUC dan lift "lebih baik dari acak." Di paper: baris referensi di tabel, bukan model utama. Di capstone: cantumkan sebagai "baseline comparison."

### Grid Eksperimen

| Skenario | Setting | Target |
|---|---|---|
| S1 | Pre-fulfillment | Broad |
| S2 | In-fulfillment | Broad |
| S3 | In-fulfillment | Severe |
| S4 *(opsional)* | Pre-fulfillment | Severe |

Setiap skenario: LR + RF + LightGBM. Total 9 model minimum.

---

## 13. Metrik Evaluasi

| Metrik | Prioritas |
|---|---|
| PR-AUC / Average Precision | Tertinggi |
| Recall kelas positif | Tertinggi |
| F1-score (threshold dari validation) | Tinggi |
| Lift / Cumulative Gain (top-k) | Tinggi |
| Calibration curve | Tinggi |
| Brier Score | Menengah |
| ROC-AUC | Menengah |
| Confusion Matrix | Menengah |

---

## 14. Calibration untuk LightGBM

> **[OPSIONAL/CONDITIONAL]**

**Langkah wajib:** Plot calibration curve + hitung Brier score untuk LightGBM pada test set.

**Langkah opsional** *(jalankan hanya jika calibration curve terlihat buruk)*:
- Post-hoc calibration menggunakan **validation set saja**
- Pilih Platt scaling (`method='sigmoid'`) atau isotonic regression (`method='isotonic'`)
- Laporkan Brier score sebelum dan sesudah

> Jangan terapkan calibration menggunakan test set.

---

## 15. Tabel Output Wajib

**Tabel 1 — Ringkasan Dataset:** jumlah order, delivered, periode, review final, broad rate, severe rate.

**Tabel 2 — Distribusi Split Temporal:** jumlah order, broad positive rate, severe positive rate per split.

**Tabel 3 — Feature Contract:** nama, definisi, rumus, satuan, sumber, agregasi, pre/in, masuk model?, catatan.

**Tabel 4 — Hasil Klasifikasi:** skenario, model, ROC-AUC, PR-AUC, Recall, F1, Brier Score.

**Tabel 5 — Top-K Capture:** skenario, model, top 5%, 10%, 20%, 30%.

**Tabel 6 — Profil Cluster:** cluster, n orders, diss rate, avg freight ratio, avg est days, avg distance, avg weight, % same-state, dominant category.

---

## 16. Grafik Wajib

**Klasifikasi:** ROC curve, PR curve, calibration plot, cumulative gain/lift chart, SHAP summary (LightGBM S2), koefisien LR (S1).

**Explanatory:** dissatisfaction rate vs `estimation_error` (bin plot), boxplot `seller_phase_days` vs diss, boxplot `logistics_phase_days` vs diss, bar chart diss per kategori, bar chart same-state vs cross-state.

**Clustering:** elbow plot, silhouette vs K, radar/bar profil cluster, bar chart diss rate per cluster.

---

## 17. Workflow Clustering

> Kerjakan setelah feature table pre-fulfillment selesai.

### 17.1 Cek Redundansi Fitur (Wajib Sebelum K-Means)

Hitung korelasi Pearson. Periksa pasangan ini secara khusus:

| Pasangan | Risiko |
|---|---|
| `price_sum` vs `payment_value_sum` | Sangat tinggi — pilih satu |
| `weight_sum` vs `volume_sum` | Tinggi — pertimbangkan drop salah satu |
| `freight_sum` vs `freight_to_price_ratio` | Sedang — keduanya mengukur hal berbeda |
| `same_state` vs `distance_km` | Parsial — keduanya boleh dipertahankan |

**Aturan:** Jika korelasi > 0.85, pilih fitur yang lebih interpretatif. Dokumentasikan keputusan di notebook 06.

### 17.2 Feature Set Clustering
Setelah cek redundansi, kandidat fitur:

```
freight_sum ATAU freight_to_price_ratio (pilih satu)
price_sum
weight_sum ATAU volume_sum (pilih satu)
estimated_delivery_days
distance_km
n_items
payment_installments_max
same_state
```

### 17.3 Preprocessing
**RobustScaler wajib** — karena `freight_sum` dan `weight_sum` memiliki outlier berat.

### 17.4 Pemilihan K
Uji K = 2 sampai 8. Pilih berdasarkan elbow + silhouette + interpretability bisnis. Jangan pilih K hanya karena silhouette terbesar jika cluster tidak bisa diberi narasi.

### 17.5 Stability Check (Wajib Sebelum Interpretasi)

```python
from sklearn.cluster import KMeans

# Training utama
km = KMeans(n_clusters=K, n_init=20, random_state=42)
km.fit(X_scaled)

# Cek stabilitas dengan beberapa seed
for seed in [0, 7, 42, 99, 123]:
    km_check = KMeans(n_clusters=K, n_init=20, random_state=seed)
    km_check.fit(X_scaled)
    # Bandingkan profil cluster (dissatisfaction rate per cluster)
    # Interpretasi hanya dilakukan jika profil konsisten lintas seed
```

### 17.6 Profiling & Narasi
Untuk tiap cluster: size, diss rate (broad + severe), avg freight, est days, distance, weight, % same-state, kategori dominan. Target: label operasional bermakna, bukan sekadar angka.

---

## 18. Explanatory Analysis

### 18.1 Expectation Violation
- Bin `estimation_error`, plot diss rate per bin
- Tunjukkan asimetri: *"5 hari lebih cepat → 16.7%; 5 hari lebih lambat → 85.9%; rasio 5.1x"*
- Framing: *"consistent with loss-aversion patterns"* — bukan klaim kausal

### 18.2 Delivery-Chain Decomposition
- `seller_phase_days` vs diss (AUC univariat: 0.572)
- `logistics_phase_days` vs diss (AUC univariat: 0.630)
- Narasi: logistics phase lebih prediktif — implikasi intervensi lebih relevan di last-mile

---

## 19. Urutan Notebook

| Notebook | Output Utama |
|---|---|
| `01_data_audit` | Row count, join success rate |
| `02_preprocessing` | Delivered-only dataset, review dedup, temporal filter, target table |
| `03_feature_engineering` | Pre-fulfillment table, in-fulfillment table, explanatory variables, **feature contract** |
| `04_modeling_classification` | **Leakage audit → assert_no_leakage()**, training, threshold dari validation, metrics, calibration, gain/lift |
| `05_explanatory_analysis` | Estimation error, delivery chain, category descriptive |
| `06_clustering` | Redundancy check, elbow, silhouette, **stability check**, cluster assignment, profiling |
| `07_reporting_assets` | Semua tabel final, semua grafik final, metrics CSV |

---

## 20. Pembagian Isi RTM

**RTM1:** Latar belakang, masalah, alasan Olist, 4 RQ, tinjauan paper, rencana metode.

**RTM2:** Deskripsi dataset, join, filter, dedup review, target definition, split + distribusi, feature engineering, **feature contract eksplisit**, leakage boundary.

**RTM3:** 3 algoritma × 3 skenario = 9 model, evaluasi per skenario (Tabel 4 + 5), clustering formal (redundancy check, K selection, stability check), profil cluster (Tabel 6).

**RTM4:** Pembahasan model terbaik, gap S1 vs S2, expectation violation, delivery-chain decomposition, cluster interpretation, implikasi intervensi, keterbatasan.

---

## 21. Research Questions

| RQ | Pertanyaan | Dijawab di |
|---|---|---|
| **RQ1** | Sejauh mana customer dissatisfaction dapat diprediksi sebelum outcome? | Setting 1 — pre-fulfillment |
| **RQ2** | Seberapa besar tambahan nilai prediktif dari sinyal in-fulfillment? | Perbandingan S1 vs S2 |
| **RQ3** | Apakah expectation violation dan delivery-chain friction lebih informatif dari binary late flag? | Explanatory analysis |
| **RQ4** | Seberapa efektif risk score untuk memprioritaskan order berpotensi dissatisfied? | Intervention prioritization + clustering |

---

## 22. Timeline

| Minggu | Fokus |
|---|---|
| **1** | Audit tabel, dedup review, delivered-only dataset, temporal split |
| **2** | Feature engineering, geolocation/distance, target, **feature contract**, missing value report |
| **3** | **Leakage audit**, training semua skenario, **threshold dari validation**, metrik, calibration, lift/gain |
| **4** | Explanatory analysis, **clustering + redundancy + stability**, profiling, final tables/figures |
| **5** | Susun RTM dan draft paper, cek konsistensi angka |

---

## 23. Prioritas Jika Waktu Mepet

| Prioritas | Komponen |
|---|---|
| **1 — Wajib** | Dataset order-level, feature contract, target, split, leakage audit, 3 skenario klasifikasi, metrik inti, lift/gain |
| **2 — Penting** | Expectation violation, delivery-chain decomposition, threshold dari validation, calibration check |
| **3 — Tambahan** | Clustering (dengan stability check) |
| **4 — Opsional** | Skenario ke-4, calibration post-hoc |

---

## 24. Checklist Kualitas Sebelum Laporan Ditulis

- [ ] Feature contract (Tabel 3) selesai sebelum notebook 04 dijalankan
- [ ] Seller representative rule dikunci: seller dengan `price` tertinggi per order
- [ ] `assert_no_leakage()` passed untuk `X_train`, `X_val`, `X_test`
- [ ] Threshold dipilih dari **validation set**, bukan test set
- [ ] Feature table tidak ada duplikat `order_id`
- [ ] Target tidak ada null
- [ ] Split temporal tidak bocor
- [ ] Semua model: train di train, tune di validation, final di **test**
- [ ] Dummy classifier dijalankan sebagai baseline konteks
- [ ] Metrik lengkap untuk semua skenario
- [ ] Lift/gain chart ada
- [ ] Calibration curve + Brier score ada
- [ ] Redundancy check fitur clustering didokumentasikan
- [ ] Clustering stability check dijalankan (`n_init=20`, beberapa seed)
- [ ] Clustering punya justifikasi K (elbow + silhouette)
- [ ] Setiap cluster punya narasi operasional
- [ ] `estimation_error`, `is_late`, `actual_delivery_days` **tidak ada** di feature matrix model

---

*Dokumen ini adalah versi final v2. Core desain tidak berubah dari v1. Refinement teknis A–H sudah diintegrasikan.*
