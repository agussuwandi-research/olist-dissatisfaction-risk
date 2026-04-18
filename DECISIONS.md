# Keputusan Terkunci — Proyek Olist

Dokumen ini berisi keputusan metodologis dan implementasi yang **sudah dikunci**. Jangan diubah tanpa alasan yang sangat kuat.

---

## 1. Fokus Proyek
### Core
Klasifikasi customer dissatisfaction risk sebelum outcome

### Extension
Clustering untuk risk segmentation

### Bukan fokus utama
- forecasting
- regresi sebagai jalur utama
- churn / retention
- association rules
- sentiment analysis utama
- dashboard-only project

---

## 2. Target
### Primary target
`target_broad = 1 jika review_score <= 3, else 0`

### Secondary target
`target_severe = 1 jika review_score <= 2, else 0`

---

## 3. Setting Modeling
### S1
Pre-fulfillment → Broad dissatisfaction

### S2
In-fulfillment → Broad dissatisfaction

### S3
In-fulfillment → Severe dissatisfaction

### S4 (opsional)
Pre-fulfillment → Severe dissatisfaction

---

## 4. Algoritma Klasifikasi
- Logistic Regression
- Random Forest
- LightGBM

### Peran
- **Logistic Regression**: interpretable baseline
- **Random Forest**: ensemble baseline
- **LightGBM**: main predictive model

### Baseline tambahan
- DummyClassifier hanya sebagai baseline konteks evaluasi
- bukan algoritma utama paper

---

## 5. Teknik Tambahan
### Dipilih
- Clustering dengan **K-Means + RobustScaler**

### Ditolak
- regresi sebagai jalur tambahan
- time series sebagai teknik tambahan
- association rules
- clustering yang terlalu kompleks sebagai default (mis. HDBSCAN / K-Prototypes) kecuali ada alasan kuat

---

## 6. Boundary Fitur
### Pre-fulfillment features
Diketahui saat `order_purchase_timestamp`

### In-fulfillment features
Tambahan:
- `approval_lag_hrs`
- `carrier_pickup_lag_hrs`

### Explanatory-only variables
- `actual_delivery_days`
- `is_late`
- `estimation_error`
- `seller_phase_days`
- `logistics_phase_days`

### Blacklist feature model
- `order_delivered_customer_date`
- `actual_delivery_days`
- `is_late`
- `estimation_error`
- `seller_phase_days`
- `logistics_phase_days`
- `review_score`
- `review_comment_message`
- `review_comment_title`
- `review_creation_date`
- `review_answer_timestamp`
- `order_id`
- `customer_id`
- `customer_unique_id`
- `seller_id`

---

## 7. Split Temporal
- **Train**: 2017-01 s/d 2017-11
- **Validation**: 2017-12 s/d 2018-02
- **Test**: 2018-03 s/d 2018-08

### Dibuang
- seluruh 2016
- tail 2018-09
- tail 2018-10

---

## 8. Review Deduplication
### Aturan
Urutkan berdasarkan `review_creation_date`, lalu **keep last per order_id**

### Alasan
- lebih defensible sebagai keputusan final pelanggan
- tidak seagresif mengambil worst review
- tidak seoptimistis mengambil best review

---

## 9. Seller Representative Rule
Gunakan **seller dengan nilai `price` tertinggi per order** sebagai seller representative.

### Konsekuensi
- `seller_state` mengikuti seller ini
- `distance_km` mengikuti seller ini

### Alasan
- lebih defensible daripada seller pertama
- lebih stabil daripada item terbanyak yang bisa tied
- mencerminkan kontribusi ekonomi dominan

---

## 10. Threshold Policy
- threshold dipilih dari **validation set**
- test set hanya untuk final reporting
- threshold tidak boleh dioptimalkan ulang di test set

---

## 11. Calibration
### Wajib
- calibration curve
- Brier score

### Opsional / conditional
- post-hoc calibration untuk LightGBM
- hanya jika calibration curve terlihat buruk
- hanya boleh memakai validation set

---

## 12. Clustering Rules
- lakukan setelah feature table pre-fulfillment selesai
- lakukan redundancy check sebelum K-Means
- gunakan RobustScaler
- pilih K berdasarkan elbow + silhouette + interpretability
- lakukan stability check lintas beberapa seed
- clustering hanya extension / risk segmentation
- clustering bukan pusat paper

---

## 13. Research Questions
### RQ1
Sejauh mana customer dissatisfaction dapat diprediksi sebelum outcome?

### RQ2
Seberapa besar tambahan nilai prediktif dari sinyal in-fulfillment?

### RQ3
Apakah expectation violation dan delivery-chain friction lebih informatif dari binary late flag?

### RQ4
Seberapa efektif risk score untuk memprioritaskan order berpotensi dissatisfied?

### Catatan
Tidak ada RQ5 terpisah untuk clustering.

---

## 14. Prioritas Proyek
### Wajib
- dataset order-level final
- feature contract
- split temporal
- leakage audit
- 3 skenario klasifikasi
- metrik inti
- lift / gain

### Penting
- expectation violation
- delivery-chain decomposition
- threshold dari validation
- calibration check

### Tambahan
- clustering

### Opsional
- S4
- calibration post-hoc

---

## 15. Prinsip Besar
- jangan tambah cabang riset baru
- jangan ubah core desain
- jangan campur explanatory variable ke feature model
- jangan sentuh test set sebelum final evaluation
- jangan mulai menulis hasil sebelum data table final beres