# Cara Deploy Dashboard ke Streamlit Community Cloud

Dashboard ini bisa diakses online secara gratis melalui Streamlit Community Cloud.

---

## Yang Dibutuhkan

- Akun GitHub (gratis): https://github.com
- Akun Streamlit Cloud (gratis, login pakai GitHub): https://streamlit.io/cloud

---

## Langkah-Langkah

### Step 1: Upload ke GitHub

1. Buka https://github.com/new — buat repository baru
   - Nama: `olist-dissatisfaction-dashboard` (atau terserah)
   - Visibility: **Public** (wajib untuk Streamlit Cloud gratis)
   - Jangan centang "Add a README file"
   - Klik **Create repository**

2. Di folder `olist_project/` di komputermu, buka terminal dan jalankan:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Olist dissatisfaction risk dashboard"
   git branch -M main
   git remote add origin https://github.com/NAMA_AKUN/NAMA_REPO.git
   git push -u origin main
   ```
   Ganti `NAMA_AKUN` dan `NAMA_REPO` sesuai akunmu.

   > **Catatan:** Push pertama akan memakan waktu beberapa menit karena
   > total repo ~220MB (termasuk data_raw dan data_interim).

### Step 2: Deploy di Streamlit Cloud

1. Buka https://share.streamlit.io (login dengan GitHub)
2. Klik **New app**
3. Isi formulir:
   - **Repository**: pilih repo yang baru dibuat
   - **Branch**: `main`
   - **Main file path**: `dashboard.py`
4. Klik **Deploy!**

Tunggu 2–5 menit. Dashboard akan live di URL seperti:
```
https://NAMA_AKUN-NAMA_REPO-dashboard-XXXXX.streamlit.app
```

---

## Yang Ada di Repo Ini

| Folder | Isi | Ukuran |
|---|---|---|
| `data_raw/` | Raw CSV Olist dataset | ~121MB |
| `data_interim/` | Hasil preprocessing & feature engineering | ~95MB |
| `outputs/tables/` | Hasil analisis (dibaca dashboard) | ~1MB |
| `outputs/figures/` | Grafik hasil analisis | ~1MB |
| `scripts/` | Pipeline script 01–11 | ~200KB |
| `dashboard.py` | Streamlit dashboard (9 tab) | ~60KB |

---

## Jika Ada Error Saat Deploy

| Error | Solusi |
|---|---|
| `ModuleNotFoundError` | Pastikan `requirements.txt` ada di root |
| `FileNotFoundError` | Pastikan folder `outputs/tables/` terupload |
| Push ditolak karena file terlalu besar | File individual masih <100MB, seharusnya OK |
