"""
dashboard.py — Olist Customer Dissatisfaction Risk Modeling
Dashboard Business Intelligence dengan penjelasan di tiap seksi (awam + detail teknis).
Jalankan: streamlit run dashboard.py
"""
import json, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Olist Dissatisfaction Risk — BI Dashboard",
                   page_icon="📦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.kpi {background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:1rem 1.2rem;height:100%;}
.kpi .v {font-size:2rem;font-weight:800;color:#0f172a;line-height:1.1;}
.kpi .l {font-size:.78rem;color:#64748b;text-transform:uppercase;letter-spacing:.04em;margin-bottom:.3rem;}
.kpi .d {font-size:.8rem;color:#94a3b8;margin-top:.3rem;}
.intro {background:#eff6ff;border-left:4px solid #2563eb;border-radius:0 8px 8px 0;
        padding:.85rem 1.1rem;font-size:.95rem;color:#1e3a5f;margin-bottom:.6rem;line-height:1.55;}
.read {background:#f8fafc;border:1px dashed #cbd5e1;border-radius:8px;padding:.6rem .9rem;
       font-size:.88rem;color:#334155;margin:.4rem 0;}
.take {background:#052e16;border-radius:8px;padding:.7rem 1rem;font-size:.9rem;color:#bbf7d0;margin:.5rem 0;}
</style>""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"
INTERIM = ROOT / "data_interim"
BLUE, RED, GREEN, GRAY, AMBER = "#2563eb", "#dc2626", "#16a34a", "#94a3b8", "#f59e0b"

COORD = {'AC':(-9.97,-67.81),'AL':(-9.66,-35.73),'AM':(-3.12,-60.02),'AP':(0.03,-51.07),
 'BA':(-12.97,-38.50),'CE':(-3.73,-38.52),'DF':(-15.79,-47.88),'ES':(-20.32,-40.34),
 'GO':(-16.68,-49.25),'MA':(-2.53,-44.30),'MG':(-19.92,-43.94),'MS':(-20.44,-54.65),
 'MT':(-15.60,-56.10),'PA':(-1.46,-48.50),'PB':(-7.12,-34.88),'PE':(-8.05,-34.90),
 'PI':(-5.09,-42.80),'PR':(-25.43,-49.27),'RJ':(-22.91,-43.17),'RN':(-5.79,-35.21),
 'RO':(-8.76,-63.90),'RR':(2.82,-60.67),'RS':(-30.03,-51.23),'SC':(-27.59,-48.55),
 'SE':(-10.91,-37.07),'SP':(-23.55,-46.63),'TO':(-10.18,-48.33)}

@st.cache_data(show_spinner=False)
def load(name, folder="tables"):
    p = (TABLES if folder == "tables" else INTERIM) / name
    try: return pd.read_csv(p)
    except Exception: return None

@st.cache_data(show_spinner=False)
def brazil_geojson():
    try:
        url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
        return json.load(urllib.request.urlopen(url, timeout=15))
    except Exception:
        return None

def plot(fig, h=430):
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=h,
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Segoe UI", size=13))
    st.plotly_chart(fig, use_container_width=True)

def kpi(col, label, value, desc=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="d">{desc}</div></div>', unsafe_allow_html=True)

def intro(txt): st.markdown(f'<div class="intro">{txt}</div>', unsafe_allow_html=True)
def read(txt):  st.markdown(f'<div class="read">📊 <b>Cara baca:</b> {txt}</div>', unsafe_allow_html=True)
def take(txt):  st.markdown(f'<div class="take">💡 <b>Intinya:</b> {txt}</div>', unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.title("📦 Olist Risk BI")
st.sidebar.caption("Prediksi ketidakpuasan pelanggan sebelum barang diterima")
PAGE = st.sidebar.radio("Navigasi", [
    "🏠 Panduan", "1 · Ringkasan", "2 · Performa Model", "3 · Peta & Geografi",
    "4 · Simulasi Intervensi", "5 · Ekspektasi & Logistik", "6 · Analisis Error",
    "7 · Segmentasi Risiko", "8 · Risk Scorer (KBS)"])
st.sidebar.markdown("---")
st.sidebar.caption("Kelompok 8 · Data Visualization & Business Intelligence")

# ═══════════════════════════════════════════════════════════════════════════════
if PAGE == "🏠 Panduan":
    st.title("Panduan Dashboard")
    intro("Dashboard ini merangkum penelitian <b>prediksi ketidakpuasan pelanggan sebelum barang diterima</b> "
          "pada data e-commerce Olist. Tujuannya: menandai order yang <b>berisiko berujung review negatif</b> "
          "lebih awal, agar tim operasional bisa bertindak sebelum pelanggan kecewa. "
          "Model terbaik: <b>Logistic Regression (skenario in-fulfillment)</b>.")
    st.subheader("Apa isi tiap menu di kiri?")
    guide = pd.DataFrame({
        "Menu": ["1 · Ringkasan","2 · Performa Model","3 · Peta & Geografi","4 · Simulasi Intervensi",
                 "5 · Ekspektasi & Logistik","6 · Analisis Error","7 · Segmentasi Risiko","8 · Risk Scorer"],
        "Menjawab pertanyaan": [
            "Seberapa besar masalahnya & seberapa bagus modelnya (angka kunci).",
            "Model & skenario mana yang terbaik, dan apakah bedanya nyata (uji statistik)?",
            "Di wilayah mana risiko ketidakpuasan paling tinggi?",
            "Kalau hanya bisa menindak sebagian order, prioritaskan yang mana?",
            "Kenapa pelanggan kecewa? (janji pengiriman vs kenyataan)",
            "Di mana model salah menebak, dan kenapa?",
            "Bisakah order dikelompokkan jadi segmen berisiko?",
            "Alat coba-coba: masukkan ciri order → keluar tingkat risikonya."]})
    st.dataframe(guide, use_container_width=True, hide_index=True)
    take("Tiap seksi punya kotak biru (penjelasan awam), kotak putus-putus 'Cara baca' di bawah grafik, "
         "dan expander '🔍 Detail teknis' bila kamu ingin definisi lengkapnya.")
    st.caption("Istilah singkat: **ketidakpuasan** = review ≤ 3 bintang · **pra-pengiriman (S1)** = info saat order dibuat · "
               "**in-fulfillment (S2)** = + sinyal proses pengiriman · **AUC** = ukuran ketepatan model (0,5=asal tebak, 1=sempurna).")

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "1 · Ringkasan":
    st.title("Ringkasan")
    intro("Gambaran umum: seberapa besar masalah ketidakpuasan, dan seberapa baik model kita menangkapnya. "
          "Empat angka di bawah adalah ringkasan paling penting dari seluruh penelitian.")
    split = load("02_split_summary.csv"); metrics = load("04_classification_metrics.csv"); topk = load("04_topk_capture.csv")
    n_total = int(split["n_orders"].sum()) if split is not None else 95568
    best = 0.627
    if metrics is not None:
        try: best = metrics.query("scenario=='S2_in_broad' and model=='LR'")["roc_auc"].iloc[0]
        except Exception: pass
    lift = 2.0
    if topk is not None:
        try: lift = topk.query("scenario=='S2_in_broad' and model=='LR'")["top_10pct_recall"].iloc[0]/0.10
        except Exception: pass
    c = st.columns(4)
    kpi(c[0], "Total order dianalisis", f"{n_total:,}".replace(",","."), "delivered + ada review")
    kpi(c[1], "Tingkat ketidakpuasan", "≈21%", "1 dari 5 order review ≤ 3")
    kpi(c[2], "Ketepatan model terbaik", f"AUC {best:.3f}", "Logistic Regression (S2)")
    kpi(c[3], "Efektivitas prioritas", f"{lift:.1f}× lift", "vs menandai order acak")
    read("Angka <b>AUC</b> mendekati 1 makin bagus; 0,5 berarti seperti menebak koin. AUC ~0,63 di sini tergolong "
         "moderat — bagian 'Analisis Error' menjelaskan kenapa lebih tinggi dari itu sulit dicapai secara jujur.")
    st.write(""); left, right = st.columns([2,1])
    monthly = load("01_monthly_distribution.csv")
    if monthly is not None and {"ym","n_orders"} <= set(monthly.columns):
        with left:
            st.subheader("Volume Order per Bulan")
            fig = px.area(monthly, x="ym", y="n_orders", color_discrete_sequence=[BLUE])
            fig.update_layout(xaxis_title="", yaxis_title="Jumlah order"); plot(fig, 340)
            read("Menunjukkan pertumbuhan transaksi Olist dari waktu ke waktu; lonjakan akhir 2017 = musim liburan.")
    if split is not None:
        with right:
            st.subheader("Pembagian Data (Train/Val/Test)")
            fig = px.bar(split, x="split", y="n_orders", color="broad_positive_rate",
                         color_continuous_scale="Reds", text="n_orders")
            fig.update_layout(xaxis_title="", yaxis_title="", coloraxis_showscale=False); plot(fig, 340)
    with st.expander("🔍 Detail teknis — definisi & pembagian data"):
        st.markdown("- **Target:** `ketidakpuasan (broad)` = review_score ≤ 3 (prevalensi ~21%).\n"
                    "- **Split temporal:** Train (2017-01..11), Validasi (2017-12..2018-02), Test (2018-03..08) — "
                    "dibagi berdasarkan waktu agar meniru kondisi nyata & mencegah kebocoran data.\n"
                    "- Validasi punya tingkat ketidakpuasan lebih tinggi (~24,9%) karena mencakup musim liburan.")
        if split is not None: st.dataframe(split, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "2 · Performa Model":
    st.title("Performa Model")
    intro("Kami menguji <b>3 algoritma</b> (Logistic Regression, Random Forest, LightGBM) pada <b>2 skenario fitur</b>: "
          "<b>S1</b> = hanya info saat order dibuat, <b>S2</b> = ditambah sinyal proses pengiriman. "
          "Tujuannya memilih model terbaik dan membuktikan bahwa perbedaannya bukan kebetulan.")
    metrics = load("04_classification_metrics.csv")
    if metrics is not None:
        d = metrics[metrics.model != "Dummy"]
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("ROC-AUC (ketepatan membedakan)")
            fig = px.bar(d, x="scenario", y="roc_auc", color="model", barmode="group",
                         color_discrete_sequence=[BLUE, GREEN, AMBER], text="roc_auc")
            fig.update_traces(texttemplate="%{text:.3f}"); fig.update_layout(yaxis_range=[.55,.67], yaxis_title="AUC (↑ lebih baik)"); plot(fig)
        with c2:
            st.subheader("PR-AUC (fokus kelas minoritas)")
            fig = px.bar(d, x="scenario", y="pr_auc", color="model", barmode="group",
                         color_discrete_sequence=[BLUE, GREEN, AMBER], text="pr_auc")
            fig.update_traces(texttemplate="%{text:.3f}"); fig.update_layout(yaxis_title="PR-AUC (↑ lebih baik)"); plot(fig)
        read("Batang lebih tinggi = lebih baik. Perhatikan <b>S2 selalu > S1</b> untuk semua algoritma "
             "→ menambah sinyal proses pengiriman benar-benar membantu. Dan <b>LR (biru) menang</b> di S2.")
        take("Model terbaik = <b>Logistic Regression (S2)</b>, AUC 0,627. Sederhana tapi mengungguli LightGBM, "
             "dan mudah dijelaskan — cocok jadi dasar Risk Scorer.")
    delong = load("08_delong_results.csv")
    if delong is not None:
        st.subheader("Apakah bedanya nyata? — Uji DeLong")
        intro("Uji DeLong = tes statistik untuk memastikan selisih AUC itu <b>signifikan</b> (nyata), "
              "bukan sekadar keberuntungan angka. Kolom <code>p_value</code> &lt; 0,05 = nyata.")
        st.dataframe(delong[["comparison","auc_A","auc_B","delta_auc","p_value","significance"]],
                     use_container_width=True, hide_index=True)
        read("Semua baris 'S1 vs S2' punya p&lt;0,001 (`***`) → peningkatan dari sinyal in-fulfillment <b>terbukti nyata</b>.")
        with st.expander("🔍 Detail teknis — arti metrik"):
            st.markdown("- **ROC-AUC:** peluang model memberi skor lebih tinggi ke order 'kecewa' dibanding 'puas' acak. 0,5=acak.\n"
                        "- **PR-AUC:** lebih informatif saat kelas positif minoritas (21%).\n"
                        "- **DeLong:** uji non-parametrik untuk membandingkan dua AUC pada data uji yang sama.\n"
                        "- Setelah tuning Optuna, LightGBM hanya setara LR (Δ+0,003, tidak signifikan) → hubungan bersifat linear.")

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "3 · Peta & Geografi":
    st.title("Peta Distribusi Pengiriman & Risiko Geografis")
    intro("Di mana ketidakpuasan paling tinggi secara geografis? Ini penting untuk <b>intervensi berbasis wilayah</b> "
          "(mis. memperkuat last-mile ke daerah tertentu).")
    stt = load("05_state_dissatisfaction.csv"); geo = brazil_geojson()
    if stt is None or "customer_state" not in stt.columns:
        st.warning("Butuh `outputs/tables/05_state_dissatisfaction.csv`.")
    else:
        stt = stt.copy()
        stt["lat"] = stt.customer_state.map(lambda s: COORD.get(s,(None,None))[0])
        stt["lon"] = stt.customer_state.map(lambda s: COORD.get(s,(None,None))[1])
        stt["diss_pct"] = (stt["diss_rate"]*100).round(1)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Choropleth — Risiko per Negara Bagian")
            if geo is not None:
                try:
                    props = geo["features"][0]["properties"]
                    key = next((k for k,v in props.items() if isinstance(v,str) and len(v)==2 and v.isupper()), "sigla")
                    fig = px.choropleth(stt, geojson=geo, locations="customer_state", featureidkey="properties."+key,
                                        color="diss_pct", color_continuous_scale="Reds", scope="south america",
                                        labels={"diss_pct":"Ketidakpuasan %"})
                    fig.update_geos(fitbounds="locations", visible=False); plot(fig, 460)
                except Exception as e: st.info(f"Choropleth tidak tersedia: {e}")
            else: st.info("Peta batas negara bagian perlu internet (gagal diambil).")
            read("Makin <b>merah</b> = tingkat ketidakpuasan makin tinggi di provinsi itu.")
        with c2:
            st.subheader("Bubble — Volume vs Risiko")
            b = stt.dropna(subset=["lat"])
            fig = px.scatter_geo(b, lat="lat", lon="lon", size="n", color="diss_pct",
                                 color_continuous_scale="Reds", scope="south america",
                                 hover_name="customer_state", size_max=40,
                                 labels={"n":"Jumlah order","diss_pct":"Ketidakpuasan %"})
            fig.update_geos(fitbounds="locations", visible=True); plot(fig, 460)
            read("<b>Ukuran</b> lingkaran = banyaknya order; <b>warna</b> = tingkat risiko. "
                 "Lingkaran besar & terang di São Paulo = pusat volume, risiko rendah.")
        st.subheader("Alur Pengiriman Seller → Customer (top 40 rute)")
        pre = load("order_level_features_pre.csv", "interim")
        if pre is not None and "seller_state" in pre.columns:
            fl = pre.groupby(["seller_state","customer_state"]).size().reset_index(name="n")
            fl = fl[fl.seller_state != fl.customer_state].sort_values("n", ascending=False).head(40)
            mx = fl.n.max(); fig = go.Figure()
            for _, r in fl.iterrows():
                a, d2 = COORD.get(r.seller_state), COORD.get(r.customer_state)
                if a and d2:
                    fig.add_trace(go.Scattergeo(lon=[a[1],d2[1]], lat=[a[0],d2[0]], mode="lines",
                                  line=dict(width=max(.5, r.n/mx*7), color="rgba(37,99,235,0.45)"), showlegend=False))
            fig.add_trace(go.Scattergeo(lon=stt.lon, lat=stt.lat, mode="markers",
                          marker=dict(size=5, color=RED), text=stt.customer_state, showlegend=False))
            fig.update_geos(scope="south america", visible=True, showcountries=True); plot(fig, 460)
            read("Tiap garis = rute pengiriman antar provinsi; makin <b>tebal</b> = makin banyak order lewat rute itu. "
                 "Terlihat São Paulo jadi hub yang memancar ke seluruh Brasil.")
        else:
            st.info("Peta alur perlu `data_interim/order_level_features_pre.csv` (tersedia saat dijalankan lokal).")
        take("Ketidakpuasan terpusat di <b>Utara & Timur Laut</b> (Maranhão 29,9%) — jauh dari hub seller São Paulo "
             "(18,6%). Makin jauh dari hub → makin sering telat → makin tinggi risiko.")

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "4 · Simulasi Intervensi":
    st.title("Simulasi Intervensi Operasional")
    intro("Bayangkan tim hanya punya sumber daya untuk menindak <b>sebagian</b> order. Grafik ini menunjukkan: "
          "kalau kita urutkan order dari paling berisiko lalu tangani X% teratas, <b>berapa persen pelanggan "
          "yang bakal kecewa berhasil kita cegat?</b>")
    topk = load("04_topk_capture.csv")
    if topk is not None:
        s2 = topk[(topk.scenario=="S2_in_broad") & (topk.model.isin(["LR","RF","LightGBM"]))]
        cols = [c for c in ["top_5pct_recall","top_10pct_recall","top_20pct_recall","top_30pct_recall"] if c in topk.columns]
        pct = [5,10,20,30]; fig = go.Figure()
        for _, r in s2.iterrows():
            ys = [0] + [r[c]*100 for c in cols] + [100]
            fig.add_trace(go.Scatter(x=[0]+pct+[100], y=ys, mode="lines+markers", name=r.model))
        fig.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines", name="Acak", line=dict(dash="dash", color=GRAY)))
        fig.update_layout(xaxis_title="% order berisiko yang ditandai", yaxis_title="% pelanggan kecewa tertangkap")
        st.subheader("Cumulative Gain (skenario S2)"); plot(fig)
        read("Sumbu-X = berapa persen order kita tandai; sumbu-Y = berapa persen pelanggan kecewa yang tertangkap. "
             "Garis <b>makin jauh di atas garis putus-putus 'Acak'</b> = model makin berguna.")
        try:
            lr = s2[s2.model=="LR"].iloc[0]; c = st.columns(3)
            kpi(c[0], "Tandai 10% teratas", f"{lr['top_10pct_recall']*100:.1f}%", "pelanggan kecewa tertangkap")
            kpi(c[1], "Lift", f"{lr['top_10pct_recall']/0.10:.2f}×", "dibanding memilih acak")
            kpi(c[2], "Tandai 20% teratas", f"{lr['top_20pct_recall']*100:.1f}%", "pelanggan kecewa tertangkap")
        except Exception: pass
        take("Cukup menandai <b>10% order paling berisiko</b>, tim sudah menangkap <b>~20%</b> calon pelanggan kecewa "
             "— dua kali lebih efisien daripada menindak order acak.")

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "5 · Ekspektasi & Logistik":
    st.title("Kenapa Pelanggan Kecewa?")
    intro("Ketidakpuasan sering bukan soal 'telat atau tidak', melainkan soal <b>selisih antara janji dan kenyataan</b> "
          "pengiriman. Bagian ini menyelidiki mekanismenya.")
    ebin = load("05_estimation_error_bins.csv"); asym = load("05_asymmetry_check.csv"); lae = load("05_late_vs_error_auc.csv")
    c1, c2 = st.columns(2)
    if ebin is not None and "diss_pct" in ebin.columns:
        with c1:
            st.subheader("Ketidakpuasan vs Selisih Estimasi")
            fig = px.bar(ebin, x="err_bin", y="diss_pct", color="diss_pct", color_continuous_scale="RdYlGn_r")
            fig.update_layout(xaxis_title="Selisih hari (− = lebih cepat, + = lebih lambat)", yaxis_title="Ketidakpuasan %", coloraxis_showscale=False); plot(fig)
            read("Ke kanan = pengiriman makin melewati tanggal yang dijanjikan → ketidakpuasan melonjak.")
    if asym is not None:
        with c2:
            st.subheader("Asimetri: Lebih Cepat vs Lebih Lambat")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"±{d} hari" for d in asym.delta_days], y=asym.early_diss_rate*100, name="Datang lebih cepat", marker_color=GREEN))
            fig.add_trace(go.Bar(x=[f"±{d} hari" for d in asym.delta_days], y=asym.late_diss_rate*100, name="Datang lebih lambat", marker_color=RED))
            fig.update_layout(barmode="group", xaxis_title="", yaxis_title="Ketidakpuasan %"); plot(fig)
            read("Bandingkan hijau (lebih cepat dari janji) vs merah (lebih lambat). Terlambat jauh lebih menyakitkan "
                 "daripada senangnya datang lebih awal.")
            take("Terlambat 5 hari → <b>85,8%</b> kecewa; lebih cepat 5 hari → hanya <b>16,6%</b> (rasio 5,2×). "
                 "Pola 'loss aversion' klasik.")
    if lae is not None:
        st.subheader("Sinyal Mana yang Lebih Baik?")
        fig = px.bar(lae, x="feature", y="auc_vs_broad_diss", color="feature",
                     color_discrete_sequence=[GRAY, BLUE], text="auc_vs_broad_diss")
        fig.update_traces(texttemplate="%{text:.3f}"); fig.update_layout(yaxis_range=[.55,.65], showlegend=False, yaxis_title="AUC", xaxis_title=""); plot(fig, 320)
        read("<code>estimation_error</code> (selisih kontinu, hari) sedikit mengungguli <code>is_late</code> "
             "(sekadar ya/tidak terlambat) → <b>seberapa jauh</b> menyimpang lebih informatif daripada sekadar 'telat'.")
    with st.expander("🔍 Detail teknis"):
        st.markdown("- Variabel di sini bersifat **pasca-outcome** → dipakai untuk *analisis*, **bukan** fitur model (mencegah kebocoran).\n"
                    "- **AUC univariat:** kekuatan satu variabel membedakan puas vs kecewa.\n"
                    "- Landasan teori: *Expectation-Disconfirmation Theory* + *loss aversion* (Prospect Theory).")

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "6 · Analisis Error":
    st.title("Di Mana Model Salah?")
    intro("Tidak ada model sempurna. Bagian ini membedah <b>kesalahan</b> model terbaik (LR S2): siapa yang terlewat, "
          "dan kenapa — ini penting untuk memahami batas wajar dari prediksi ini.")
    err = load("09_error_summary.csv")
    if err is not None:
        r = err.iloc[0]; c = st.columns(4)
        kpi(c[0], "Recall", f"{r['recall']*100:.0f}%", "kecewa yang tertangkap")
        kpi(c[1], "Precision", f"{r['precision']*100:.0f}%", "dari yang di-flag, benar kecewa")
        kpi(c[2], "FN bintang 3", f"{r['fn_pct_score3']*100:.0f}%", "yang terlewat, ketidakpuasan ringan")
        kpi(c[3], "FN tepat waktu", f"{r['fn_pct_not_late']*100:.0f}%", "terlewat padahal tidak telat")
        st.subheader("Confusion Matrix (Tebakan vs Kenyataan)")
        cm = [[int(r["TN"]), int(r["FP"])], [int(r["FN"]), int(r["TP"])]]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Prediksi: Puas","Prediksi: Kecewa"], y=["Aktual: Puas","Aktual: Kecewa"])
        fig.update_layout(coloraxis_showscale=False); plot(fig, 380)
        read("Diagonal (kiri-atas & kanan-bawah) = tebakan benar. <b>FN</b> (kiri-bawah) = pelanggan kecewa yang "
             "<b>terlewat</b>; <b>FP</b> (kanan-atas) = puas tapi ter-flag (alarm palsu).")
        take(f"<b>{r['fn_pct_not_late']*100:.0f}% kasus yang terlewat justru tiba tepat waktu</b> → kekecewaannya "
             "kemungkinan dari faktor <b>di luar logistik</b> (kualitas produk) yang memang tak ada di data. "
             "Inilah kenapa AUC ~0,63 adalah plafon yang jujur.")
    fcat = load("09_fn_rate_by_category.csv")
    if fcat is not None and "fn_rate" in fcat.columns:
        st.subheader("Kategori Produk yang Paling Sering Terlewat")
        top = fcat.sort_values("fn_rate", ascending=False).head(12)
        fig = px.bar(top, x="fn_rate", y="top_category", orientation="h", color="fn_rate", color_continuous_scale="Reds")
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="FN rate (makin tinggi = makin sering terlewat)", yaxis_title="", coloraxis_showscale=False); plot(fig, 420)
        read("Kategori di atas = model paling sering gagal mendeteksi ketidakpuasannya → kandidat perbaikan data/fitur.")

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "7 · Segmentasi Risiko":
    st.title("Segmentasi Risiko (K-Means)")
    intro("Selain memprediksi per order, kita coba <b>mengelompokkan</b> order jadi beberapa segmen dengan profil mirip, "
          "untuk memudahkan strategi intervensi per kelompok. Ini analisis pelengkap.")
    cp = load("06_cluster_profile.csv")
    if cp is not None:
        c1, c2 = st.columns([1,1])
        with c1:
            st.subheader("Ketidakpuasan per Segmen")
            fig = px.bar(cp, x=cp.cluster.astype(str), y=cp.broad_diss_rate*100, color="broad_diss_rate",
                         color_continuous_scale="Reds", text=(cp.broad_diss_rate*100).round(1))
            fig.update_layout(xaxis_title="Segmen (cluster)", yaxis_title="Ketidakpuasan %", coloraxis_showscale=False); plot(fig)
        with c2:
            st.subheader("Ukuran tiap Segmen")
            fig = px.pie(cp, names=cp.cluster.astype(str), values="n_orders", hole=.5,
                         color_discrete_sequence=[BLUE, RED, AMBER, GREEN]); plot(fig)
        read("Bandingkan tinggi batang (tingkat risiko) dengan besar potongan pie (ukuran segmen). "
             "Segmen kecil tapi berisiko tinggi = prioritas quality assurance.")
        with st.expander("🔍 Detail profil tiap segmen"):
            st.dataframe(cp, use_container_width=True, hide_index=True)
            st.caption("Kolom avg_* = rata-rata karakteristik order di segmen itu (ongkir, harga, berat, dll).")

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "8 · Risk Scorer (KBS)":
    st.title("Risk Scorer — Alat Skoring Risiko")
    intro("Ini <b>penerjemahan model menjadi alat praktis</b>: geser faktor-faktor sebuah order, lalu sistem "
          "memperkirakan tingkat risikonya (Rendah/Sedang/Tinggi) + rekomendasi tindakan. "
          "Bobotnya mengikuti arah pengaruh dari model Logistic Regression.")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("**Atur ciri-ciri order:**")
        carrier = st.slider("Lama seller serah paket ke kurir (jam)", 0, 240, 48, help="Makin lama → risiko naik. Sinyal terkuat.")
        estday = st.slider("Estimasi hari pengiriman (janji ke pelanggan)", 3, 60, 25, help="Janji makin lama → risiko naik.")
        fratio = st.slider("Rasio ongkir terhadap harga", 0.0, 1.0, 0.25, 0.01, help="Ongkir mahal relatif harga → risiko naik.")
        dist = st.slider("Jarak seller–pelanggan (km)", 0, 3000, 600, 50, help="Makin jauh → risiko naik.")
        approval = st.slider("Lama konfirmasi pembayaran (jam)", 0, 96, 12)
        same = st.checkbox("Seller & pelanggan satu provinsi", value=False, help="Satu provinsi → risiko turun.")
    pts = (min(carrier,240)/240*0.34 + min(estday,60)/60*0.26 + min(fratio,1.0)*0.16
           + min(dist,3000)/3000*0.14 + min(approval,96)/96*0.10 - (0.10 if same else 0))
    score = float(np.clip(pts, 0, 1))
    tier = "TINGGI" if score>=0.55 else ("SEDANG" if score>=0.35 else "RENDAH")
    color = {"TINGGI":RED,"SEDANG":AMBER,"RENDAH":GREEN}[tier]
    action = {"TINGGI":"Prioritas intervensi: percepat pickup kurir, notifikasi proaktif ke pelanggan, cek performa seller.",
              "SEDANG":"Pantau: kirim update status pengiriman, siapkan mitigasi bila proses melambat.",
              "RENDAH":"Tidak perlu tindakan khusus."}[tier]
    with c2:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=score*100,
              gauge={"axis":{"range":[0,100]}, "bar":{"color":color},
                     "steps":[{"range":[0,35],"color":"#dcfce7"},{"range":[35,55],"color":"#fef9c3"},{"range":[55,100],"color":"#fee2e2"}]},
              title={"text":"Skor risiko (0–100)"}))
        fig.update_layout(height=300, margin=dict(l=20,r=20,t=50,b=10)); st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<h3 style='color:{color};margin:0'>Tier: {tier}</h3>", unsafe_allow_html=True)
        st.write(action)
    read("Geser slider di kiri → jarum & tier di kanan berubah. Hijau = risiko rendah, merah = tinggi.")
    st.caption("Catatan: scorer ini rule-based/ilustratif (bobot mengikuti arah koefisien LR). Untuk prediksi "
               "presisi, gunakan model Logistic Regression penuh di notebook.")
