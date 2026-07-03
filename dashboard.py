"""
dashboard.py — Olist Customer Dissatisfaction Risk Modeling
Dashboard Business Intelligence: performa model, peta geografis, simulasi intervensi,
analisis error, segmentasi, dan risk scorer (knowledge-based system).
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

# ── Tema ringkas ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
h1,h2,h3 {font-family: 'Segoe UI', sans-serif;}
.kpi {background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;padding:1rem 1.2rem;}
.kpi .v {font-size:2rem;font-weight:800;color:#0f172a;line-height:1.1;}
.kpi .l {font-size:.8rem;color:#64748b;text-transform:uppercase;letter-spacing:.04em;margin-bottom:.3rem;}
.kpi .d {font-size:.8rem;color:#94a3b8;margin-top:.3rem;}
.note {background:#f1f5f9;border-radius:8px;padding:.7rem 1rem;font-size:.9rem;color:#334155;}
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

# ── Sidebar ─────────────────────────────────────────────────────────────────────
st.sidebar.title("📦 Olist Risk BI")
st.sidebar.caption("Prediksi ketidakpuasan pelanggan sebelum pengiriman")
PAGE = st.sidebar.radio("Navigasi", [
    "Ringkasan", "Performa Model", "Peta & Geografi", "Simulasi Intervensi",
    "Ekspektasi & Logistik", "Analisis Error", "Segmentasi Risiko", "Risk Scorer (KBS)"])
st.sidebar.markdown("---")
st.sidebar.caption("Kelompok 8 · Data Visualization & Business Intelligence")

# ═══════════════════════════════════════════════════════════════════════════════
if PAGE == "Ringkasan":
    st.title("Ringkasan Penelitian")
    st.markdown('<div class="note">Sistem <b>risk scoring ketidakpuasan pelanggan sebelum barang diterima</b> '
                'pada Olist Brazilian E-Commerce. Model terbaik: <b>Logistic Regression (S2 in-fulfillment)</b>.</div>',
                unsafe_allow_html=True)
    st.write("")
    split = load("02_split_summary.csv"); metrics = load("04_classification_metrics.csv")
    topk = load("04_topk_capture.csv")
    n_total = int(split["n_orders"].sum()) if split is not None else 95568
    c = st.columns(4)
    kpi(c[0], "Total order", f"{n_total:,}".replace(",", "."), "delivered + ada review")
    diss = split["broad_positive_rate"].mean() if split is not None else 0.21
    kpi(c[1], "Rata-rata ketidakpuasan", "≈21%", "review ≤ 3")
    best = 0.627
    if metrics is not None:
        try: best = metrics.query("scenario=='S2_in_broad' and model=='LR'")["roc_auc"].iloc[0]
        except Exception: pass
    kpi(c[2], "ROC-AUC model terbaik", f"{best:.3f}", "LR · S2 in-fulfillment")
    lift = 2.0
    if topk is not None:
        try: lift = topk.query("scenario=='S2_in_broad' and model=='LR'")["top_10pct_recall"].iloc[0]/0.10
        except Exception: pass
    kpi(c[3], "Lift (top-10%)", f"{lift:.1f}×", "vs pemilihan acak")

    st.write(""); left, right = st.columns([2, 1])
    monthly = load("01_monthly_distribution.csv")
    if monthly is not None and {"ym", "n_orders"} <= set(monthly.columns):
        with left:
            st.subheader("Volume Order per Bulan")
            fig = px.area(monthly, x="ym", y="n_orders", color_discrete_sequence=[BLUE])
            fig.update_layout(xaxis_title="", yaxis_title="Order"); plot(fig, 360)
    if split is not None:
        with right:
            st.subheader("Distribusi Split")
            fig = px.bar(split, x="split", y="n_orders", color="broad_positive_rate",
                         color_continuous_scale="Reds", text="n_orders")
            fig.update_layout(xaxis_title="", yaxis_title="Order", coloraxis_showscale=False); plot(fig, 360)
        st.markdown('<div class="note">Positive rate <b>validasi lebih tinggi</b> (musim liburan) → '
                    'threshold dipilih dari validasi lalu dikunci sebelum test.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Performa Model":
    st.title("Performa Model Klasifikasi")
    metrics = load("04_classification_metrics.csv")
    if metrics is not None:
        d = metrics[metrics.model != "Dummy"]
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("ROC-AUC per Skenario & Model")
            fig = px.bar(d, x="scenario", y="roc_auc", color="model", barmode="group",
                         color_discrete_sequence=[BLUE, GREEN, AMBER], text="roc_auc")
            fig.update_traces(texttemplate="%{text:.3f}"); fig.update_layout(yaxis_range=[0.55, 0.67]); plot(fig)
        with c2:
            st.subheader("PR-AUC per Skenario & Model")
            fig = px.bar(d, x="scenario", y="pr_auc", color="model", barmode="group",
                         color_discrete_sequence=[BLUE, GREEN, AMBER], text="pr_auc")
            fig.update_traces(texttemplate="%{text:.3f}"); plot(fig)
        st.subheader("Tabel Metrik Lengkap")
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.markdown('<div class="note"><b>LR (S2)</b> adalah model terbaik: AUC tertinggi & interpretable. '
                    'LightGBM tuned hanya setara LR.</div>', unsafe_allow_html=True)
    delong = load("08_delong_results.csv")
    if delong is not None:
        st.subheader("Uji Signifikansi (DeLong)")
        st.dataframe(delong[["comparison", "auc_A", "auc_B", "delta_auc", "p_value", "significance"]],
                     use_container_width=True, hide_index=True)
        st.markdown('<div class="note"><b>RQ2:</b> gain S2 vs S1 <b>signifikan</b> (p<0,001) untuk semua algoritma.</div>',
                    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Peta & Geografi":
    st.title("Peta Distribusi Pengiriman & Risiko Geografis")
    stt = load("05_state_dissatisfaction.csv")
    geo = brazil_geojson()
    if stt is None or "customer_state" not in stt.columns:
        st.warning("Butuh `outputs/tables/05_state_dissatisfaction.csv`. Jalankan script 05 terlebih dahulu.")
    else:
        stt = stt.copy()
        stt["lat"] = stt.customer_state.map(lambda s: COORD.get(s, (None, None))[0])
        stt["lon"] = stt.customer_state.map(lambda s: COORD.get(s, (None, None))[1])
        stt["diss_pct"] = (stt["diss_rate"] * 100).round(1)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Choropleth — Risiko per Negara Bagian")
            if geo is not None:
                try:
                    props = geo["features"][0]["properties"]
                    key = next((k for k, v in props.items() if isinstance(v, str) and len(v) == 2 and v.isupper()), "sigla")
                    fig = px.choropleth(stt, geojson=geo, locations="customer_state",
                                        featureidkey="properties." + key, color="diss_pct",
                                        color_continuous_scale="Reds", scope="south america",
                                        labels={"diss_pct": "Ketidakpuasan %"})
                    fig.update_geos(fitbounds="locations", visible=False); plot(fig, 460)
                except Exception as e:
                    st.info(f"Choropleth tidak tersedia: {e}")
            else:
                st.info("GeoJSON batas negara bagian gagal diambil (perlu internet).")
        with c2:
            st.subheader("Bubble — Volume & Risiko")
            b = stt.dropna(subset=["lat"])
            fig = px.scatter_geo(b, lat="lat", lon="lon", size="n", color="diss_pct",
                                 color_continuous_scale="Reds", scope="south america",
                                 hover_name="customer_state", size_max=40,
                                 labels={"n": "Order", "diss_pct": "Ketidakpuasan %"})
            fig.update_geos(fitbounds="locations", visible=True); plot(fig, 460)

        st.subheader("Alur Pengiriman Seller → Customer (top 40 rute lintas-provinsi)")
        pre = load("order_level_features_pre.csv", "interim")
        if pre is not None and "seller_state" in pre.columns:
            fl = pre.groupby(["seller_state", "customer_state"]).size().reset_index(name="n")
            fl = fl[fl.seller_state != fl.customer_state].sort_values("n", ascending=False).head(40)
            mx = fl.n.max()
            fig = go.Figure()
            for _, r in fl.iterrows():
                a, b2 = COORD.get(r.seller_state), COORD.get(r.customer_state)
                if a and b2:
                    fig.add_trace(go.Scattergeo(lon=[a[1], b2[1]], lat=[a[0], b2[0]], mode="lines",
                                  line=dict(width=max(0.5, r.n / mx * 7), color="rgba(37,99,235,0.45)"), showlegend=False))
            fig.add_trace(go.Scattergeo(lon=stt.lon, lat=stt.lat, mode="markers",
                          marker=dict(size=5, color=RED), text=stt.customer_state, showlegend=False))
            fig.update_geos(scope="south america", visible=True, showcountries=True); plot(fig, 460)
        st.markdown('<div class="note">Ketidakpuasan terkonsentrasi di <b>Utara & Timur Laut</b> (MA 29,9%, AL 28,5%) — '
                    'jauh dari <b>hub seller São Paulo</b> (18,6%). Implikasi: prioritas last-mile rute SP → Utara.</div>',
                    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Simulasi Intervensi":
    st.title("Simulasi Intervensi Operasional (RQ4)")
    topk = load("04_topk_capture.csv")
    if topk is not None:
        s2 = topk[(topk.scenario == "S2_in_broad") & (topk.model.isin(["LR", "RF", "LightGBM"]))]
        cols = [c for c in ["top_5pct_recall", "top_10pct_recall", "top_20pct_recall", "top_30pct_recall"] if c in topk.columns]
        pct = [5, 10, 20, 30]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0] + pct + [100], y=[0] + [0, 0, 0, 0] + [100], visible=False))
        for _, r in s2.iterrows():
            ys = [0] + [r[c] * 100 for c in cols] + [100]
            fig.add_trace(go.Scatter(x=[0] + pct + [100], y=ys, mode="lines+markers", name=r.model))
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Acak",
                                 line=dict(dash="dash", color=GRAY)))
        fig.update_layout(xaxis_title="% order di-flag (prioritas)", yaxis_title="% pelanggan kecewa tertangkap")
        st.subheader("Cumulative Gain — S2"); plot(fig)
        try:
            lr = s2[s2.model == "LR"].iloc[0]
            c = st.columns(3)
            kpi(c[0], "Top-10% capture", f"{lr['top_10pct_recall']*100:.1f}%", "LR S2")
            kpi(c[1], "Lift", f"{lr['top_10pct_recall']/0.10:.2f}×", "vs acak")
            kpi(c[2], "Top-20% capture", f"{lr['top_20pct_recall']*100:.1f}%", "LR S2")
        except Exception: pass
        st.markdown('<div class="note">Menandai <b>10%</b> order paling berisiko menangkap <b>~20%</b> pelanggan '
                    'kecewa (lift ~2×) — tim operasional cukup fokus ke sebagian kecil order.</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Ekspektasi & Logistik":
    st.title("Pelanggaran Ekspektasi & Rantai Pengiriman (RQ3)")
    ebin = load("05_estimation_error_bins.csv"); asym = load("05_asymmetry_check.csv")
    lae = load("05_late_vs_error_auc.csv"); chain = load("05_delivery_chain_quartiles.csv")
    c1, c2 = st.columns(2)
    if ebin is not None and "diss_pct" in ebin.columns:
        with c1:
            st.subheader("Ketidakpuasan vs Selisih Estimasi")
            fig = px.bar(ebin, x="err_bin", y="diss_pct", color="diss_pct", color_continuous_scale="RdYlGn_r")
            fig.update_layout(xaxis_title="Selisih estimasi (hari)", yaxis_title="Ketidakpuasan %", coloraxis_showscale=False); plot(fig)
    if asym is not None:
        with c2:
            st.subheader("Asimetri: Lebih Cepat vs Lebih Lambat")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"±{d}" for d in asym.delta_days], y=asym.early_diss_rate*100, name="Lebih cepat", marker_color=GREEN))
            fig.add_trace(go.Bar(x=[f"±{d}" for d in asym.delta_days], y=asym.late_diss_rate*100, name="Lebih lambat", marker_color=RED))
            fig.update_layout(barmode="group", xaxis_title="Selisih (hari)", yaxis_title="Ketidakpuasan %"); plot(fig)
            st.markdown('<div class="note">±5 hari: <b>85,8%</b> vs <b>16,6%</b> → rasio <b>5,2×</b> (loss aversion).</div>', unsafe_allow_html=True)
    if lae is not None:
        st.subheader("Sinyal Kontinu vs Flag Biner")
        fig = px.bar(lae, x="feature", y="auc_vs_broad_diss", color="feature",
                     color_discrete_sequence=[GRAY, BLUE], text="auc_vs_broad_diss")
        fig.update_traces(texttemplate="%{text:.3f}"); fig.update_layout(yaxis_range=[0.55, 0.65], showlegend=False, yaxis_title="AUC"); plot(fig, 340)
    if chain is not None:
        st.subheader("Dekomposisi Rantai Pengiriman (per kuartil durasi)")
        fig = px.bar(chain, x="quartile", y="diss_rate", color="phase", barmode="group",
                     color_discrete_sequence=[AMBER, RED])
        fig.update_layout(yaxis_title="Ketidakpuasan (proporsi)"); plot(fig, 340)

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Analisis Error":
    st.title("Analisis Error — Model Terbaik (LR S2)")
    err = load("09_error_summary.csv")
    if err is not None:
        r = err.iloc[0]
        c = st.columns(4)
        kpi(c[0], "Recall", f"{r['recall']*100:.0f}%", "kelas kecewa")
        kpi(c[1], "Precision", f"{r['precision']*100:.0f}%", "")
        kpi(c[2], "FN score-3", f"{r['fn_pct_score3']*100:.0f}%", "ketidakpuasan ringan")
        kpi(c[3], "FN tepat waktu", f"{r['fn_pct_not_late']*100:.0f}%", "bukan karena telat")
        st.subheader("Confusion Matrix")
        cm = [[int(r["TN"]), int(r["FP"])], [int(r["FN"]), int(r["TP"])]]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Prediksi: Puas", "Prediksi: Kecewa"], y=["Aktual: Puas", "Aktual: Kecewa"])
        fig.update_layout(coloraxis_showscale=False); plot(fig, 380)
        st.markdown(f'<div class="note"><b>{r["fn_pct_not_late"]*100:.0f}%</b> kasus yang terlewat (FN) justru tiba '
                    'tepat waktu → sebagian ketidakpuasan berasal dari faktor pasca-pengiriman (kualitas produk) '
                    'yang tak ada di fitur pra-outcome.</div>', unsafe_allow_html=True)
    fcat = load("09_fn_rate_by_category.csv")
    if fcat is not None and "fn_rate" in fcat.columns:
        st.subheader("FN Rate per Kategori Produk")
        top = fcat.sort_values("fn_rate", ascending=False).head(12)
        fig = px.bar(top, x="fn_rate", y="top_category", orientation="h", color="fn_rate",
                     color_continuous_scale="Reds")
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="FN rate", coloraxis_showscale=False); plot(fig, 420)

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Segmentasi Risiko":
    st.title("Segmentasi Risiko — K-Means")
    cp = load("06_cluster_profile.csv")
    if cp is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Ketidakpuasan per Cluster")
            fig = px.bar(cp, x=cp.cluster.astype(str), y=cp.broad_diss_rate*100, color="broad_diss_rate",
                         color_continuous_scale="Reds", text=(cp.broad_diss_rate*100).round(1))
            fig.update_layout(xaxis_title="Cluster", yaxis_title="Ketidakpuasan %", coloraxis_showscale=False); plot(fig)
        with c2:
            st.subheader("Ukuran Segmen")
            fig = px.pie(cp, names=cp.cluster.astype(str), values="n_orders", hole=0.5,
                         color_discrete_sequence=[BLUE, RED, AMBER, GREEN])
            plot(fig)
        st.subheader("Profil Cluster")
        st.dataframe(cp, use_container_width=True, hide_index=True)
    elb = load("06_elbow_silhouette.csv")
    if elb is not None:
        st.subheader("Pemilihan K (Elbow & Silhouette)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=elb.K, y=elb.silhouette, mode="lines+markers", name="Silhouette", line=dict(color=RED)))
        fig.update_layout(xaxis_title="K", yaxis_title="Silhouette"); plot(fig, 320)

# ═══════════════════════════════════════════════════════════════════════════════
elif PAGE == "Risk Scorer (KBS)":
    st.title("Risk Scorer — Knowledge-Based System")
    st.markdown('<div class="note">Scorecard berbasis aturan dari <b>arah koefisien Logistic Regression</b>. '
                'Geser faktor risiko untuk mengestimasi tier risiko sebuah order.</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        carrier = st.slider("Carrier pickup lag (jam) — seller serah ke kurir", 0, 240, 48)
        estday = st.slider("Estimasi hari pengiriman (janji)", 3, 60, 25)
        fratio = st.slider("Freight-to-price ratio (ongkir/harga)", 0.0, 1.0, 0.25, 0.01)
        dist = st.slider("Jarak seller–pelanggan (km)", 0, 3000, 600, 50)
        approval = st.slider("Approval lag (jam) — konfirmasi bayar", 0, 96, 12)
        same = st.checkbox("Satu provinsi (same-state)", value=False)
    # scorecard: bobot ilustratif dari arah koefisien LR (dinormalisasi 0..1)
    pts = (min(carrier,240)/240*0.34 + min(estday,60)/60*0.26 + min(fratio,1.0)*0.16
           + min(dist,3000)/3000*0.14 + min(approval,96)/96*0.10 - (0.10 if same else 0))
    score = float(np.clip(pts, 0, 1))
    tier = "TINGGI" if score >= 0.55 else ("SEDANG" if score >= 0.35 else "RENDAH")
    color = {"TINGGI": RED, "SEDANG": AMBER, "RENDAH": GREEN}[tier]
    action = {"TINGGI": "Prioritas intervensi: percepat pickup kurir, notifikasi proaktif, cek performa seller.",
              "SEDANG": "Pantau: kirim update status pengiriman, siapkan mitigasi bila proses melambat.",
              "RENDAH": "Tidak perlu tindakan khusus."}[tier]
    with c2:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=score*100,
              gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                     "steps": [{"range": [0, 35], "color": "#dcfce7"}, {"range": [35, 55], "color": "#fef9c3"},
                               {"range": [55, 100], "color": "#fee2e2"}]},
              title={"text": "Skor risiko"}))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10)); st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<h3 style='color:{color};margin:0'>Tier: {tier}</h3>", unsafe_allow_html=True)
        st.write(action)
    st.caption("Catatan: scorecard bersifat ilustratif (rule-based). Prediksi presisi memakai model Logistic Regression penuh pada notebook.")
