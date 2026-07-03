"""
dashboard.py — Olist Customer Dissatisfaction Risk Modeling
Business Intelligence dashboard: tampilan premium, 5 seksi, konten lengkap.
Jalankan: streamlit run dashboard.py
"""
import json, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Olist Risk Intelligence — BI Dashboard",
                   page_icon="📦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [class*="css"] { font-family:'Inter',system-ui,-apple-system,sans-serif; }
.stApp { background:#f5f6f8; }
.block-container { padding-top:1.4rem; max-width:1320px; }
.hero { background:linear-gradient(120deg,#0f172a 0%,#1e293b 50%,#1d4ed8 100%); border-radius:16px; padding:1.5rem 1.8rem; margin-bottom:1.3rem; }
.hero h1 { font-size:1.55rem; font-weight:800; color:#fff; margin:0; letter-spacing:-.01em; }
.hero p { font-size:.95rem; color:#c7d2e5; margin:.4rem 0 0; }
.hero .chips { margin-top:.95rem; display:flex; gap:.5rem; flex-wrap:wrap; }
.hero .chip { background:rgba(255,255,255,.13); color:#e5edf7; font-size:.74rem; font-weight:500; padding:.3rem .75rem; border-radius:999px; }
.kpi { background:#fff; border:1px solid #e7eaef; border-top:3px solid #2563eb; border-radius:14px; padding:1.05rem 1.2rem; height:100%; box-shadow:0 1px 3px rgba(15,23,42,.05); }
.kpi .l { font-size:.7rem; color:#64748b; text-transform:uppercase; letter-spacing:.05em; font-weight:700; margin-bottom:.45rem; }
.kpi .v { font-size:1.95rem; font-weight:800; color:#0f172a; line-height:1; }
.kpi .d { font-size:.77rem; color:#94a3b8; margin-top:.5rem; }
.intro { background:#eff6ff; border-left:4px solid #2563eb; border-radius:0 10px 10px 0; padding:.85rem 1.1rem; font-size:.92rem; color:#1e3a5f; margin:.2rem 0 1rem; line-height:1.6; }
.read { background:#fff; border:1px solid #e5e9f0; border-radius:10px; padding:.6rem .95rem; font-size:.85rem; color:#475569; margin:.55rem 0; }
.take { background:#052e1a; border-radius:10px; padding:.8rem 1.05rem; font-size:.88rem; color:#a7f3d0; margin:.65rem 0; line-height:1.55; }
.sec { font-size:.72rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; color:#2563eb; border-bottom:2px solid #e5e9f0; padding-bottom:.35rem; margin:1.7rem 0 .9rem; }
.brand { background:linear-gradient(120deg,#1e293b,#1d4ed8); border-radius:12px; padding:.9rem 1rem; margin-bottom:.6rem; }
.brand .t { color:#fff; font-weight:800; font-size:1.02rem; }
.brand .s { color:#c7d2e5; font-size:.74rem; margin-top:.15rem; }
</style>""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"
INTERIM = ROOT / "data_interim"
BLUE, SLATE, LIGHT, RED, GREEN, AMBER, DARK = "#2563eb", "#64748b", "#cbd5e1", "#dc2626", "#16a34a", "#f59e0b", "#0f172a"
MODEL_COLORS = {"LR": BLUE, "RF": SLATE, "LightGBM": LIGHT}
SCEN_LABEL = {"S1_pre_broad": "S1 · pra-kirim", "S2_in_broad": "S2 · in-fulfillment", "S3_in_severe": "S3 · severe"}

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

def style(fig, h=420):
    fig.update_layout(margin=dict(l=10, r=10, t=34, b=10), height=h,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", size=13, color="#334155"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=""),
                      xaxis=dict(gridcolor="#eef1f5", zeroline=False),
                      yaxis=dict(gridcolor="#eef1f5", zeroline=False))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def kpi(col, label, value, desc=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="d">{desc}</div></div>', unsafe_allow_html=True)

def hero(title, subtitle, chips=None):
    ch = "".join(f'<span class="chip">{c}</span>' for c in (chips or []))
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p><div class="chips">{ch}</div></div>',
                unsafe_allow_html=True)

def intro(txt): st.markdown(f'<div class="intro">{txt}</div>', unsafe_allow_html=True)
def read(txt):  st.markdown(f'<div class="read">📊 <b>Cara baca:</b> {txt}</div>', unsafe_allow_html=True)
def take(txt):  st.markdown(f'<div class="take">💡 <b>Intinya:</b> {txt}</div>', unsafe_allow_html=True)
def sec(txt):   st.markdown(f'<div class="sec">{txt}</div>', unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div class="brand"><div class="t">📦 Olist Risk BI</div>'
                    '<div class="s">Ketidakpuasan pelanggan · prediksi dini</div></div>', unsafe_allow_html=True)
PAGE = st.sidebar.radio("Navigasi", ["Ringkasan", "Performa & Dampak", "Geografi & Logistik",
                                     "Segmentasi Risiko", "Risk Scorer"])
with st.sidebar.expander("ℹ️ Tentang & istilah"):
    st.markdown("- **Ketidakpuasan** = review ≤ 3 bintang (target `broad`).\n"
                "- **S1 · pra-kirim** = fitur yang diketahui saat order dibuat.\n"
                "- **S2 · in-fulfillment** = S1 + sinyal proses pengiriman (model utama).\n"
                "- **S3 · severe** = target lebih ketat (review ≤ 2).\n"
                "- **ROC-AUC** = ketepatan model membedakan (0,5 = asal tebak, 1 = sempurna).\n"
                "- **Model terbaik:** Logistic Regression (S2), ROC-AUC 0,627.")
st.sidebar.caption("Kelompok 8 · Data Visualization & Business Intelligence")

# ═════════════════════════════ 1 · RINGKASAN ═════════════════════════════════════
if PAGE == "Ringkasan":
    split = load("02_split_summary.csv"); metrics = load("04_classification_metrics.csv"); topk = load("04_topk_capture.csv")
    n_total = int(split["n_orders"].sum()) if split is not None else 95568
    best = 0.627
    if metrics is not None:
        try: best = float(metrics.query("scenario=='S2_in_broad' and model=='LR'")["roc_auc"].iloc[0])
        except Exception: pass
    lift = 2.0
    if topk is not None:
        try: lift = float(topk.query("scenario=='S2_in_broad' and model=='LR'")["top_10pct_recall"].iloc[0]) / 0.10
        except Exception: pass
    hero("Ringkasan", "Seberapa besar masalah ketidakpuasan, dan seberapa baik model menangkapnya.",
         [f"{n_total:,}".replace(",", ".") + " order", "Logistic Regression", f"ROC-AUC {best:.3f}".replace(".", ",")])
    intro("Empat angka ini adalah inti dari seluruh penelitian: skala masalah, ketepatan model, dan seberapa "
          "efisien model memprioritaskan order berisiko.")
    c = st.columns(4)
    kpi(c[0], "Order dianalisis", f"{n_total:,}".replace(",", "."), "delivered + ada review")
    kpi(c[1], "Tingkat ketidakpuasan", "≈ 21%", "1 dari 5 order · review ≤ 3")
    kpi(c[2], "Ketepatan model terbaik", f"{best:.3f}".replace(".", ","), "ROC-AUC · LR in-fulfillment")
    kpi(c[3], "Efektivitas prioritas", f"{lift:.1f}×".replace(".", ","), "tangkap 20% di 10% teratas")
    read("<b>ROC-AUC</b> mendekati 1 makin baik; 0,5 = menebak koin. Nilai ~0,63 tergolong moderat — "
         "seksi <i>Performa & Dampak</i> menjelaskan mengapa lebih tinggi dari itu sulit dicapai secara jujur.")
    left, right = st.columns([2, 1])
    monthly = load("01_monthly_distribution.csv")
    if monthly is not None and {"ym", "n_orders"} <= set(monthly.columns):
        with left:
            sec("Volume order per bulan")
            fig = px.area(monthly, x="ym", y="n_orders", color_discrete_sequence=[BLUE])
            fig.update_traces(line=dict(width=2), fillcolor="rgba(37,99,235,0.12)")
            fig.update_layout(xaxis_title="", yaxis_title="Jumlah order"); style(fig, 330)
            read("Pertumbuhan transaksi Olist; lonjakan akhir 2017 = musim liburan.")
    if split is not None:
        with right:
            sec("Pembagian data")
            sp = split.copy(); sp["pct"] = (sp["broad_positive_rate"] * 100).round(1)
            fig = px.bar(sp, x="split", y="n_orders", color="pct", color_continuous_scale="Blues", text="n_orders")
            fig.update_layout(xaxis_title="", yaxis_title="", coloraxis_showscale=False); style(fig, 330)
            read("Split <b>temporal</b> (bukan acak) agar meniru kondisi nyata & cegah kebocoran.")
    with st.expander("🔍 Detail teknis — definisi target & pembagian data"):
        st.markdown("- **Target (broad):** `review_score ≤ 3` → 1. Prevalensi ~21%.\n"
                    "- **Split temporal:** Train 2017-01..11 · Validasi 2017-12..2018-02 · Test 2018-03..08.\n"
                    "- Validasi memuat musim liburan → tingkat ketidakpuasan lebih tinggi (~24,9%).")
        if split is not None: st.dataframe(split, use_container_width=True, hide_index=True)

# ═════════════════════════ 2 · PERFORMA & DAMPAK ════════════════════════════════
elif PAGE == "Performa & Dampak":
    hero("Performa & Dampak Model", "Model mana yang terbaik, seberapa nyata bedanya, dan seberapa berguna untuk prioritas.",
         ["3 algoritma", "2 skenario fitur", "uji DeLong"])

    metrics = load("04_classification_metrics.csv")
    sec("Perbandingan model")
    intro("Tiga algoritma (Logistic Regression, Random Forest, LightGBM) diuji pada skenario fitur "
          "<b>S1</b> (pra-kirim), <b>S2</b> (in-fulfillment), dan <b>S3</b> (target severe). "
          "<b>LR disorot biru</b>; pesaing sengaja diabukan agar mudah dibandingkan.")
    if metrics is not None:
        m = metrics[metrics.model != "Dummy"].copy()
        m["Skenario"] = m.scenario.map(SCEN_LABEL).fillna(m.scenario)
        cat = {"model": ["LR", "RF", "LightGBM"], "Skenario": list(SCEN_LABEL.values())}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**ROC-AUC** — ketepatan membedakan")
            fig = px.bar(m, x="Skenario", y="roc_auc", color="model", barmode="group",
                         color_discrete_map=MODEL_COLORS, category_orders=cat, text="roc_auc")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
            fig.update_layout(yaxis_range=[.55, .68], yaxis_title="ROC-AUC", xaxis_title=""); style(fig, 380)
        with c2:
            st.markdown("**PR-AUC** — fokus kelas minoritas")
            fig = px.bar(m, x="Skenario", y="pr_auc", color="model", barmode="group",
                         color_discrete_map=MODEL_COLORS, category_orders=cat, text="pr_auc")
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
            fig.update_layout(yaxis_title="PR-AUC", xaxis_title=""); style(fig, 380)
        read("Batang lebih tinggi = lebih baik. <b>S2 &gt; S1</b> untuk semua algoritma → sinyal proses "
             "pengiriman benar-benar menambah nilai. Di S2, <b>LR (biru) unggul</b>.")
        take("Model terbaik = <b>Logistic Regression (S2)</b>, ROC-AUC 0,627 — mengungguli LightGBM dan "
             "setara RF, tapi jauh lebih mudah dijelaskan (koefisien langsung dibaca). Cocok jadi dasar Risk Scorer.")
        with st.expander("🔍 Tabel metrik lengkap (semua skenario & model)"):
            show = metrics.copy(); show["scenario"] = show.scenario.map(SCEN_LABEL).fillna(show.scenario)
            st.dataframe(show.rename(columns={"scenario": "Skenario", "model": "Model", "roc_auc": "ROC-AUC",
                         "pr_auc": "PR-AUC", "recall": "Recall", "precision": "Precision", "f1": "F1",
                         "brier_score": "Brier", "threshold": "Threshold"}), use_container_width=True, hide_index=True)

    delong = load("08_delong_results.csv")
    if delong is not None:
        sec("Signifikansi statistik — uji DeLong")
        intro("Uji DeLong memastikan selisih AUC itu <b>nyata</b> (signifikan), bukan kebetulan angka. "
              "Kolom <code>p_value</code> &lt; 0,05 = nyata; <code>***</code> = sangat nyata.")
        st.dataframe(delong[["comparison", "auc_A", "auc_B", "delta_auc", "p_value", "significance"]]
                     .rename(columns={"comparison": "Perbandingan", "delta_auc": "Δ AUC", "significance": "Sig."}),
                     use_container_width=True, hide_index=True)
        read("Semua 'S1 vs S2' <b>p&lt;0,001</b> → peningkatan in-fulfillment terbukti nyata. "
             "'LR vs RF' <b>ns</b> (setara), 'LR vs LightGBM' <b>***</b> (LR menang).")

    topk = load("04_topk_capture.csv")
    if topk is not None:
        sec("Simulasi prioritas intervensi")
        intro("Jika tim hanya bisa menindak <b>sebagian</b> order: urutkan dari paling berisiko, tangani X% teratas — "
              "berapa % calon pelanggan kecewa yang tercegat? (kurva <i>cumulative gain</i>).")
        s2 = topk[(topk.scenario == "S2_in_broad") & (topk.model.isin(["LR", "RF", "LightGBM"]))]
        cols = [c for c in ["top_5pct_recall", "top_10pct_recall", "top_20pct_recall", "top_30pct_recall"] if c in topk.columns]
        pct = [5, 10, 20, 30]
        lw = {"LR": 3, "RF": 1.6, "LightGBM": 1.6}
        fig = go.Figure()
        for _, r in s2.iterrows():
            ys = [0] + [r[c] * 100 for c in cols] + [100]
            fig.add_trace(go.Scatter(x=[0] + pct + [100], y=ys, mode="lines+markers" if r.model == "LR" else "lines",
                          name=r.model, line=dict(color=MODEL_COLORS.get(r.model, SLATE), width=lw.get(r.model, 1.6))))
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", name="Acak", line=dict(dash="dash", color="#cbd5e1")))
        try:
            lr = s2[s2.model == "LR"].iloc[0]
            fig.add_annotation(x=10, y=lr["top_10pct_recall"] * 100, text="10% → {:.0f}%".format(lr["top_10pct_recall"] * 100),
                               showarrow=True, arrowhead=2, ax=40, ay=-30, font=dict(color=BLUE, size=12))
        except Exception: pass
        fig.update_layout(xaxis_title="% order berisiko ditandai", yaxis_title="% pelanggan kecewa tertangkap")
        style(fig, 400)
        read("Garis makin jauh <b>di atas</b> garis putus-putus 'Acak' = model makin berguna untuk memprioritaskan.")
        try:
            lr = s2[s2.model == "LR"].iloc[0]; c = st.columns(3)
            kpi(c[0], "Tandai 10% teratas", f"{lr['top_10pct_recall']*100:.1f}%".replace(".", ","), "pelanggan kecewa tertangkap")
            kpi(c[1], "Lift", f"{lr['top_10pct_recall']/0.10:.2f}×".replace(".", ","), "vs menandai acak")
            kpi(c[2], "Tandai 20% teratas", f"{lr['top_20pct_recall']*100:.1f}%".replace(".", ","), "pelanggan kecewa tertangkap")
        except Exception: pass
        take("Cukup menandai <b>10% order paling berisiko</b>, tim menangkap <b>~20%</b> calon pelanggan kecewa — "
             "dua kali lebih efisien daripada acak.")

    err = load("09_error_summary.csv")
    if err is not None:
        sec("Analisis error model terbaik (LR S2)")
        intro("Tak ada model sempurna. Bagian ini membedah <b>kesalahannya</b>: siapa yang terlewat, dan kenapa — "
              "kunci untuk memahami batas wajar prediksi.")
        r = err.iloc[0]; c = st.columns(4)
        kpi(c[0], "Recall", f"{r['recall']*100:.0f}%", "kecewa yang tertangkap")
        kpi(c[1], "Precision", f"{r['precision']*100:.0f}%", "dari yang di-flag, benar kecewa")
        kpi(c[2], "FN bintang 3", f"{r['fn_pct_score3']*100:.0f}%", "yang terlewat = kecewa ringan")
        kpi(c[3], "FN tepat waktu", f"{r['fn_pct_not_late']*100:.0f}%", "terlewat padahal tak telat")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Confusion matrix** — tebakan vs kenyataan")
            cm = [[int(r["TN"]), int(r["FP"])], [int(r["FN"]), int(r["TP"])]]
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            x=["Prediksi: Puas", "Prediksi: Kecewa"], y=["Aktual: Puas", "Aktual: Kecewa"])
            fig.update_layout(coloraxis_showscale=False); style(fig, 360)
            read("Diagonal = tebakan benar. <b>FN</b> (kiri-bawah) = kecewa yang <b>terlewat</b>; "
                 "<b>FP</b> (kanan-atas) = alarm palsu.")
        with c2:
            fcat = load("09_fn_rate_by_category.csv")
            if fcat is not None and "fn_rate" in fcat.columns:
                st.markdown("**Kategori paling sering terlewat** (FN rate)")
                top = fcat.sort_values("fn_rate", ascending=False).head(12)
                fig = px.bar(top, x="fn_rate", y="top_category", orientation="h",
                             color="fn_rate", color_continuous_scale="Reds")
                fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="FN rate", yaxis_title="",
                                  coloraxis_showscale=False); style(fig, 360)
                read("Kategori di atas = paling sering gagal terdeteksi → kandidat perbaikan data/fitur.")
        take(f"<b>{r['fn_pct_not_late']*100:.0f}% kasus terlewat justru tiba tepat waktu</b> → kekecewaannya "
             "kemungkinan dari faktor <b>di luar logistik</b> (kualitas produk) yang tak ada di data. "
             "Itulah kenapa ROC-AUC ~0,63 adalah plafon yang jujur.")

# ═══════════════════════ 3 · GEOGRAFI & LOGISTIK ════════════════════════════════
elif PAGE == "Geografi & Logistik":
    hero("Geografi & Logistik", "Di mana risiko ketidakpuasan tertinggi, dan kenapa pelanggan kecewa.",
         ["27 negara bagian", "hub São Paulo", "ekspektasi vs kenyataan"])
    stt = load("05_state_dissatisfaction.csv"); geo = brazil_geojson()

    sec("Peta risiko & distribusi pengiriman")
    intro("Ketidakpuasan tidak merata secara geografis. Peta membantu <b>intervensi berbasis wilayah</b> "
          "(mis. memperkuat last-mile ke daerah tertentu).")
    if stt is None or "customer_state" not in stt.columns:
        st.warning("Butuh `outputs/tables/05_state_dissatisfaction.csv`.")
    else:
        stt = stt.copy()
        stt["lat"] = stt.customer_state.map(lambda s: COORD.get(s, (None, None))[0])
        stt["lon"] = stt.customer_state.map(lambda s: COORD.get(s, (None, None))[1])
        stt["diss_pct"] = (stt["diss_rate"] * 100).round(1)
        stt["late_pct"] = (stt["late_rate"] * 100).round(1)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Choropleth** — risiko per negara bagian")
            if geo is not None:
                try:
                    props = geo["features"][0]["properties"]
                    key = next((k for k, v in props.items() if isinstance(v, str) and len(v) == 2 and v.isupper()), "sigla")
                    fig = px.choropleth(stt, geojson=geo, locations="customer_state", featureidkey="properties." + key,
                                        color="diss_pct", color_continuous_scale="Reds", scope="south america",
                                        labels={"diss_pct": "Ketidakpuasan %"})
                    fig.update_geos(fitbounds="locations", visible=False); style(fig, 450)
                except Exception as e:
                    st.info(f"Choropleth tidak tersedia: {e}")
            else:
                st.info("Peta batas negara bagian perlu internet (gagal diambil).")
            read("Makin <b>merah</b> = ketidakpuasan makin tinggi.")
        with c2:
            st.markdown("**Bubble** — volume vs risiko")
            b = stt.dropna(subset=["lat"])
            fig = px.scatter_geo(b, lat="lat", lon="lon", size="n", color="diss_pct",
                                 color_continuous_scale="Reds", scope="south america",
                                 hover_name="customer_state", size_max=40,
                                 labels={"n": "Jumlah order", "diss_pct": "Ketidakpuasan %"})
            fig.update_geos(fitbounds="locations", visible=True); style(fig, 450)
            read("<b>Ukuran</b> = jumlah order; <b>warna</b> = risiko. São Paulo besar & terang = volume besar, risiko rendah.")
        st.markdown("**Alur pengiriman seller → customer** (top 40 rute)")
        pre = load("order_level_features_pre.csv", "interim")
        if pre is not None and "seller_state" in pre.columns:
            fl = pre.groupby(["seller_state", "customer_state"]).size().reset_index(name="n")
            fl = fl[fl.seller_state != fl.customer_state].sort_values("n", ascending=False).head(40)
            mx = fl.n.max(); fig = go.Figure()
            for _, r in fl.iterrows():
                a, d2 = COORD.get(r.seller_state), COORD.get(r.customer_state)
                if a and d2:
                    fig.add_trace(go.Scattergeo(lon=[a[1], d2[1]], lat=[a[0], d2[0]], mode="lines",
                                  line=dict(width=max(.5, r.n / mx * 7), color="rgba(37,99,235,0.45)"), showlegend=False))
            fig.add_trace(go.Scattergeo(lon=stt.lon, lat=stt.lat, mode="markers",
                          marker=dict(size=5, color=RED), text=stt.customer_state, showlegend=False))
            fig.update_geos(scope="south america", visible=True, showcountries=True); style(fig, 450)
            read("Tiap garis = rute antar provinsi; makin <b>tebal</b> = makin ramai. São Paulo = hub yang memancar ke seluruh Brasil.")
        else:
            st.info("Peta alur perlu `data_interim/order_level_features_pre.csv` (tersedia saat dijalankan lokal).")
        take("Ketidakpuasan terpusat di <b>Utara & Timur Laut</b> (Maranhão 29,9%) — jauh dari hub São Paulo (18,6%). "
             "Makin jauh dari hub → makin sering telat → makin tinggi risiko.")
        with st.expander("🔍 Tabel per negara bagian (volume, ketidakpuasan, keterlambatan)"):
            tb = stt[["customer_state", "n", "diss_pct", "late_pct"]].sort_values("diss_pct", ascending=False)
            st.dataframe(tb.rename(columns={"customer_state": "Provinsi", "n": "Order",
                         "diss_pct": "Ketidakpuasan %", "late_pct": "Telat %"}),
                         use_container_width=True, hide_index=True)

    sec("Ekspektasi vs kenyataan pengiriman")
    intro("Ketidakpuasan sering bukan soal 'telat atau tidak', melainkan <b>selisih antara janji dan kenyataan</b>.")
    ebin = load("05_estimation_error_bins.csv"); asym = load("05_asymmetry_check.csv"); lae = load("05_late_vs_error_auc.csv")
    c1, c2 = st.columns(2)
    if ebin is not None and "diss_pct" in ebin.columns:
        with c1:
            st.markdown("**Ketidakpuasan vs selisih estimasi**")
            fig = px.bar(ebin, x="err_bin", y="diss_pct", color="diss_pct", color_continuous_scale="RdYlGn_r")
            fig.update_layout(xaxis_title="Selisih hari (− lebih cepat · + lebih lambat)", yaxis_title="Ketidakpuasan %",
                              coloraxis_showscale=False); style(fig, 360)
            read("Ke kanan = makin melewati tanggal janji → ketidakpuasan melonjak tajam.")
    if asym is not None:
        with c2:
            st.markdown("**Asimetri: lebih cepat vs lebih lambat**")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"±{d} hari" for d in asym.delta_days], y=asym.early_diss_rate * 100,
                                 name="Lebih cepat", marker_color=GREEN))
            fig.add_trace(go.Bar(x=[f"±{d} hari" for d in asym.delta_days], y=asym.late_diss_rate * 100,
                                 name="Lebih lambat", marker_color=RED))
            fig.update_layout(barmode="group", xaxis_title="", yaxis_title="Ketidakpuasan %"); style(fig, 360)
            read("Hijau (lebih cepat) vs merah (lebih lambat). Terlambat jauh lebih menyakitkan daripada senangnya datang awal.")
    if lae is not None:
        st.markdown("**Sinyal mana yang lebih informatif?**")
        fig = px.bar(lae, x="feature", y="auc_vs_broad_diss", color="feature",
                     color_discrete_map={"is_late": SLATE, "estimation_error": BLUE}, text="auc_vs_broad_diss")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_range=[.55, .65], showlegend=False, yaxis_title="AUC univariat", xaxis_title=""); style(fig, 320)
        read("<code>estimation_error</code> (selisih kontinu, hari) mengungguli <code>is_late</code> (ya/tidak) → "
             "<b>seberapa jauh</b> menyimpang lebih penting daripada sekadar 'telat'.")
    take("Terlambat 5 hari → <b>85,8%</b> kecewa; lebih cepat 5 hari → hanya <b>16,6%</b> (rasio 5,2×). "
         "Pola <i>loss aversion</i> klasik — landasan Expectation-Disconfirmation Theory.")
    with st.expander("🔍 Detail teknis — variabel ekspektasi"):
        st.markdown("- Variabel di sini bersifat **pasca-outcome** → hanya untuk *analisis explanatory*, **bukan** fitur model.\n"
                    "- **AUC univariat:** kekuatan satu variabel membedakan puas vs kecewa.\n"
                    "- Teori: *Expectation-Disconfirmation Theory* + *loss aversion* (Prospect Theory).")

# ══════════════════════════ 4 · SEGMENTASI ══════════════════════════════════════
elif PAGE == "Segmentasi Risiko":
    hero("Segmentasi Risiko (K-Means)", "Mengelompokkan order jadi segmen berprofil mirip untuk strategi intervensi.",
         ["analisis pelengkap", "K-Means", "RobustScaler"])
    intro("Selain memprediksi per order, order dikelompokkan jadi beberapa <b>segmen</b> berkarakter mirip agar "
          "intervensi bisa dirancang per kelompok.")
    cp = load("06_cluster_profile.csv")
    if cp is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            sec("Ketidakpuasan per segmen")
            fig = px.bar(cp, x=cp.cluster.astype(str), y=cp.broad_diss_rate * 100,
                         color="broad_diss_rate", color_continuous_scale="Reds",
                         text=(cp.broad_diss_rate * 100).round(1))
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(xaxis_title="Segmen (cluster)", yaxis_title="Ketidakpuasan %",
                              coloraxis_showscale=False); style(fig, 360)
        with c2:
            sec("Ukuran tiap segmen")
            fig = px.pie(cp, names=cp.cluster.astype(str), values="n_orders", hole=.55,
                         color_discrete_sequence=[BLUE, RED, AMBER, GREEN])
            fig.update_layout(legend_title="Cluster"); style(fig, 360)
        read("Bandingkan tinggi batang (risiko) dengan besar potongan pie (ukuran). "
             "Segmen kecil tapi berisiko tinggi = prioritas quality assurance.")
        take("Cluster 1 (order berat & mahal, jauh dari hub) punya ketidakpuasan <b>29,3%</b> vs cluster massal "
             "<b>20,5%</b> — segmen premium/berat butuh penanganan logistik khusus.")
        with st.expander("🔍 Profil lengkap tiap segmen"):
            st.dataframe(cp, use_container_width=True, hide_index=True)
            st.caption("Kolom avg_* = rata-rata karakteristik order (ongkir, harga, berat, estimasi hari, jarak, dll).")

# ══════════════════════════ 5 · RISK SCORER ═════════════════════════════════════
elif PAGE == "Risk Scorer":
    hero("Risk Scorer — alat skoring risiko", "Terjemahan model jadi alat praktis: atur ciri order → keluar tingkat risiko + rekomendasi.",
         ["rule-based", "arah koefisien LR", "ilustratif"])
    intro("Geser faktor-faktor sebuah order, sistem memperkirakan tingkat risiko (Rendah/Sedang/Tinggi) beserta "
          "rekomendasi tindakan. Bobot mengikuti arah pengaruh dari model Logistic Regression.")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("**Atur ciri-ciri order:**")
        carrier = st.slider("Lama seller serah paket ke kurir (jam)", 0, 240, 48, help="Makin lama → risiko naik. Sinyal terkuat.")
        estday = st.slider("Estimasi hari pengiriman (janji ke pelanggan)", 3, 60, 25, help="Janji makin lama → risiko naik.")
        fratio = st.slider("Rasio ongkir terhadap harga", 0.0, 1.0, 0.25, 0.01, help="Ongkir mahal relatif harga → risiko naik.")
        dist = st.slider("Jarak seller–pelanggan (km)", 0, 3000, 600, 50, help="Makin jauh → risiko naik.")
        approval = st.slider("Lama konfirmasi pembayaran (jam)", 0, 96, 12)
        same = st.checkbox("Seller & pelanggan satu provinsi", value=False, help="Satu provinsi → risiko turun.")
    pts = (min(carrier, 240) / 240 * 0.34 + min(estday, 60) / 60 * 0.26 + min(fratio, 1.0) * 0.16
           + min(dist, 3000) / 3000 * 0.14 + min(approval, 96) / 96 * 0.10 - (0.10 if same else 0))
    score = float(np.clip(pts, 0, 1))
    tier = "TINGGI" if score >= 0.55 else ("SEDANG" if score >= 0.35 else "RENDAH")
    color = {"TINGGI": RED, "SEDANG": AMBER, "RENDAH": GREEN}[tier]
    action = {"TINGGI": "Prioritas intervensi: percepat pickup kurir, notifikasi proaktif ke pelanggan, cek performa seller.",
              "SEDANG": "Pantau: kirim update status pengiriman, siapkan mitigasi bila proses melambat.",
              "RENDAH": "Tidak perlu tindakan khusus."}[tier]
    with c2:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=score * 100,
              gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                     "steps": [{"range": [0, 35], "color": "#dcfce7"}, {"range": [35, 55], "color": "#fef9c3"},
                               {"range": [55, 100], "color": "#fee2e2"}]},
              title={"text": "Skor risiko (0–100)"}))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=10),
                          font=dict(family="Inter, sans-serif")); st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"<h3 style='color:{color};margin:.2rem 0'>Tier: {tier}</h3>", unsafe_allow_html=True)
        st.write(action)
    read("Geser slider di kiri → jarum & tier di kanan berubah. Hijau = risiko rendah, merah = tinggi.")
    st.caption("Catatan: scorer ini rule-based/ilustratif (bobot mengikuti arah koefisien LR). Untuk prediksi presisi, "
               "gunakan model Logistic Regression penuh di notebook.")
