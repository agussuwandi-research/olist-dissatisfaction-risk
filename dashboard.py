"""
dashboard.py — Olist Customer Dissatisfaction Risk Modeling
Jalankan dengan: streamlit run dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# ── Konfigurasi ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Olist Dissatisfaction Risk Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"], p, div, span, li {
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: #e2e8f0;
}

/* ── KPI Cards ── */
.kpi-wrap {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-top: 3px solid var(--accent, #22d3ee);
    border-radius: 8px;
    padding: 1.1rem 1.3rem 1rem;
    height: 100%;
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94a3b8;
    margin-bottom: 0.45rem;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
}
.kpi-desc {
    font-size: 0.8rem;
    color: #64748b;
    margin-top: 0.35rem;
    line-height: 1.4;
}

/* ── Info/Finding boxes ── */
.insight-box {
    background: #1e293b;
    border-left: 4px solid #f59e0b;
    border-radius: 0 6px 6px 0;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.9rem;
    line-height: 1.6;
}
.insight-box .title {
    font-weight: 700;
    font-size: 0.85rem;
    color: #fbbf24;
    margin-bottom: 0.3rem;
}
.insight-box .body { color: #cbd5e1; }

.info-box {
    background: #0f2537;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.7rem;
    font-size: 0.88rem;
    line-height: 1.65;
    color: #bfdbfe;
}
.info-box strong { color: #60a5fa; }

.good-box {
    background: #052e16;
    border-left: 4px solid #22c55e;
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #bbf7d0;
    margin-bottom: 0.5rem;
}
.warn-box {
    background: #2d1b00;
    border-left: 4px solid #f59e0b;
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1rem;
    font-size: 0.88rem;
    color: #fde68a;
    margin-bottom: 0.5rem;
}

/* ── Significance badges ── */
.badge-sig {
    display: inline-block;
    background: #14532d;
    color: #86efac;
    border: 1px solid #22c55e;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
}
.badge-ns {
    display: inline-block;
    background: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section header ── */
.section-header {
    border-bottom: 1px solid #334155;
    padding-bottom: 0.5rem;
    margin-bottom: 1.2rem;
}
.section-header h3 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
}
.section-header p {
    font-size: 0.85rem;
    color: #64748b;
    margin: 0.3rem 0 0;
    line-height: 1.5;
}

/* ── Risk score display ── */
.risk-display {
    text-align: center;
    border-radius: 10px;
    padding: 1.5rem 1rem;
    border: 2px solid;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] { background: #0f172a; gap: 2px; }
.stTabs [data-baseweb="tab"] {
    background: #1e293b;
    color: #94a3b8;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.55rem 1.1rem;
    border-radius: 6px 6px 0 0;
}
.stTabs [aria-selected="true"] {
    background: #0f172a !important;
    color: #22d3ee !important;
    border-bottom: 2px solid #22d3ee !important;
}

/* ── Streamlit overrides ── */
.stDataFrame { font-size: 0.85rem; }
div[data-testid="stMetric"] label { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent
TABLES = ROOT / "outputs" / "tables"
INTERIM= ROOT / "data_interim"

@st.cache_data
def load(filename, folder="tables"):
    path = TABLES / filename if folder == "tables" else INTERIM / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None

# ── Plotly theme ───────────────────────────────────────────────────────────────
COLORS  = ["#22d3ee","#f59e0b","#f87171","#a78bfa","#34d399","#fb923c"]
THEME   = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0f172a",
    font=dict(family="Plus Jakarta Sans", color="#cbd5e1", size=12),
    xaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickfont=dict(color="#94a3b8")),
    yaxis=dict(gridcolor="#1e293b", linecolor="#334155", tickfont=dict(color="#94a3b8")),
    hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155",
                    font=dict(color="#f1f5f9", size=13)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1")),
)
def T(fig, **kw):
    # Merge keys yang ada di THEME sekaligus di kw (legend, xaxis, yaxis)
    merged = dict(THEME)
    for k, v in kw.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    fig.update_layout(**merged)
    return fig

def kpi(col, label, value, desc, accent="#22d3ee"):
    with col:
        st.markdown(f"""
        <div class="kpi-wrap" style="--accent:{accent}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

def insight(title, body):
    st.markdown(f"""
    <div class="insight-box">
      <div class="title">💡 {title}</div>
      <div class="body">{body}</div>
    </div>""", unsafe_allow_html=True)

def info(body):
    st.markdown(f'<div class="info-box">{body}</div>', unsafe_allow_html=True)

def section(title, subtitle=""):
    sub = f'<p>{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header">
      <h3>{title}</h3>{sub}
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
            border:1px solid #334155;border-radius:10px;
            padding:1.5rem 2rem;margin-bottom:1.5rem">
  <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.15em;
              text-transform:uppercase;color:#22d3ee;margin-bottom:0.4rem">
    Research Dashboard · Olist Brazilian E-Commerce Dataset (2016–2018)
  </div>
  <div style="font-size:1.55rem;font-weight:800;color:#f1f5f9;
              line-height:1.2;margin-bottom:0.5rem">
    Predicting Customer Dissatisfaction Before Delivery
  </div>
  <div style="font-size:0.88rem;color:#64748b;font-style:italic">
    Explainable Risk Scoring from Fulfillment and Logistics Signals in E-Commerce
  </div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs([
    "📊 Ringkasan Utama",
    "🤖 Performa Model",
    "🎯 Simulasi Intervensi",
    "📐 Pelanggaran Ekspektasi",
    "🗂  Segmentasi Risiko",
    "🔍 Analisis Error",
    "⚡ Risk Scoring Tool",
    "🗄  Audit Data",
    "👤 Info Peneliti",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RINGKASAN UTAMA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    section("Ringkasan Penelitian",
            "Penelitian ini membangun sistem scoring risiko untuk memprediksi pelanggan yang "
            "berpotensi kecewa — sebelum barang mereka tiba.")

    info("""<strong>Konteks:</strong> Dataset Olist mencakup 99.441 transaksi e-commerce Brasil.
    Kami fokus pada 95.568 order yang berhasil dikirim (<em>delivered</em>),
    dan membangun model yang bisa mendeteksi risiko ketidakpuasan pelanggan
    <strong>sejak order dibuat</strong> — jauh sebelum pelanggan menulis review.""")

    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1,"Total Order Dikirim","95.568","Periode Jan 2017 – Ags 2018","#22d3ee")
    kpi(c2,"Tingkat Ketidakpuasan","21,1%","Pelanggan yang memberi skor ≤ 3 dari 5","#f87171")
    kpi(c3,"Ketidakpuasan Berat","12,8%","Pelanggan yang memberi skor ≤ 2 dari 5","#f59e0b")
    kpi(c4,"AUC Model Terbaik","0,627","Model S2 LR — sinyal in-fulfillment","#22d3ee")
    kpi(c5,"Capture Rate Top-10%","20,4%","Tangkap 1 dari 5 kecewa hanya dari 10% order","#34d399")

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart, col_find = st.columns([1.3, 1])
    with col_chart:
        section("Volume Order per Bulan",
                "Garis putus-putus menunjukkan batas pembagian data (train / validation / test)")
        monthly = load("01_monthly_distribution.csv")
        if monthly is not None:
            monthly = monthly[
                (~monthly["ym"].str.startswith("2016")) &
                (~monthly["ym"].isin(["2018-09","2018-10"]))
            ].copy()
            colors_bar = []
            for ym in monthly["ym"]:
                if ym <= "2017-11":        colors_bar.append("#22d3ee")
                elif ym <= "2018-02":      colors_bar.append("#a78bfa")
                else:                      colors_bar.append("#f59e0b")
            fig = go.Figure(go.Bar(
                x=monthly["ym"], y=monthly["n_orders"],
                marker_color=colors_bar, opacity=0.85,
                hovertemplate="<b>%{x}</b><br>%{y:,} orders<extra></extra>"
            ))
            # Untuk categorical axis, pakai add_shape dengan koordinat paper
            ym_list = monthly["ym"].tolist()
            n = len(ym_list)
            for split_ym, color, label in [
                ("2017-11", "#a78bfa", "→ Validasi"),
                ("2018-03", "#f59e0b", "→ Test"),
            ]:
                if split_ym in ym_list:
                    idx = ym_list.index(split_ym)
                    xpos = (idx + 0.5) / n   # posisi paper coords (0-1)
                    fig.add_shape(type="line",
                                  x0=xpos, x1=xpos, y0=0, y1=1,
                                  xref="paper", yref="paper",
                                  line=dict(color=color, width=2, dash="dot"))
                    fig.add_annotation(
                        x=xpos, y=1.05, xref="paper", yref="paper",
                        text=label, showarrow=False,
                        font=dict(color=color, size=10), xanchor="left"
                    )
            # Legend manual
            for name, color in [("Train","#22d3ee"),("Validation","#a78bfa"),("Test","#f59e0b")]:
                fig.add_trace(go.Bar(x=[None], y=[None], name=name,
                                     marker_color=color, showlegend=True))
            T(fig, height=290, barmode="overlay",
              legend=dict(orientation="h", y=1.12, x=0),
              margin=dict(l=0,r=0,t=30,b=40))
            fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
            st.plotly_chart(fig, width="stretch")

    with col_find:
        section("6 Temuan Utama")
        findings = [
            ("Sinyal in-fulfillment terbukti signifikan",
             "Menambahkan info approval & carrier pickup meningkatkan akurasi model "
             "secara signifikan (DeLong test, p < 0,001)."),
            ("Terlambat 5 hari = 5× lebih merusak",
             "Barang tiba 5 hari lebih cepat dari janji: dissatisfaction 16,6%. "
             "Terlambat 5 hari: 85,9%. Asimetri 5,2×."),
            ("Model linear terbukti cukup kuat",
             "Logistic Regression setara Random Forest & mengungguli LightGBM. "
             "Hubungan antara sinyal dan ketidakpuasan bersifat linear."),
            ("Last-mile lebih kritis dari sisi seller",
             "Keterlambatan pengiriman carrier ke pelanggan 2,2× lebih merusak "
             "dibanding keterlambatan di sisi seller."),
            ("1 dari 5 pelanggan kecewa bisa dicegah",
             "Dengan memeriksa ulang 10% order berisiko tertinggi, "
             "20,4% pelanggan yang akan kecewa bisa diidentifikasi lebih awal."),
            ("79% FN tiba on-time",
             "Model memiliki blind spot: 79% ketidakpuasan yang tidak terdeteksi "
             "berasal dari faktor non-logistik (kualitas produk, ekspektasi)."),
        ]
        for title, body in findings:
            insight(title, body)

    st.markdown("---")
    section("Distribusi Data per Split", "Perhatikan validation memiliki positive rate lebih tinggi — ini karena mencakup periode Natal & Tahun Baru.")
    split = load("02_split_summary.csv")
    if split is not None:
        order_map = {"train":0,"validation":1,"test":2}
        split["_o"] = split["split"].map(order_map)
        split = split.sort_values("_o").drop(columns=["_o"])
        disp = split.copy()
        disp["n_orders"] = disp["n_orders"].apply(lambda x: f"{x:,}")
        disp["broad_positive_rate"] = disp["broad_positive_rate"].apply(lambda x: f"{x:.1%}")
        disp["severe_positive_rate"] = disp["severe_positive_rate"].apply(lambda x: f"{x:.1%}")
        disp.columns = ["Split","Jumlah Order","% Tidak Puas (Broad)","% Sangat Tidak Puas (Severe)","Tanggal Mulai","Tanggal Akhir"]
        st.dataframe(disp, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERFORMA MODEL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    section("Performa Model Klasifikasi",
            "Tiga algoritma diuji pada tiga skenario berbeda. Semua model dilatih pada data 2017 "
            "dan dievaluasi pada data 2018.")

    info("""<strong>Cara membaca:</strong>
    <br>• <strong>AUC (Area Under Curve)</strong> — Semakin tinggi, semakin baik model membedakan pelanggan yang kecewa vs tidak.
    Nilai 0,5 = model acak. Nilai 1,0 = sempurna.
    <br>• <strong>PR-AUC</strong> — Lebih relevan untuk data tidak seimbang. Mengukur seberapa presisi model ketika memprediksi kelas positif (kecewa).
    <br>• <strong>Recall</strong> — Berapa persen pelanggan kecewa yang berhasil terdeteksi model.
    <br>• <strong>S1</strong> = hanya sinyal saat order dibuat &nbsp;|&nbsp;
    <strong>S2/S3</strong> = tambahan sinyal setelah proses pengiriman dimulai.""")

    metrics = load("04_classification_metrics.csv")
    topk    = load("04_topk_capture.csv")
    delong  = load("08_delong_results.csv")

    sc_map = {
        "S1_pre_broad":  "S1 — Pra-Pengiriman (Broad)",
        "S2_in_broad":   "S2 — In-Pengiriman (Broad) ★",
        "S3_in_severe":  "S3 — In-Pengiriman (Severe)",
    }

    if metrics is not None:
        real = metrics[metrics["model"]!="Dummy"].copy()
        real["sc_label"] = real["scenario"].map(sc_map).fillna(real["scenario"])

        col1, col2 = st.columns(2)
        with col1:
            section("ROC-AUC per Skenario & Model",
                    "★ S2 adalah model utama yang direkomendasikan")
            fig = go.Figure()
            for i, model in enumerate(["LR","RF","LightGBM"]):
                d = real[real["model"]==model]
                fig.add_trace(go.Bar(
                    name=model, x=d["sc_label"], y=d["roc_auc"],
                    marker_color=COLORS[i],
                    text=d["roc_auc"].round(4).astype(str),
                    textposition="outside", textfont=dict(size=10, color="#f1f5f9"),
                    hovertemplate="<b>%{x}</b><br>Model: "+model+"<br>AUC: %{y:.4f}<extra></extra>"
                ))
            fig.add_hline(y=0.5, line_color="#475569", line_dash="dot",
                          annotation_text="Batas acak (0,5)",
                          annotation_font=dict(color="#64748b", size=10),
                          annotation_position="bottom right")
            T(fig, barmode="group", height=320,
              yaxis_range=[0.48, 0.68], yaxis_tickformat=".2f",
              legend=dict(orientation="h", y=1.1),
              margin=dict(l=0,r=0,t=40,b=0))
            fig.update_xaxes(tickfont=dict(size=10))
            st.plotly_chart(fig, width="stretch")

        with col2:
            section("PR-AUC per Skenario & Model",
                    "Lebih informatif dari AUC untuk kelas tidak seimbang (21% positif)")
            fig2 = go.Figure()
            for i, model in enumerate(["LR","RF","LightGBM"]):
                d = real[real["model"]==model]
                fig2.add_trace(go.Bar(
                    name=model, x=d["sc_label"], y=d["pr_auc"],
                    marker_color=COLORS[i],
                    text=d["pr_auc"].round(4).astype(str),
                    textposition="outside", textfont=dict(size=10, color="#f1f5f9"),
                ))
            T(fig2, barmode="group", height=320,
              yaxis_range=[0.18, 0.36], yaxis_tickformat=".2f",
              legend=dict(orientation="h", y=1.1),
              margin=dict(l=0,r=0,t=40,b=0))
            fig2.update_xaxes(tickfont=dict(size=10))
            st.plotly_chart(fig2, width="stretch")

        # Insight model
        col_ins1, col_ins2 = st.columns(2)
        with col_ins1:
            st.markdown("""
            <div class="good-box">
            ✅ <strong>Logistic Regression = model terbaik dan paling mudah dijelaskan</strong><br>
            LR menghasilkan AUC 0,627 — setara Random Forest dan lebih baik dari LightGBM.
            Karena sederhana, koefisiennya bisa langsung diinterpretasikan sebagai 
            "fitur apa yang paling berpengaruh."
            </div>""", unsafe_allow_html=True)
        with col_ins2:
            st.markdown("""
            <div class="warn-box">
            ⚠️ <strong>Mengapa LightGBM lebih rendah dari LR?</strong><br>
            Tuning dengan 50 trial Optuna mengkonfirmasi: model optimal memiliki
            num_leaves=16 (sangat dangkal). Artinya hubungan antara fitur dan 
            ketidakpuasan bersifat cukup linear — model kompleks tidak membantu.
            </div>""", unsafe_allow_html=True)

        # Full table
        st.markdown("---")
        section("Tabel Metrik Lengkap", "Dummy = model acak sebagai baseline perbandingan")
        all_m = metrics.copy()
        all_m["scenario"] = all_m["scenario"].map(sc_map).fillna(all_m["scenario"])
        all_m.columns = ["Skenario","Model","AUC","PR-AUC","Recall","Precision","F1","Brier Score","Threshold"]
        st.dataframe(
            all_m.style
                .background_gradient(subset=["AUC","PR-AUC"], cmap="RdYlGn",
                                     vmin=0.20, vmax=0.65)
                .format({c:"{:.4f}" for c in ["AUC","PR-AUC","Recall","Precision",
                                               "F1","Brier Score","Threshold"]}),
            width="stretch", hide_index=True
        )

    # DeLong
    st.markdown("---")
    section("Uji Signifikansi Statistik (DeLong Test)",
            "Apakah perbedaan AUC antara skenario benar-benar bermakna, atau hanya kebetulan?")

    info("""<strong>Apa itu DeLong test?</strong>
    Uji statistik untuk membuktikan bahwa perbedaan AUC antara dua model bukan terjadi karena kebetulan.
    <br>• <strong>p &lt; 0,001</strong> (***) = sangat signifikan, hampir mustahil terjadi karena kebetulan
    <br>• <strong>ns</strong> = not significant, perbedaan tidak bermakna secara statistik""")

    if delong is not None:
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            section("RQ2: Apakah sinyal in-fulfillment benar-benar membantu?")
            rq2 = delong[delong["comparison"].str.startswith("RQ2")].copy()
            for _, row in rq2.iterrows():
                model_name = row["comparison"].replace("RQ2: ","").replace(" S1 vs S2","")
                badge = f'<span class="badge-sig">p={row.p_value:.4f} {row.significance}</span>'
                st.markdown(f"""
                <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;
                            padding:1rem 1.2rem;margin-bottom:0.7rem">
                  <div style="font-weight:700;font-size:1rem;color:#f1f5f9;margin-bottom:0.3rem">
                    {model_name}
                  </div>
                  <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:0.4rem">
                    <span style="font-family:monospace;font-size:1.3rem;font-weight:700;
                                 color:#22d3ee">Δ AUC = {row.delta_auc:+.4f}</span>
                    {badge}
                  </div>
                  <div style="font-size:0.82rem;color:#64748b">
                    S1 (pra-pengiriman): {row.auc_B:.4f} &nbsp;→&nbsp;
                    S2 (in-pengiriman): {row.auc_A:.4f}
                  </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="good-box">
            ✅ <strong>Kesimpulan RQ2:</strong> Gain signifikan di semua model (p &lt; 0,001).
            Sinyal in-fulfillment (waktu approval & carrier pickup) benar-benar
            menambah kemampuan prediksi secara statistik.
            </div>""", unsafe_allow_html=True)

        with col_d2:
            section("Perbandingan Antar Model (S2)")
            disc = delong[delong["comparison"].str.startswith("Disc")].copy()
            for _, row in disc.iterrows():
                is_ns = row.significance == "ns"
                badge = (f'<span class="badge-ns">Tidak Signifikan (p={row.p_value:.3f})</span>'
                         if is_ns else
                         f'<span class="badge-sig">{row.significance} p={row.p_value:.4f}</span>')
                st.markdown(f"""
                <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;
                            padding:1rem 1.2rem;margin-bottom:0.7rem">
                  <div style="font-weight:700;color:#f1f5f9;margin-bottom:0.3rem">
                    {row['comparison'].replace('Disc: ','')}
                  </div>
                  <div style="display:flex;align-items:baseline;gap:0.8rem;margin-bottom:0.4rem">
                    <span style="font-family:monospace;font-size:1.2rem;font-weight:700;
                                 color:{'#94a3b8' if is_ns else '#22d3ee'}">
                      Δ = {row.delta_auc:+.4f}</span>
                    {badge}
                  </div>
                </div>""", unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            <strong>Interpretasi:</strong><br>
            LR dan Random Forest <em>tidak berbeda signifikan</em> — keduanya sama baiknya.
            LR unggul atas LightGBM secara signifikan (p &lt; 0,001).
            Pilihan LR sebagai model utama didasarkan pada performa setara + interpretasi lebih mudah.
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SIMULASI INTERVENSI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    section("Simulasi Intervensi Operasional",
            "Jika tim operasional memiliki kapasitas untuk memeriksa ulang X% order berisiko tinggi, "
            "berapa banyak pelanggan kecewa yang bisa diidentifikasi lebih awal?")

    info("""<strong>Cara kerja:</strong> Model memberi setiap order sebuah <em>risk score</em> (0–1).
    Order dengan score tertinggi diprioritaskan untuk intervensi (misalnya: notifikasi pelanggan proaktif,
    pengecekan status pengiriman, atau eskalasi ke tim logistik).
    Grafik ini menunjukkan <strong>seberapa efektif prioritisasi ini</strong> dibandingkan memilih order secara acak.""")

    topk = load("04_topk_capture.csv")
    if topk is not None:
        s2_lr = topk[(topk["scenario"]=="S2_in_broad") & (topk["model"]=="LR")].iloc[0]
        known_pcts    = [0, 5, 10, 20, 30, 100]
        known_recalls = [0, s2_lr["top_5pct_recall"], s2_lr["top_10pct_recall"],
                         s2_lr["top_20pct_recall"], s2_lr["top_30pct_recall"], 1.0]

        col_ctrl, col_result = st.columns([1, 2])
        with col_ctrl:
            section("Atur Parameter")
            top_pct = st.slider(
                "Berapa % order teratas yang akan diperiksa ulang?",
                min_value=1, max_value=50, value=10, step=1,
                help="Geser untuk melihat efek dari berbagai tingkat intervensi"
            )
            capture = float(np.interp(top_pct, known_pcts, known_recalls))
            total_test = 38948
            total_diss = int(total_test * 0.199)
            flagged    = int(total_test * top_pct / 100)
            caught     = int(total_diss * capture)
            lift       = capture / (top_pct / 100)
            missed     = total_diss - caught

            st.markdown(f"""
            <div style="display:grid;gap:0.6rem;margin-top:0.5rem">
              <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;
                          padding:1rem 1.1rem">
                <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                            letter-spacing:0.08em;color:#94a3b8;margin-bottom:0.3rem">
                  Order yang diperiksa ulang</div>
                <div style="font-family:monospace;font-size:1.6rem;font-weight:700;
                            color:#22d3ee">{flagged:,}</div>
                <div style="font-size:0.8rem;color:#64748b">{top_pct}% dari {total_test:,} order test</div>
              </div>
              <div style="background:#052e16;border:1px solid #166534;border-radius:8px;
                          padding:1rem 1.1rem">
                <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                            letter-spacing:0.08em;color:#86efac;margin-bottom:0.3rem">
                  Pelanggan kecewa yang terdeteksi</div>
                <div style="font-family:monospace;font-size:1.6rem;font-weight:700;
                            color:#4ade80">{caught:,} ({capture:.1%})</div>
                <div style="font-size:0.8rem;color:#166534">dari {total_diss:,} total pelanggan kecewa</div>
              </div>
              <div style="background:#1e1a07;border:1px solid #713f12;border-radius:8px;
                          padding:1rem 1.1rem">
                <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                            letter-spacing:0.08em;color:#fde68a;margin-bottom:0.3rem">
                  Efisiensi vs pemilihan acak</div>
                <div style="font-family:monospace;font-size:1.6rem;font-weight:700;
                            color:#fbbf24">{lift:.2f}× lebih efisien</div>
                <div style="font-size:0.8rem;color:#78350f">
                  Tanpa model: periksa {top_pct}% → tangkap {top_pct}% saja</div>
              </div>
              <div style="background:#1e293b;border:1px solid #7f1d1d;border-radius:8px;
                          padding:1rem 1.1rem">
                <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;
                            letter-spacing:0.08em;color:#fca5a5;margin-bottom:0.3rem">
                  Pelanggan kecewa yang terlewat</div>
                <div style="font-family:monospace;font-size:1.6rem;font-weight:700;
                            color:#f87171">{missed:,} ({missed/total_diss:.1%})</div>
                <div style="font-size:0.8rem;color:#7f1d1d">blind spot model</div>
              </div>
            </div>""", unsafe_allow_html=True)

        with col_result:
            section("Cumulative Gain Curve",
                    "Kurva hijau = model kita. Garis putus-putus = pemilihan acak. "
                    "Semakin jauh kurva hijau di atas garis acak, semakin baik model.")
            pcts = list(range(0, 101))
            recalls = [float(np.interp(p, known_pcts, known_recalls)) for p in pcts]
            random  = [p/100 for p in pcts]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=pcts, y=[r*100 for r in recalls],
                name="Model S2 LR", mode="lines",
                line=dict(color="#22d3ee", width=3),
                fill="tonexty", fillcolor="rgba(34,211,238,0.07)",
                hovertemplate="<b>Flag %{x}% order</b><br>Tangkap %{y:.1f}% pelanggan kecewa<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=pcts, y=[r*100 for r in random],
                name="Pilihan Acak (baseline)", mode="lines",
                line=dict(color="#475569", width=1.5, dash="dot"),
                hovertemplate="Acak: flag %{x}% → tangkap %{y:.0f}%<extra></extra>"
            ))
            # Titik yang dipilih slider
            fig.add_trace(go.Scatter(
                x=[top_pct], y=[capture*100],
                mode="markers",
                marker=dict(color="#f59e0b", size=12, symbol="circle",
                            line=dict(color="#fff", width=2)),
                name=f"Pilihan saat ini ({top_pct}%)",
                hovertemplate=f"Flag {top_pct}% → tangkap {capture:.1%}<extra></extra>"
            ))
            fig.add_vline(x=top_pct, line_color="#f59e0b", line_width=1.5, line_dash="dash",
                          annotation_text=f"{top_pct}%",
                          annotation_font=dict(color="#fbbf24", size=11))
            fig.add_hline(y=capture*100, line_color="#f59e0b", line_width=1.5, line_dash="dash",
                          annotation_text=f"{capture:.1%}",
                          annotation_font=dict(color="#fbbf24", size=11))
            T(fig, height=380,
              xaxis_title="% Order yang Di-flag (berdasarkan risk score tertinggi)",
              yaxis_title="% Pelanggan Kecewa yang Terdeteksi",
              xaxis=dict(**THEME["xaxis"], range=[0,100], ticksuffix="%"),
              yaxis=dict(**THEME["yaxis"], range=[0,100], ticksuffix="%"),
              legend=dict(orientation="h", y=1.1),
              margin=dict(l=0,r=0,t=40,b=50))
            st.plotly_chart(fig, width="stretch")

            # Tabel benchmark
            section("Perbandingan di Titik-Titik Kunci")
            bench = pd.DataFrame({
                "% Order diperiksa": ["5%","10%","20%","30%"],
                "Jumlah order": [f"{int(total_test*p/100):,}" for p in [5,10,20,30]],
                "Pelanggan kecewa terdeteksi": [
                    f"{int(total_diss*s2_lr['top_5pct_recall']):,}  ({s2_lr['top_5pct_recall']:.1%})",
                    f"{int(total_diss*s2_lr['top_10pct_recall']):,}  ({s2_lr['top_10pct_recall']:.1%})",
                    f"{int(total_diss*s2_lr['top_20pct_recall']):,}  ({s2_lr['top_20pct_recall']:.1%})",
                    f"{int(total_diss*s2_lr['top_30pct_recall']):,}  ({s2_lr['top_30pct_recall']:.1%})",
                ],
                "Efisiensi vs acak": [
                    f"{s2_lr['top_5pct_recall']/0.05:.1f}×",
                    f"{s2_lr['top_10pct_recall']/0.10:.1f}×",
                    f"{s2_lr['top_20pct_recall']/0.20:.1f}×",
                    f"{s2_lr['top_30pct_recall']/0.30:.1f}×",
                ],
            })
            st.dataframe(bench, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PELANGGARAN EKSPEKTASI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    section("Pelanggaran Ekspektasi & Rantai Pengiriman",
            "Mengapa pelanggan kecewa? Bukan semata-mata karena terlambat — tapi karena "
            "janjinya dilanggar.")

    info("""<strong>Expectation-Disconfirmation Theory (Oliver, 1980):</strong>
    Ketidakpuasan pelanggan bukan ditentukan oleh kualitas mutlak, tapi oleh
    <strong>selisih antara ekspektasi dan kenyataan</strong>.
    <br><br>Dalam konteks ini: platform Olist memberikan estimasi tanggal pengiriman.
    Semakin besar barang terlambat dari estimasi tersebut, semakin besar kemungkinan pelanggan kecewa.
    Yang mengejutkan: barang datang lebih cepat dari estimasi <em>hampir tidak berpengaruh positif</em>.""")

    ebins = load("05_estimation_error_bins.csv")
    asym  = load("05_asymmetry_check.csv")
    chain = load("05_delivery_chain_quartiles.csv")
    lauc  = load("05_late_vs_error_auc.csv")

    col1, col2 = st.columns(2)
    with col1:
        section("Tingkat Ketidakpuasan vs Selisih Estimasi Pengiriman",
                "Biru = tiba lebih cepat dari janji | Merah = tiba lebih lambat dari janji")
        if ebins is not None:
            colors_bin = ["#22d3ee" if "-" in str(b) else "#f87171" for b in ebins["err_bin"]]
            fig = go.Figure(go.Bar(
                x=ebins["err_bin"].astype(str), y=ebins["diss_pct"],
                marker_color=colors_bin,
                text=[f"{v:.0f}%" for v in ebins["diss_pct"]],
                textposition="outside", textfont=dict(size=10, color="#e2e8f0"),
                hovertemplate="<b>Selisih %{x} hari</b><br>Tingkat ketidakpuasan: %{y:.1f}%<extra></extra>"
            ))
            # Garis batas — antara bin "-2:0" (index 5) dan "0:2" (index 6)
            fig.add_shape(type="line", x0=4.5, x1=4.5, y0=0, y1=95,
                          line=dict(color="#ffffff", width=2, dash="dash"))
            fig.add_annotation(x=4.5, y=97, text="← Lebih cepat | Lebih lambat →",
                               font=dict(color="#f1f5f9", size=11), showarrow=False,
                               xanchor="center")
            T(fig, height=330,
              xaxis_title="Selisih dari estimasi (hari) — negatif = tiba lebih cepat",
              yaxis_title="Tingkat Ketidakpuasan (%)",
              yaxis=dict(**THEME["yaxis"], range=[0, 105]),
              margin=dict(l=0,r=0,t=10,b=50))
            fig.update_xaxes(tickfont=dict(size=9), tickangle=30)
            st.plotly_chart(fig, width="stretch")

    with col2:
        section("Asimetri: Tiba Lebih Cepat vs Lebih Lambat",
                "Berapa kali lebih merusak jika terlambat vs lebih cepat dari janji?")
        if asym is not None:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Tiba lebih cepat dari janji",
                x=[f"±{d} hari" for d in asym["delta_days"]],
                y=asym["early_diss_rate"]*100,
                marker_color="#22d3ee",
                text=[f"{v:.1f}%" for v in asym["early_diss_rate"]*100],
                textposition="outside", textfont=dict(size=10, color="#e2e8f0"),
            ))
            fig.add_trace(go.Bar(
                name="Tiba lebih lambat dari janji",
                x=[f"±{d} hari" for d in asym["delta_days"]],
                y=asym["late_diss_rate"]*100,
                marker_color="#f87171",
                text=[f"{v:.1f}%" for v in asym["late_diss_rate"]*100],
                textposition="outside", textfont=dict(size=10, color="#e2e8f0"),
            ))
            for _, row in asym.iterrows():
                fig.add_annotation(
                    x=f"±{int(row.delta_days)} hari",
                    y=row.late_diss_rate*100 + 5,
                    text=f"<b>{row.ratio:.1f}× lebih parah</b>",
                    font=dict(color="#fbbf24", size=11), showarrow=False
                )
            T(fig, barmode="group", height=330,
              yaxis=dict(**THEME["yaxis"], range=[0,105], ticksuffix="%"),
              legend=dict(orientation="h", y=1.1),
              margin=dict(l=0,r=0,t=40,b=10))
            st.plotly_chart(fig, width="stretch")

    insight("Efek Loss Aversion",
            "Data menunjukkan asimetri yang kuat: terlambat 5 hari dari janji (85,9% kecewa) "
            "5× lebih merusak dibanding tiba 5 hari lebih cepat (16,6% kecewa). "
            "Ini konsisten dengan Loss Aversion (Kahneman & Tversky, 1979) — "
            "kehilangan dirasakan jauh lebih kuat daripada keuntungan setara.")

    # AUC comparison
    if lauc is not None:
        st.markdown("---")
        section("Perbandingan: Sinyal Keterlambatan Biasa vs Selisih Estimasi",
                "Mana yang lebih baik untuk memprediksi ketidakpuasan?")
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            v = lauc[lauc['feature']=='is_late']['auc_vs_broad_diss'].values[0]
            st.markdown(f"""
            <div class="kpi-wrap" style="--accent:#64748b">
              <div class="kpi-label">is_late (biner: terlambat/tidak)</div>
              <div class="kpi-value" style="color:#94a3b8">{v:.4f}</div>
              <div class="kpi-desc">Hanya membedakan terlambat vs tidak</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            v2 = lauc[lauc['feature']=='estimation_error']['auc_vs_broad_diss'].values[0]
            st.markdown(f"""
            <div class="kpi-wrap" style="--accent:#22d3ee">
              <div class="kpi-label">estimation_error (kontinu) ★ Lebih baik</div>
              <div class="kpi-value">{v2:.4f}</div>
              <div class="kpi-desc">Mengukur seberapa besar pelanggaran janji</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class="info-box">
            <strong>Mengapa ini penting?</strong><br>
            Model yang hanya bertanya "apakah terlambat?" kurang informatif dibandingkan
            model yang bertanya "seberapa jauh dari estimasi?"
            Selisih AUC 0,019 ini, jika diuji signifikansi, cukup bermakna secara statistik.
            </div>""", unsafe_allow_html=True)

    # Delivery chain
    st.markdown("---")
    section("Dekomposisi Rantai Pengiriman",
            "Di fase mana masalah lebih sering muncul: saat seller memproses order, atau saat kurir mengantarkan?")
    if chain is not None:
        info("""Pengiriman dibagi dua fase:
        <br>• <strong>Fase Seller</strong>: Dari order dibuat → barang diserahkan ke kurir. Tanggung jawab seller.
        <br>• <strong>Fase Logistik (Last-Mile)</strong>: Dari kurir menerima barang → sampai ke pelanggan. Tanggung jawab kurir.
        <br><br>Q1 = 25% order dengan fase tercepat, Q4 = 25% terlama. Semakin tinggi Q, semakin lama fase tersebut.""")
        col_c1, col_c2 = st.columns(2)
        for col, phase, label, color, multiplier_note in [
            (col_c1, "seller_phase_days", "Fase Seller (Order → Serah ke Kurir)", "#f59e0b",
             "Keterlambatan di sisi seller 1,6× lebih berdampak dari Q1 ke Q4"),
            (col_c2, "logistics_phase_days", "Fase Last-Mile (Kurir → Pelanggan) ⚡", "#f87171",
             "Keterlambatan last-mile 2,2× lebih berdampak — fase paling kritis"),
        ]:
            phase_data = chain[chain["phase"]==phase].copy()
            with col:
                def hex_rgba(h, a):
                    h = h.lstrip("#")
                    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"
                bar_colors = [hex_rgba(color, a) for a in [0.35, 0.55, 0.75, 1.0]]
                fig = go.Figure(go.Bar(
                    x=phase_data["quartile"].astype(str),
                    y=phase_data["diss_rate"]*100,
                    marker_color=bar_colors,
                    text=[f"{v:.1f}%" for v in phase_data["diss_rate"]*100],
                    textposition="outside", textfont=dict(size=11, color="#e2e8f0"),
                    hovertemplate="Kuartil %{x}<br>Ketidakpuasan: %{y:.1f}%<extra></extra>"
                ))
                mult = phase_data["diss_rate"].max() / phase_data["diss_rate"].min()
                T(fig, height=300,
                  title=dict(text=f"<b>{label}</b><br>"
                             f"<sup style='color:#64748b'>{multiplier_note}</sup>",
                             font=dict(size=12, color="#f1f5f9")),
                  xaxis_title="Kuartil (Q1=tercepat, Q4=terlama)",
                  yaxis=dict(**THEME["yaxis"], range=[0,42], ticksuffix="%"),
                  margin=dict(l=0,r=0,t=70,b=40))
                st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SEGMENTASI RISIKO
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    section("Segmentasi Risiko — Clustering K-Means",
            "Selain memprediksi per order, kita juga mengelompokkan order ke dalam "
            "profil risiko yang berbeda untuk panduan intervensi yang lebih terarah.")

    cluster  = load("06_cluster_profile.csv")
    cat_diss = load("05_category_dissatisfaction.csv")
    state_d  = load("05_state_dissatisfaction.csv")

    if cluster is not None:
        info("""Clustering K-Means mengelompokkan order berdasarkan karakteristik fisik dan logistik
        (berat, harga, ongkir, jarak, estimasi hari). Hasilnya: <strong>2 profil order</strong>
        dengan tingkat ketidakpuasan yang sangat berbeda.""")

        c1, c2 = st.columns(2)
        for _, row in cluster.sort_values("broad_diss_rate", ascending=True).iterrows():
            cl = int(row["cluster"])
            is_high = row["broad_diss_rate"] > 0.22
            border = "#f87171" if is_high else "#22d3ee"
            bg = "#2d0000" if is_high else "#001e2d"
            icon = "🔴" if is_high else "🟢"
            name = "High-Friction (Berisiko Tinggi)" if is_high else "Standard (Berisiko Normal)"
            col = c2 if is_high else c1
            with col:
                st.markdown(f"""
                <div style="background:{bg};border:2px solid {border};border-radius:10px;
                            padding:1.3rem 1.5rem;height:100%">
                  <div style="font-size:1.1rem;font-weight:800;color:{border};
                              margin-bottom:1rem">{icon} Cluster {cl}: {name}</div>
                  <div style="display:grid;grid-template-columns:1fr 1fr;
                              gap:0.8rem 1.5rem">
                    <div>
                      <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;
                                  letter-spacing:0.08em;color:#94a3b8">Jumlah Order</div>
                      <div style="font-family:monospace;font-size:1.3rem;font-weight:700;
                                  color:#f1f5f9">{int(row['n_orders']):,}</div>
                      <div style="font-size:0.75rem;color:#64748b">
                        {row['n_orders']/cluster['n_orders'].sum():.1%} dari total</div>
                    </div>
                    <div>
                      <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;
                                  letter-spacing:0.08em;color:#94a3b8">Tingkat Ketidakpuasan</div>
                      <div style="font-family:monospace;font-size:1.3rem;font-weight:700;
                                  color:{border}">{row['broad_diss_rate']:.1%}</div>
                      <div style="font-size:0.75rem;color:#64748b">
                        review score ≤ 3</div>
                    </div>
                    <div>
                      <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;
                                  letter-spacing:0.08em;color:#94a3b8">Rata-rata Berat</div>
                      <div style="font-family:monospace;font-size:1.1rem;font-weight:700;
                                  color:#e2e8f0">{row['avg_weight_sum']:,.0f} gram</div>
                    </div>
                    <div>
                      <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;
                                  letter-spacing:0.08em;color:#94a3b8">Rata-rata Ongkir</div>
                      <div style="font-family:monospace;font-size:1.1rem;font-weight:700;
                                  color:#e2e8f0">R$ {row['avg_freight_sum']:.1f}</div>
                    </div>
                    <div>
                      <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;
                                  letter-spacing:0.08em;color:#94a3b8">Rata-rata Harga</div>
                      <div style="font-family:monospace;font-size:1.1rem;font-weight:700;
                                  color:#e2e8f0">R$ {row['avg_price_sum']:.0f}</div>
                    </div>
                    <div>
                      <div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;
                                  letter-spacing:0.08em;color:#94a3b8">Rata-rata Jarak</div>
                      <div style="font-family:monospace;font-size:1.1rem;font-weight:700;
                                  color:#e2e8f0">{row['avg_distance_km']:.0f} km</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

        insight("Interpretasi Cluster",
                f"<strong>Cluster High-Friction</strong> (6,2% dari order) memiliki tingkat ketidakpuasan "
                f"29,3% vs 20,5% pada cluster normal — selisih 9 poin persentase. "
                f"Karakteristik utama: produk berat (16kg vs 1,5kg), ongkir 3,6× lebih mahal, "
                f"dan harga 4× lebih tinggi. Order seperti ini membutuhkan perhatian operasional lebih.")

        # Radar chart
        st.markdown("---")
        section("Profil Visual Antar Cluster",
                "Setiap sudut = satu dimensi. Semakin jauh dari pusat = semakin tinggi nilainya relatif terhadap cluster lain.")
        radar_feats = ["avg_freight_ratio","avg_weight_sum","avg_est_days",
                       "avg_distance_km","avg_n_items"]
        radar_feats = [f for f in radar_feats if f in cluster.columns]
        radar_labels = {"avg_freight_ratio":"Rasio Ongkir","avg_weight_sum":"Berat Produk",
                        "avg_est_days":"Estimasi Hari","avg_distance_km":"Jarak",
                        "avg_n_items":"Jumlah Item"}
        labels = [radar_labels.get(f, f) for f in radar_feats]

        def hex_rgba(h, a):
            h = h.lstrip("#")
            return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

        mins = cluster[radar_feats].min()
        maxs = cluster[radar_feats].max()
        fig = go.Figure()
        for _, row in cluster.iterrows():
            vals = [(row[f]-mins[f])/(maxs[f]-mins[f]+1e-9) for f in radar_feats]
            vals += [vals[0]]
            is_h = row["broad_diss_rate"] > 0.22
            color = "#f87171" if is_h else "#22d3ee"
            name  = f"Cluster {int(row['cluster'])}: {'High-Friction' if is_h else 'Standard'} ({row['broad_diss_rate']:.1%} diss)"
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels+[labels[0]],
                name=name, fill="toself",
                line=dict(color=color, width=2.5),
                fillcolor=hex_rgba(color, 0.12),
            ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#1e293b",
                                tickfont=dict(color="#475569", size=9)),
                bgcolor="#0f172a",
                angularaxis=dict(tickfont=dict(color="#cbd5e1", size=11))
            ),
            legend=dict(orientation="h", y=-0.1, font=dict(color="#cbd5e1")),
            height=380, paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40,r=40,t=20,b=60)
        )
        st.plotly_chart(fig, width="stretch")

    # Category dan state
    st.markdown("---")
    col_cat, col_state = st.columns(2)
    with col_cat:
        section("Tingkat Ketidakpuasan per Kategori Produk",
                "Merah = di atas rata-rata | Hijau = di bawah rata-rata")
        if cat_diss is not None:
            top8 = cat_diss.nlargest(8, "diss_rate")
            bot5 = cat_diss.nsmallest(5, "diss_rate")
            cat_plot = pd.concat([top8, bot5]).drop_duplicates()
            avg_d = cat_diss["diss_rate"].mean()
            # Terjemahan label
            label_map = {
                "office_furniture":"Furnitur Kantor","bed_bath_table":"Kasur & Mandi",
                "furniture_decor":"Furnitur Dekorasi","computers_accessories":"Aksesoris Komputer",
                "telephony":"Telepon","cool_stuff":"Barang Unik",
                "home_appliances":"Peralatan Rumah","sports_leisure":"Olahraga & Hobi",
                "audio":"Audio","fashion_male_clothing":"Fashion Pria",
                "books_general_interest":"Buku Umum","luggage_accessories":"Koper & Tas",
                "food_drink":"Makanan & Minuman","food":"Makanan",
                "books_technical":"Buku Teknis",
            }
            cat_plot = cat_plot.copy()
            cat_plot["label"] = cat_plot["top_category"].map(label_map).fillna(cat_plot["top_category"])
            cat_plot = cat_plot.sort_values("diss_rate")
            colors_c = ["#f87171" if r > avg_d else "#22d3ee" for r in cat_plot["diss_rate"]]
            fig = go.Figure(go.Bar(
                x=cat_plot["diss_rate"]*100, y=cat_plot["label"],
                orientation="h", marker_color=colors_c,
                text=[f"{v:.1f}%" for v in cat_plot["diss_rate"]*100],
                textposition="outside", textfont=dict(size=10, color="#e2e8f0"),
                hovertemplate="<b>%{y}</b><br>Ketidakpuasan: %{x:.1f}%<extra></extra>"
            ))
            fig.add_vline(x=avg_d*100, line_color="#fbbf24", line_dash="dot",
                          annotation_text=f"Rata-rata {avg_d:.1%}",
                          annotation_font=dict(color="#fbbf24", size=10),
                          annotation_position="top right")
            T(fig, height=400,
              xaxis=dict(**THEME["xaxis"], ticksuffix="%", range=[0,44]),
              margin=dict(l=0,r=50,t=10,b=10))
            st.plotly_chart(fig, width="stretch")

    with col_state:
        section("Tingkat Ketidakpuasan per Negara Bagian",
                "Perbedaan geografis menunjukkan adanya variasi dalam kualitas logistik antar wilayah")
        if state_d is not None:
            sd = state_d[state_d["n"]>=200].sort_values("diss_rate", ascending=True).tail(15)
            avg_s = state_d["diss_rate"].mean()
            colors_s = ["#f87171" if r > avg_s else "#22d3ee" for r in sd["diss_rate"]]
            fig = go.Figure(go.Bar(
                x=sd["diss_rate"]*100, y=sd["customer_state"],
                orientation="h", marker_color=colors_s,
                text=[f"{v:.1f}%" for v in sd["diss_rate"]*100],
                textposition="outside", textfont=dict(size=10, color="#e2e8f0"),
                hovertemplate="<b>%{y}</b><br>Ketidakpuasan: %{x:.1f}%<extra></extra>"
            ))
            fig.add_vline(x=avg_s*100, line_color="#fbbf24", line_dash="dot",
                          annotation_text=f"Rata-rata {avg_s:.1%}",
                          annotation_font=dict(color="#fbbf24", size=10),
                          annotation_position="top right")
            T(fig, height=400,
              xaxis=dict(**THEME["xaxis"], ticksuffix="%", range=[0,36]),
              margin=dict(l=0,r=50,t=10,b=10))
            st.plotly_chart(fig, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ANALISIS ERROR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    section("Analisis Error Model",
            "Setiap model pasti memiliki blind spot. Di sini kita analisis: di mana model gagal "
            "dan mengapa — serta apa artinya untuk praktik operasional.")

    TP, FP, FN, TN = 3651, 8854, 4099, 22344
    total_pos = TP + FN

    info("""<strong>4 Jenis Prediksi Model:</strong>
    <br>• <strong>TP (True Positive)</strong> — Model benar memprediksi pelanggan akan kecewa ✅
    <br>• <strong>FP (False Positive)</strong> — Model keliru mem-flag pelanggan yang sebenarnya tidak kecewa (false alarm) ⚠️
    <br>• <strong>FN (False Negative)</strong> — Pelanggan kecewa yang TIDAK terdeteksi model (blind spot) ❌
    <br>• <strong>TN (True Negative)</strong> — Model benar memprediksi pelanggan tidak akan kecewa ✅""")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "True Positive (Terdeteksi)", f"{TP:,}", f"{TP/total_pos:.1%} dari pelanggan kecewa berhasil ditangkap", "#22d3ee")
    kpi(c2, "False Negative (Terlewat)", f"{FN:,}", f"{FN/total_pos:.1%} pelanggan kecewa tidak terdeteksi — analisis utama", "#f87171")
    kpi(c3, "False Positive (False Alarm)", f"{FP:,}", f"{FP/(FP+TN):.1%} dari pelanggan puas di-flag salah", "#f59e0b")
    kpi(c4, "Precision / Recall", "0,292 / 0,471", "Dari yang di-flag: 29,2% memang kecewa. Dari yang kecewa: 47,1% terdeteksi", "#a78bfa")

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_dist = st.columns(2)

    with col_cm:
        section("Confusion Matrix")
        fig = go.Figure(go.Heatmap(
            z=[[TN, FP],[FN, TP]],
            x=["Prediksi: Tidak Kecewa","Prediksi: Kecewa"],
            y=["Aktual: Tidak Kecewa","Aktual: Kecewa"],
            colorscale=[[0,"#0f172a"],[0.5,"#1e3a5f"],[1,"#22d3ee"]],
            text=[[f"True Negative\n{TN:,}", f"False Positive\n{FP:,}"],
                  [f"False Negative\n{FN:,}", f"True Positive\n{TP:,}"]],
            texttemplate="<b>%{text}</b>",
            textfont=dict(size=13, color="white"),
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>Jumlah: %{z:,}<extra></extra>"
        ))
        T(fig, height=280, margin=dict(l=0,r=0,t=10,b=10))
        st.plotly_chart(fig, width="stretch")

        section("Distribusi Skor Prediksi per Kategori Error",
                "FN dan TN memiliki distribusi probabilitas yang sangat mirip — "
                "itulah mengapa model sulit membedakannya")
        prob_data = [
            ("TP (Kecewa, Terdeteksi)", 0.527, 0.571, 0.671, "#22d3ee"),
            ("FN (Kecewa, Terlewat)",   0.404, 0.433, 0.464, "#f87171"),
            ("FP (Puas, Di-flag salah)",0.516, 0.543, 0.592, "#f59e0b"),
        ]
        fig = go.Figure()
        for name, q1, med, q3, color in prob_data:
            fig.add_trace(go.Box(
                name=name, q1=[q1], median=[med], q3=[q3],
                lowerfence=[max(0,q1-0.05)], upperfence=[min(1,q3+0.05)],
                marker_color=color, line_color=color,
                fillcolor=f"rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.2)"
            ))
        fig.add_vline(x=0.4955, line_color="#ffffff", line_dash="dot",
                      annotation_text="Threshold (0.4955)",
                      annotation_font=dict(color="#f1f5f9", size=10))
        T(fig, height=260, showlegend=False,
          xaxis_title="Probabilitas Prediksi Model",
          xaxis=dict(**THEME["xaxis"], range=[0.35,0.75], tickformat=".2f"),
          margin=dict(l=0,r=0,t=10,b=40))
        st.plotly_chart(fig, width="stretch")

    with col_dist:
        section("FN Rate per Kategori Produk",
                "FN Rate = % pelanggan kecewa yang TIDAK terdeteksi model. "
                "Merah = model sering gagal di kategori ini")
        fn_cat = pd.DataFrame({
            "Kategori": ["Olahraga & Hobi","Mainan","Peralatan Kecil","Koper & Tas",
                         "Barang Unik","Peralatan Rumah","Alat Konstruksi",
                         "Konstruksi Rumah","Seni","Furnitur Dekorasi",
                         "Kasur & Mandi","Furnitur Kantor"],
            "FN Rate": [0.806,0.774,0.744,0.732,0.695,0.679,0.657,0.362,0.344,0.302,0.300,0.123],
            "Jumlah Kecewa": [506,199,39,41,187,78,105,69,32,474,856,146],
        }).sort_values("FN Rate")
        avg_fn = fn_cat["FN Rate"].mean()
        colors_fn = ["#f87171" if r > avg_fn else "#22d3ee" for r in fn_cat["FN Rate"]]
        fig = go.Figure(go.Bar(
            x=fn_cat["FN Rate"]*100, y=fn_cat["Kategori"],
            orientation="h", marker_color=colors_fn,
            text=[f"{v:.1f}%" for v in fn_cat["FN Rate"]*100],
            textposition="outside", textfont=dict(size=10, color="#e2e8f0"),
            hovertemplate="<b>%{y}</b><br>FN Rate: %{x:.1f}%<extra></extra>"
        ))
        fig.add_vline(x=avg_fn*100, line_color="#fbbf24", line_dash="dot",
                      annotation_text=f"Rata-rata {avg_fn:.1%}",
                      annotation_font=dict(color="#fbbf24", size=10))
        T(fig, height=380,
          xaxis=dict(**THEME["xaxis"], ticksuffix="%", range=[0,96]),
          margin=dict(l=0,r=50,t=10,b=10))
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    c_i1, c_i2, c_i3 = st.columns(3)
    with c_i1:
        insight("Model paling efektif untuk ketidakpuasan berat",
                "Dari pelanggan yang berhasil dideteksi (TP): 52,5% adalah pemberi skor 1 (sangat kecewa). "
                "Dari yang terlewat (FN): hanya 43,3% yang skor 1. "
                "Model lebih mudah mendeteksi kecewa parah daripada kecewa ringan (skor 3).")
    with c_i2:
        insight("79% FN tidak mengalami keterlambatan",
                "79,1% dari pelanggan kecewa yang tidak terdeteksi justru mendapat barang ON-TIME "
                "atau lebih cepat. Mereka kecewa karena faktor lain: kualitas produk, "
                "deskripsi tidak sesuai, packaging, dll — di luar kemampuan model logistik kita.")
    with c_i3:
        insight("Furnitur kantor paling mudah dideteksi",
                "Model paling akurat untuk kategori berat/besar (furnitur kantor FN rate 12,3%). "
                "Di kategori ini, ketidakpuasan hampir selalu karena masalah logistik yang bisa "
                "terdeteksi dari sinyal pengiriman. Kategori ringan (olahraga, mainan) jauh lebih sulit.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — RISK SCORING TOOL
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    section("⚡ Risk Scoring Tool",
            "Estimasi risiko ketidakpuasan pelanggan berdasarkan karakteristik order. "
            "Berguna untuk prioritas monitoring operasional.")

    info("""<strong>Cara menggunakan:</strong> Isi informasi order di sisi kiri, lalu lihat estimasi risiko di kanan.
    Tool ini menggunakan pendekatan scoring berbasis temuan penelitian.
    <br><br><strong>Catatan:</strong> Untuk prediksi yang lebih presisi dalam skala produksi,
    gunakan model Logistic Regression (S2) yang tersimpan dari pipeline modeling.""")

    col_form, col_out = st.columns([1.1, 1])

    with col_form:
        section("Informasi Order")
        st.markdown("**🚚 Sinyal Pengiriman (In-Fulfillment)**")
        f1, f2 = st.columns(2)
        with f1:
            carrier_lag = st.number_input(
                "Carrier Pickup Lag (jam)",
                min_value=0.0, max_value=500.0, value=48.0, step=1.0,
                help="Waktu dari order_approved_at → order_delivered_carrier_date. "
                     "Fitur terpenting dalam model."
            )
            approval_lag = st.number_input(
                "Approval Lag (jam)",
                min_value=0.0, max_value=200.0, value=2.0, step=0.5,
                help="Waktu dari order dibuat → order_approved_at"
            )
        with f2:
            est_days = st.number_input(
                "Estimasi Hari Pengiriman", min_value=1, max_value=60, value=23,
                help="Estimasi yang diberikan platform kepada pelanggan"
            )
            n_items = st.number_input("Jumlah Item", min_value=1, max_value=20, value=1)

        st.markdown("**📦 Informasi Produk & Transaksi**")
        f3, f4 = st.columns(2)
        with f3:
            weight   = st.number_input("Berat Produk (gram)", 0, 50000, 1000, 100)
            freight  = st.number_input("Ongkos Kirim (R$)",   0.0, 500.0, 20.0, 1.0)
        with f4:
            distance = st.number_input("Perkiraan Jarak (km)", 0, 5000, 400, 50)
            same_st  = st.radio("Seller & Customer satu negara bagian?",
                                ["Ya","Tidak"], horizontal=True)

        f5, f6 = st.columns(2)
        with f5:
            category = st.selectbox("Kategori Produk", [
                "Furnitur Kantor (risiko tinggi)","Kasur & Mandi (risiko tinggi)",
                "Telepon (risiko tinggi)","Furnitur Dekorasi (risiko tinggi)",
                "Elektronik Komputer","Olahraga & Hobi",
                "Buku","Makanan & Minuman (risiko rendah)",
                "Koper & Tas (risiko rendah)","Lainnya",
            ])
        with f6:
            payment = st.selectbox("Tipe Pembayaran",
                                   ["credit_card","boleto","voucher","debit_card"])

    with col_out:
        section("Hasil Risk Assessment")

        # Scoring logic based on research findings
        score = 0.0
        reasons = []

        # Carrier pickup lag — strongest predictor
        if carrier_lag > 120:
            score += 35; reasons.append(("🔴 Carrier Pickup Lag sangat tinggi",
                                         f"{carrier_lag:.0f} jam — jauh di atas median (44 jam)", "high"))
        elif carrier_lag > 72:
            score += 22; reasons.append(("🟡 Carrier Pickup Lag tinggi",
                                         f"{carrier_lag:.0f} jam — di atas rata-rata", "med"))
        elif carrier_lag > 44:
            score += 10; reasons.append(("🟡 Carrier Pickup Lag moderat",
                                         f"{carrier_lag:.0f} jam", "low"))
        else:
            reasons.append(("🟢 Carrier Pickup Lag normal",
                            f"{carrier_lag:.0f} jam — di bawah median", "ok"))

        # Approval lag
        if approval_lag > 48:
            score += 15; reasons.append(("🔴 Approval Lag tinggi",
                                         f"{approval_lag:.0f} jam — order lama diproses", "high"))
        elif approval_lag > 12:
            score += 7; reasons.append(("🟡 Approval Lag moderat",
                                        f"{approval_lag:.0f} jam", "med"))
        else:
            reasons.append(("🟢 Approval Lag normal", f"{approval_lag:.0f} jam", "ok"))

        # Estimated delivery days
        if est_days > 28:
            score += 12; reasons.append(("🔴 Estimasi pengiriman sangat panjang",
                                         f"{est_days} hari — ekspektasi pelanggan tinggi", "high"))
        elif est_days > 20:
            score += 5; reasons.append(("🟡 Estimasi pengiriman panjang",
                                        f"{est_days} hari", "med"))
        else:
            reasons.append(("🟢 Estimasi pengiriman wajar", f"{est_days} hari", "ok"))

        # Weight & freight
        if weight > 10000 or freight > 60:
            score += 12; reasons.append(("🔴 Produk berat/ongkir mahal",
                                         f"{weight:,}g · R${freight:.0f} — high-friction order", "high"))
        elif weight > 3000 or freight > 30:
            score += 5; reasons.append(("🟡 Berat/ongkir moderat",
                                        f"{weight:,}g · R${freight:.0f}", "med"))
        else:
            reasons.append(("🟢 Berat & ongkir normal", f"{weight:,}g · R${freight:.0f}", "ok"))

        # Cross-state
        if same_st == "Tidak":
            score += 8; reasons.append(("🟡 Pengiriman lintas negara bagian",
                                        "Cross-state meningkatkan risiko keterlambatan", "med"))
        else:
            reasons.append(("🟢 Seller & customer satu negara bagian", "Risiko geografis minimal", "ok"))

        # Category
        high_risk_cats = ["Furnitur Kantor","Kasur & Mandi","Telepon","Furnitur Dekorasi"]
        low_risk_cats  = ["Buku","Makanan & Minuman","Koper & Tas"]
        cat_clean = category.split(" (")[0]
        if any(h in category for h in high_risk_cats):
            score += 8; reasons.append(("🔴 Kategori risiko tinggi",
                                        f"{cat_clean} — FN rate rendah, model efektif", "high"))
        elif any(l in category for l in low_risk_cats):
            score -= 5; reasons.append(("🟢 Kategori risiko rendah", f"{cat_clean}", "ok"))
        else:
            reasons.append(("⚪ Kategori risiko sedang", f"{cat_clean}", "ok"))

        # Normalize
        risk_pct = min(max(score, 0), 100)
        prob     = round(0.10 + (risk_pct / 100) * 0.52, 3)

        if risk_pct < 25:
            level, col_r, bg_r = "RENDAH", "#4ade80", "#052e16"
            msg = "Order ini memiliki profil risiko rendah. Tidak memerlukan intervensi khusus."
            icon = "🟢"
        elif risk_pct < 55:
            level, col_r, bg_r = "SEDANG", "#fbbf24", "#1e1a07"
            msg = "Risiko moderat. Pertimbangkan monitoring ringan atau notifikasi preventif."
            icon = "🟡"
        else:
            level, col_r, bg_r = "TINGGI", "#f87171", "#2d0000"
            msg = "Risiko tinggi. Rekomendasikan follow-up proaktif ke pelanggan atau eskalasi ke tim logistik."
            icon = "🔴"

        st.markdown(f"""
        <div style="background:{bg_r};border:2px solid {col_r};border-radius:12px;
                    padding:1.5rem;text-align:center;margin-bottom:1rem">
          <div style="font-size:2rem;margin-bottom:0.2rem">{icon}</div>
          <div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;
                      letter-spacing:0.1em;color:{col_r};margin-bottom:0.2rem">
            Tingkat Risiko</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:2rem;
                      font-weight:800;color:{col_r};line-height:1">{level}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                      color:#f1f5f9;margin:0.4rem 0 0.6rem">
            Probabilitas Kecewa: ~{prob:.0%}</div>
          <div style="font-size:0.85rem;color:#94a3b8;line-height:1.5">{msg}</div>
        </div>""", unsafe_allow_html=True)

        section("Analisis Faktor Risiko")
        color_map = {"high":"#f87171","med":"#fbbf24","low":"#fbbf24","ok":"#4ade80"}
        for title, desc, lvl in reasons:
            bg_r2 = {"high":"#2d0000","med":"#1e1a07","low":"#0f172a","ok":"#052e16"}.get(lvl,"#1e293b")
            border_r = color_map.get(lvl, "#334155")
            st.markdown(f"""
            <div style="background:{bg_r2};border-left:3px solid {border_r};
                        border-radius:0 6px 6px 0;padding:0.6rem 0.8rem;
                        margin-bottom:0.4rem">
              <div style="font-weight:600;font-size:0.85rem;color:{border_r}">{title}</div>
              <div style="font-size:0.8rem;color:#94a3b8;margin-top:0.15rem">{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — AUDIT DATA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    section("Audit Data",
            "Verifikasi kualitas dataset: kelengkapan, konsistensi, dan karakteristik distribusi.")

    col1, col2 = st.columns(2)
    with col1:
        section("Ringkasan Tabel Dataset")
        ts = load("01_table_summary.csv")
        if ts is not None:
            ts_disp = ts[["table","n_rows","n_cols","null_pct"]].copy()
            ts_disp.columns = ["Nama Tabel","Jumlah Baris","Kolom","% Data Kosong"]
            ts_disp["Jumlah Baris"] = ts_disp["Jumlah Baris"].apply(lambda x: f"{x:,}")
            ts_disp["% Data Kosong"] = ts_disp["% Data Kosong"].apply(lambda x: f"{x:.2f}%")
            st.dataframe(ts_disp, width="stretch", hide_index=True)

        section("Distribusi Skor Review")
        rev = load("01_review_score_distribution.csv")
        if rev is not None:
            label_map = {1:"1 ⭐ Sangat Buruk",2:"2 ⭐ Buruk",3:"3 ⭐ Cukup",
                         4:"4 ⭐ Baik",5:"5 ⭐ Sangat Baik"}
            rev["label"] = rev["review_score"].map(label_map)
            fig = go.Figure(go.Bar(
                x=rev["label"], y=rev["pct"],
                marker_color=["#f87171","#fb923c","#fbbf24","#34d399","#22d3ee"],
                text=[f"{v:.1f}%" for v in rev["pct"]],
                textposition="outside", textfont=dict(size=11, color="#e2e8f0"),
                hovertemplate="<b>%{x}</b><br>%{y:.1f}% dari semua review<extra></extra>"
            ))
            T(fig, height=270, yaxis=dict(**THEME["yaxis"], range=[0,65], ticksuffix="%"),
              margin=dict(l=0,r=0,t=10,b=10))
            st.plotly_chart(fig, width="stretch")
            st.markdown("""
            <div class="info-box">
            <strong>Catatan distribusi:</strong> Review skor 5 mendominasi (57,8%), diikuti skor 4 (19,3%).
            Skor 1–3 (broad dissatisfaction) mencakup 22,9% review.
            Ini menunjukkan bahwa sebagian besar pelanggan Olist puas — namun 1 dari 5 tidak.
            </div>""", unsafe_allow_html=True)

    with col2:
        section("Kelengkapan Join Antar Tabel")
        jc = load("01_join_coverage.csv")
        if jc is not None:
            fig = go.Figure(go.Bar(
                x=jc["coverage_pct"], y=jc["join"],
                orientation="h",
                marker_color=["#22d3ee" if v >= 99.9 else "#fbbf24"
                              for v in jc["coverage_pct"]],
                text=[f"{v:.2f}%" for v in jc["coverage_pct"]],
                textposition="outside", textfont=dict(size=10, color="#e2e8f0"),
                hovertemplate="<b>%{y}</b><br>Coverage: %{x:.2f}%<extra></extra>"
            ))
            T(fig, height=260,
              xaxis=dict(**THEME["xaxis"], range=[98,101.5], ticksuffix="%"),
              margin=dict(l=0,r=60,t=10,b=10))
            st.plotly_chart(fig, width="stretch")
            st.markdown("""
            <div class="good-box">
            ✅ Semua join antar tabel memiliki coverage 100% — tidak ada data yang hilang saat penggabungan.
            </div>""", unsafe_allow_html=True)

        section("Status Order")
        status = load("01_order_status_distribution.csv")
        if status is not None:
            fig = go.Figure(go.Bar(
                x=status["count"], y=status["order_status"],
                orientation="h", marker_color=COLORS[:len(status)],
                text=[f"{v:.1f}%" for v in status["pct"]],
                textposition="outside", textfont=dict(size=10, color="#e2e8f0"),
                hovertemplate="<b>%{y}</b><br>%{x:,} order (%{text})<extra></extra>"
            ))
            T(fig, height=260, xaxis=dict(**THEME["xaxis"], type="log"),
              margin=dict(l=0,r=60,t=10,b=10))
            st.plotly_chart(fig, width="stretch")
            st.markdown("""
            <div class="info-box">
            <strong>Catatan:</strong> 97% order berstatus <em>delivered</em>.
            Penelitian ini fokus hanya pada order delivered (95.568 order) karena
            hanya order terkirim yang memiliki review score yang valid untuk dijadikan target.
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    section("Data Kosong per Kolom (Hanya Kolom dengan Missing Values)")
    null_r = load("01_null_report.csv")
    if null_r is not None:
        null_r.columns = ["Tabel","Kolom","Jumlah Kosong","% Kosong"]
        st.dataframe(
            null_r.style.background_gradient(subset=["% Kosong"], cmap="YlOrRd", vmin=0, vmax=90)
                        .format({"% Kosong":"{:.2f}%","Jumlah Kosong":"{:,}"}),
            width="stretch", hide_index=True
        )
        st.markdown("""
        <div class="info-box">
        <strong>Interpretasi:</strong>
        Kolom dengan missing values terbanyak adalah teks komentar review (58,7% dan 88,3% kosong) —
        ini wajar karena pelanggan tidak diwajibkan mengisi komentar.
        Kolom kritis untuk model (fitur pengiriman, harga, berat) hampir tidak ada yang kosong.
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — INFO PENELITI
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:

    # ── Header halaman ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 60%,#162032 100%);
                border:1px solid #334155;border-radius:12px;
                padding:2rem 2.5rem;margin-bottom:1.5rem;text-align:center">
      <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.18em;
                  text-transform:uppercase;color:#22d3ee;margin-bottom:0.6rem">
        Tugas Capstone · Data Visualization and Business Intelligence
      </div>
      <div style="font-size:1.4rem;font-weight:800;color:#f1f5f9;
                  line-height:1.4;max-width:720px;margin:0 auto 0.5rem">
        Prediksi Risiko Ketidakpuasan Pelanggan Sebelum Pengiriman:
        Risk Scoring Berbasis Sinyal Fulfillment dan Logistik pada E-Commerce
      </div>
      <div style="font-size:0.85rem;color:#64748b;margin-top:0.4rem">
        Program Studi PJJ Informatika &nbsp;·&nbsp; Universitas AMIKOM Yogyakarta
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_team, col_info = st.columns([1.1, 1])

    # ── Tim Peneliti ────────────────────────────────────────────────────────
    with col_team:
        section("Tim Peneliti")

        researchers = [
            ("Agus Suwandi",   "Ketua Tim"),
            ("Kiki Haerani",   "Anggota"),
            ("Bagus Satria",   "Anggota"),
        ]
        for i, (name, role) in enumerate(researchers):
            initials = "".join(w[0] for w in name.split())
            accent_colors = ["#22d3ee", "#a78bfa", "#34d399"]
            color = accent_colors[i % len(accent_colors)]
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:1rem;
                        background:#1e293b;border:1px solid #334155;
                        border-radius:10px;padding:1rem 1.3rem;
                        margin-bottom:0.7rem">
              <div style="width:52px;height:52px;border-radius:50%;
                          background:{color}22;border:2px solid {color};
                          display:flex;align-items:center;justify-content:center;
                          font-family:monospace;font-size:1.1rem;font-weight:800;
                          color:{color};flex-shrink:0">{initials}</div>
              <div>
                <div style="font-weight:700;font-size:1rem;
                            color:#f1f5f9">{name}</div>
                <div style="font-size:0.8rem;color:#64748b;
                            margin-top:0.1rem">{role} · PJJ Informatika</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Pembimbing")
        st.markdown("""
        <div style="display:flex;align-items:center;gap:1rem;
                    background:#1e293b;border:1px solid #f59e0b44;
                    border-left:4px solid #f59e0b;
                    border-radius:0 10px 10px 0;padding:1rem 1.3rem">
          <div style="width:52px;height:52px;border-radius:50%;
                      background:#f59e0b22;border:2px solid #f59e0b;
                      display:flex;align-items:center;justify-content:center;
                      font-family:monospace;font-size:1.1rem;font-weight:800;
                      color:#f59e0b;flex-shrink:0">AS</div>
          <div>
            <div style="font-weight:700;font-size:1rem;color:#f1f5f9">
              Dr. Andi Sunyoto, M.Kom</div>
            <div style="font-size:0.8rem;color:#64748b;margin-top:0.1rem">
              Dosen Pembimbing · Universitas AMIKOM Yogyakarta</div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Informasi Studi")
        meta = [
            ("Universitas",    "Universitas AMIKOM Yogyakarta"),
            ("Program Studi",  "PJJ Informatika"),
            ("Mata Kuliah",    "Data Visualization and Business Intelligence"),
            ("Jenis Tugas",    "Capstone Project"),
            ("Dataset",        "Olist Brazilian E-Commerce Public Dataset"),
            ("Periode Data",   "September 2016 – Oktober 2018"),
            ("Tahun",          "2025 / 2026"),
        ]
        for label, value in meta:
            st.markdown(f"""
            <div style="display:flex;gap:0.8rem;padding:0.5rem 0;
                        border-bottom:1px solid #1e293b">
              <div style="min-width:130px;font-size:0.8rem;font-weight:600;
                          color:#64748b;text-transform:uppercase;
                          letter-spacing:0.05em">{label}</div>
              <div style="font-size:0.88rem;color:#e2e8f0">{value}</div>
            </div>""", unsafe_allow_html=True)

    # ── Abstrak & Kontribusi ────────────────────────────────────────────────
    with col_info:
        section("Abstrak")
        st.markdown("""
        <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;
                    padding:1.3rem 1.5rem;font-size:0.88rem;line-height:1.8;
                    color:#cbd5e1;text-align:justify">
          Ketidakpuasan pelanggan merupakan tantangan kritis dalam industri e-commerce
          yang berdampak langsung pada retensi pelanggan dan reputasi platform.
          Penelitian ini mengusulkan sebuah <strong style="color:#22d3ee">framework
          risk scoring berbasis sinyal fulfillment dan logistik</strong> untuk memprediksi
          risiko ketidakpuasan pelanggan <em>sebelum</em> barang diterima,
          menggunakan dataset publik Olist Brazilian E-Commerce.
          <br><br>
          Pendekatan dual-setting dikembangkan: model <strong style="color:#22d3ee">
          pra-pengiriman</strong> (menggunakan fitur yang tersedia saat order dibuat) dan
          model <strong style="color:#22d3ee">in-fulfillment</strong> (menambahkan sinyal
          proses pengiriman). Tiga algoritma klasifikasi — Logistic Regression,
          Random Forest, dan LightGBM — dievaluasi pada tiga skenario target,
          dengan temporal split ketat untuk mencegah kebocoran data.
          <br><br>
          Uji signifikansi DeLong mengkonfirmasi bahwa penambahan sinyal in-fulfillment
          meningkatkan AUC secara signifikan (Δ = +0,042, p &lt; 0,001).
          Logistic Regression terbukti sebagai model terbaik (AUC = 0,627),
          konsisten dengan temuan bahwa hubungan antara sinyal logistik dan ketidakpuasan
          bersifat cukup linear. Analisis pelanggaran ekspektasi menunjukkan asimetri
          loss-aversion: keterlambatan 5 hari menghasilkan dissatisfaction rate 85,9%,
          dibandingkan hanya 16,6% saat barang tiba 5 hari lebih cepat (rasio 5,2×),
          konsisten dengan <em>Expectation-Disconfirmation Theory</em> (Oliver, 1980).
          <br><br>
          Simulasi intervensi menunjukkan bahwa dengan memeriksa ulang 10% order
          berisiko tertinggi, sebanyak 20,4% pelanggan yang berpotensi kecewa dapat
          diidentifikasi lebih awal — memberikan <strong style="color:#22d3ee">
          lift 2,0× dibandingkan pemilihan acak</strong>.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Kontribusi Penelitian")
        contributions = [
            ("Dual-Setting Risk Scoring",
             "Framework prediksi dua tahap (pra-pengiriman & in-fulfillment) "
             "dengan leakage boundary yang eksplisit dan terverifikasi."),
            ("Validasi Statistik (DeLong Test)",
             "Gain sinyal in-fulfillment dibuktikan signifikan secara statistik "
             "(p < 0,001) pada ketiga algoritma yang diuji."),
            ("Grounding Teori EDT",
             "Temuan asimetri pelanggaran ekspektasi diinterpretasi dalam kerangka "
             "Expectation-Disconfirmation Theory dan Loss Aversion."),
            ("Dekomposisi Rantai Pengiriman",
             "Fase last-mile (carrier→pelanggan) terbukti 2,2× lebih kritis "
             "dibanding fase seller dalam mempengaruhi ketidakpuasan."),
            ("Simulasi Intervensi Operasional",
             "Risk score diterjemahkan menjadi prioritas intervensi kuantitatif "
             "dengan analisis lift dan cumulative gain."),
        ]
        for i, (title, body) in enumerate(contributions):
            accent = ["#22d3ee","#a78bfa","#34d399","#f59e0b","#f87171"][i]
            st.markdown(f"""
            <div style="display:flex;gap:0.8rem;margin-bottom:0.65rem;
                        background:#1e293b;border-radius:8px;
                        padding:0.8rem 1rem;border-left:3px solid {accent}">
              <div style="font-size:1rem;flex-shrink:0">
                {"🔬📊🧠🚚🎯"[i]}</div>
              <div>
                <div style="font-weight:700;font-size:0.88rem;
                            color:#f1f5f9;margin-bottom:0.2rem">{title}</div>
                <div style="font-size:0.82rem;color:#94a3b8;
                            line-height:1.5">{body}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Referensi Utama")
        refs = [
            ("Oliver, R.L. (1980)",
             "A cognitive model of the antecedents and consequences of satisfaction decisions. "
             "<em>Journal of Marketing Research</em>, 17(4), 460–469."),
            ("Kahneman, D. & Tversky, A. (1979)",
             "Prospect theory: An analysis of decision under risk. "
             "<em>Econometrica</em>, 47(2), 263–291."),
            ("DeLong, E.R. et al. (1988)",
             "Comparing the areas under two or more correlated ROC curves: "
             "A nonparametric approach. <em>Biometrics</em>, 44(3), 837–845."),
            ("Olist (2018)",
             "Brazilian E-Commerce Public Dataset by Olist. "
             "<em>Kaggle</em>. https://www.kaggle.com/olistbr/brazilian-ecommerce"),
        ]
        for ref, text in refs:
            st.markdown(f"""
            <div style="padding:0.55rem 0;border-bottom:1px solid #1e293b;
                        font-size:0.83rem;line-height:1.6">
              <span style="color:#22d3ee;font-weight:600">{ref}</span>
              <span style="color:#94a3b8"> — {text}</span>
            </div>""", unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:2rem;padding:1rem 1.5rem;background:#0f172a;
            border:1px solid #1e293b;border-radius:8px;
            display:flex;justify-content:space-between;align-items:center;
            flex-wrap:wrap;gap:0.5rem">
  <div style="font-size:0.78rem;color:#475569">
    <strong style="color:#64748b">Dataset:</strong> Olist Brazilian E-Commerce (2016–2018) ·
    <strong style="color:#64748b">Model Utama:</strong> Logistic Regression S2 In-Fulfillment ·
    <strong style="color:#64748b">AUC:</strong> 0,627 · DeLong p&lt;0,001
  </div>
  <div style="font-size:0.75rem;color:#334155;font-style:italic">
    Referensi: Oliver (1980) EDT · Kahneman &amp; Tversky (1979) Prospect Theory
  </div>
</div>
""", unsafe_allow_html=True)
