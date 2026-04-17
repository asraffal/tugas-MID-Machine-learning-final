import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import joblib# type: ignore
import plotly.graph_objects as go# type: ignore
from sklearn.cluster import KMeans, AgglomerativeClustering, BisectingKMeans# type: ignore
from sklearn.metrics import silhouette_score, davies_bouldin_score# type: ignore

# ── PAGE CONFIG ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CO2 Clustering",
    page_icon="🌍",
    layout="wide",
)

# ── SIMPLE DARK CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
body, .main, [class*="css"] {
    background-color: #1a1a2e;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background-color: #16213e;
}
.block-container {
    padding-top: 2rem;
}
div[data-testid="stMetric"] {
    background: #16213e;
    border: 1px solid #0f3460;
    border-radius: 8px;
    padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)

# ── KONSTANTA ──────────────────────────────────────────────────────────
FEATURES = ['mean', 'std', 'growth', 'trend', 'mean_p1', 'mean_p2', 'mean_p3']
CLUSTER_COLORS = {
    'Emisi Rendah': '#4caf50',
    'Emisi Sedang': '#2196f3',
    'Emisi Tinggi': '#f44336',
}

# ── LOAD MODEL & DATA ──────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model  = joblib.load('model_bisecting_kmeans.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def load_data():
    return pd.read_csv('clustering_results.csv')

try:
    model, scaler = load_assets()
    df = load_data()
    loaded = True
except Exception as e:
    loaded = False
    err = str(e)

# ── SIDEBAR ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌍 CO2 Clustering")
    st.caption("Analisis Emisi Karbon Global")
    st.divider()
    page = st.radio("Halaman", [
        "📊 Performa Model",
        "📋 Hasil Clustering",
        "🔮 Prediksi Baru",
    ])
    if loaded:
        st.divider()
        st.metric("Total Negara", len(df))

# ── ERROR ──────────────────────────────────────────────────────────────
if not loaded:
    st.error("❌ File tidak ditemukan. Pastikan file berikut ada di folder yang sama:")
    st.code("model_bisecting_kmeans.pkl\nscaler.pkl\nclustering_results.csv")
    st.stop()

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — PERFORMA MODEL
# ══════════════════════════════════════════════════════════════════════
if page == "📊 Performa Model":
    st.header("📊 Performa Model Clustering")
    st.caption("Perbandingan 3 algoritma setelah tuning · target Silhouette ≥ 90%")

    X = scaler.transform(df[FEATURES].values)

    labels_km  = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42).fit_predict(X)
    labels_agg = AgglomerativeClustering(n_clusters=3, linkage='ward').fit_predict(X)
    labels_bkm = model.predict(X)

    algo_data = {
        'KMeans':         {'labels': labels_km,  'color': '#f44336'},
        'Agglomerative':  {'labels': labels_agg, 'color': '#2196f3'},
        'BisectingKMeans':{'labels': labels_bkm, 'color': '#4caf50'},
    }

    # Hitung metrik
    for name, d in algo_data.items():
        d['sil'] = silhouette_score(X, d['labels'])
        d['dbi'] = davies_bouldin_score(X, d['labels'])

    # ── Metric cards ──
    cols = st.columns(3)
    for i, (name, d) in enumerate(algo_data.items()):
        status = "✅ Tercapai" if d['sil'] >= 0.90 else "❌ Belum"
        cols[i].metric(
            label=name,
            value=f"{d['sil']*100:.2f}%",
            delta=status,
        )

    st.divider()

    # ── Chart Silhouette ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Silhouette Score")
        fig = go.Figure()
        for name, d in algo_data.items():
            fig.add_trace(go.Bar(
                x=[name], y=[round(d['sil'] * 100, 2)],
                marker_color=d['color'],
                text=f"{d['sil']*100:.2f}%",
                textposition='inside',
                name=name,
            ))
        fig.add_hline(y=90, line_dash='dash', line_color='white',
                      annotation_text='Target 90%')
        fig.update_layout(
            paper_bgcolor='#16213e', plot_bgcolor='#16213e',
            font_color='#e0e0e0', showlegend=False,
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(range=[0, 115], gridcolor='#0f3460'),
            xaxis=dict(gridcolor='#0f3460'),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Davies-Bouldin Index")
        fig2 = go.Figure()
        for name, d in algo_data.items():
            fig2.add_trace(go.Bar(
                x=[round(d['dbi'], 4)], y=[name],
                orientation='h',
                marker_color=d['color'],
                text=f"{d['dbi']:.4f}",
                textposition='inside',
                name=name,
            ))
        fig2.update_layout(
            paper_bgcolor='#16213e', plot_bgcolor='#16213e',
            font_color='#e0e0e0', showlegend=False,
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title='DBI (lebih rendah = lebih baik)', gridcolor='#0f3460'),
            yaxis=dict(gridcolor='#0f3460'),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tabel ringkas ──
    st.subheader("Tabel Perbandingan")
    rows = []
    for name, d in algo_data.items():
        rows.append({
            'Algoritma': name,
            'Silhouette (%)': f"{d['sil']*100:.2f}%",
            'DBI': f"{d['dbi']:.4f}",
            'Target ≥ 90%': '✅ Tercapai' if d['sil'] >= 0.90 else '❌ Belum',
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ── Distribusi cluster pie ──
    st.subheader("Distribusi Cluster (BisectingKMeans)")
    dist = df['label'].value_counts().reset_index()
    dist.columns = ['label', 'jumlah']

    fig_pie = go.Figure(go.Pie(
        labels=dist['label'],
        values=dist['jumlah'],
        marker=dict(colors=[CLUSTER_COLORS.get(l, '#888') for l in dist['label']]),
        textinfo='label+percent+value',
        hole=0.4,
    ))
    fig_pie.update_layout(
        paper_bgcolor='#16213e', font_color='#e0e0e0',
        height=320, margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — TABEL HASIL CLUSTERING
# ══════════════════════════════════════════════════════════════════════
elif page == "📋 Hasil Clustering":
    st.header("📋 Tabel Hasil Clustering")
    st.caption("Daftar seluruh negara dan cluster emisi CO₂-nya")

    # Filter
    col1, col2 = st.columns([2, 2])
    with col1:
        search = st.text_input("🔍 Cari negara", placeholder="Ketik nama atau kode negara...")
    with col2:
        cluster_filter = st.multiselect(
            "Filter cluster",
            options=['Emisi Rendah', 'Emisi Sedang', 'Emisi Tinggi'],
            default=['Emisi Rendah', 'Emisi Sedang', 'Emisi Tinggi'],
        )

    df_show = df.copy()
    if search:
        mask = (
            df_show['country_name'].str.contains(search, case=False, na=False) |
            df_show['country_code'].str.contains(search, case=False, na=False)
        )
        df_show = df_show[mask]
    if cluster_filter:
        df_show = df_show[df_show['label'].isin(cluster_filter)]

    # Summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ditampilkan", len(df_show))
    c2.metric("🟢 Rendah", len(df_show[df_show['label'] == 'Emisi Rendah']))
    c3.metric("🔵 Sedang",  len(df_show[df_show['label'] == 'Emisi Sedang']))
    c4.metric("🔴 Tinggi",  len(df_show[df_show['label'] == 'Emisi Tinggi']))

    # Tabel
    display_cols = {
        'country_code': 'Kode',
        'country_name': 'Negara',
        'label':        'Cluster',
        'mean':         'Rata-rata (kt)',
        'std':          'Std Dev (kt)',
        'growth':       'Pertumbuhan (kt)',
        'trend':        'Tren (kt/thn)',
    }
    df_disp = df_show[list(display_cols.keys())].rename(columns=display_cols).copy()
    for col in ['Rata-rata (kt)', 'Std Dev (kt)', 'Pertumbuhan (kt)', 'Tren (kt/thn)']:
        df_disp[col] = df_disp[col].apply(lambda x: f"{x:,.0f}")

    st.dataframe(df_disp, hide_index=True, use_container_width=True, height=420)

    # Top 10
    st.subheader("Top 10 Negara Emisi Tertinggi")
    top10 = df.nlargest(10, 'mean').copy()
    fig_bar = go.Figure(go.Bar(
        x=top10['mean'],
        y=top10['country_name'],
        orientation='h',
        marker_color=[CLUSTER_COLORS.get(l, '#888') for l in top10['label']],
        text=top10['mean'].apply(lambda x: f"{x:,.0f} kt"),
        textposition='outside',
    ))
    fig_bar.update_layout(
        paper_bgcolor='#16213e', plot_bgcolor='#16213e',
        font_color='#e0e0e0', height=380,
        margin=dict(l=0, r=80, t=0, b=0),
        xaxis=dict(title='Rata-rata Emisi CO₂ (kt)', gridcolor='#0f3460'),
        yaxis=dict(autorange='reversed', gridcolor='#0f3460'),
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDIKSI NEGARA BARU
# ══════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediksi Baru":
    st.header("🔮 Prediksi Cluster Negara Baru")
    st.caption("Masukkan data emisi CO₂ untuk memprediksi kelompok emisi suatu negara")

    col_input, col_out = st.columns(2, gap="large")

    with col_input:
        st.subheader("Input Data (satuan: kt)")
        country_name = st.text_input("Nama negara (opsional)")

        c1, c2 = st.columns(2)
        with c1:
            mean_v   = st.number_input("Rata-rata emisi",    min_value=0.0, value=50000.0, step=1000.0, format="%.0f")
            growth_v = st.number_input("Pertumbuhan total",  value=10000.0, step=1000.0,  format="%.0f")
            mean_p1  = st.number_input("Rata-rata periode 1",min_value=0.0, value=40000.0, step=1000.0, format="%.0f")
            mean_p3  = st.number_input("Rata-rata periode 3",min_value=0.0, value=60000.0, step=1000.0, format="%.0f")
        with c2:
            std_v    = st.number_input("Standar deviasi",    min_value=0.0, value=15000.0, step=1000.0, format="%.0f")
            trend_v  = st.number_input("Tren per tahun",     value=500.0,   step=100.0,   format="%.1f")
            mean_p2  = st.number_input("Rata-rata periode 2",min_value=0.0, value=50000.0, step=1000.0, format="%.0f")

        predict = st.button("🔍 Prediksi", use_container_width=True, type="primary")

    with col_out:
        st.subheader("Hasil")

        if predict:
            inp = np.array([[mean_v, std_v, growth_v, trend_v, mean_p1, mean_p2, mean_p3]])
            inp_scaled  = scaler.transform(inp)
            cluster_id  = model.predict(inp_scaled)[0]

            mean_per = df.groupby('cluster')['mean'].mean()
            sorted_c = mean_per.sort_values().index.tolist()
            label_map = {
                sorted_c[0]: 'Emisi Rendah',
                sorted_c[1]: 'Emisi Sedang',
                sorted_c[2]: 'Emisi Tinggi',
            }
            pred_label = label_map.get(cluster_id, f'Cluster {cluster_id}')
            pred_color = CLUSTER_COLORS.get(pred_label, '#888')

            emoji = {'Emisi Rendah': '🟢', 'Emisi Sedang': '🔵', 'Emisi Tinggi': '🔴'}.get(pred_label, '⚪')
            desc  = {
                'Emisi Rendah': 'Negara dengan aktivitas industri terbatas atau negara kecil.',
                'Emisi Sedang': 'Negara berkembang dengan pertumbuhan industri moderat.',
                'Emisi Tinggi': 'Negara industri besar dengan konsumsi energi fosil tinggi.',
            }.get(pred_label, '')

            name_str = f" — {country_name}" if country_name else ""

            st.markdown(f"""
            <div style="border:1px solid {pred_color};border-radius:10px;padding:24px;text-align:center;margin-bottom:16px">
                <div style="font-size:2.5rem">{emoji}</div>
                <div style="font-size:1.4rem;font-weight:700;color:{pred_color};margin:8px 0">{pred_label}</div>
                <div style="font-size:0.82rem;color:#9ca3af">Cluster {cluster_id}{name_str}</div>
                <div style="font-size:0.88rem;color:#c0c0c0;margin-top:10px">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # Radar chart
            cluster_means = df[df['label'] == pred_label][FEATURES].mean()
            all_max = df[FEATURES].max()
            inp_norm = (inp[0] / (all_max.values + 1e-9)).tolist()
            cl_norm  = (cluster_means.values / (all_max.values + 1e-9)).tolist()
            cats = ['Mean', 'Std', 'Growth', 'Trend', 'P1', 'P2', 'P3']

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=inp_norm + [inp_norm[0]], theta=cats + [cats[0]],
                fill='toself', line=dict(color=pred_color, width=2),
                name='Input',
            ))
            fig_r.add_trace(go.Scatterpolar(
                r=cl_norm + [cl_norm[0]], theta=cats + [cats[0]],
                fill='toself', line=dict(color='#888', width=1.5, dash='dot'),
                name=f'Rata-rata {pred_label}',
            ))
            fig_r.update_layout(
                polar=dict(
                    bgcolor='#16213e',
                    radialaxis=dict(visible=True, range=[0,1], gridcolor='#0f3460', color='#9ca3af'),
                    angularaxis=dict(gridcolor='#0f3460', color='#e0e0e0'),
                ),
                paper_bgcolor='#16213e', font_color='#e0e0e0',
                legend=dict(bgcolor='#16213e'),
                height=280, margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        else:
            st.info("Isi form di sebelah kiri lalu tekan tombol **Prediksi**.")

        # Statistik referensi
        st.subheader("Statistik Referensi per Cluster")
        ref = df.groupby('label')[['mean', 'std', 'growth', 'trend']].mean().round(0)
        ref.columns = ['Rata-rata', 'Std Dev', 'Pertumbuhan', 'Tren']
        st.dataframe(ref.style.format("{:,.0f}"), use_container_width=True)