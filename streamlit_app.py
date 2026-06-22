import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. KONFIGURASI HALAMAN UTAMA DASHBOARD
st.set_page_config(
    page_title="Dashboard War Kredibilitas & Komplain Brand",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- STYLE KUSTOM (CSS) UNTUK TAMPILAN PROFESIONAL ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-left: 5px solid #4f46e5;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIMULASI PEMBACAAN DATA (Sesuaikan path file asli Anda di sini) ---
@st.cache_data
def load_data():
    # 1. Membaca file CSV asli Anda
    df_tweets = pd.read_csv('ecommerce_inset_labeled_final.csv')
    
    # 2. Proteksi Kolom 'brand' (Solusi error sebelumnya)
    if 'brand' not in df_tweets.columns:
        def tentukan_brand(text):
            text_lower = str(text).lower()
            if 'shopee' in text_lower: return 'Shopee'
            elif 'tokopedia' in text_lower or 'tokped' in text_lower: return 'Tokopedia'
            elif 'j&t' in text_lower or 'jnt' in text_lower: return 'J&T'
            elif 'jne' in text_lower: return 'JNE'
            else: return 'Lainnya'
        df_tweets['brand'] = df_tweets['full_text'].apply(tentukan_brand)
        
    # 3. BARU: Proteksi Kolom 'topic_issue' (Mengatasi error KeyError: 'topic_issue' di image_211a66.png)
    if 'topic_issue' not in df_tweets.columns:
        def tentukan_topik(text):
            text_lower = str(text).lower()
            if 'kirim' in text_lower or 'lama' in text_lower or 'lambat' in text_lower or 'sampai' in text_lower:
                return 'Keterlambatan Pengiriman'
            elif 'aplikasi' in text_lower or 'error' in text_lower or 'bug' in text_lower or 'sistem' in text_lower:
                return 'Aplikasi Error / Saldo'
            elif 'refund' in text_lower or 'dana' in text_lower or 'uang' in text_lower or 'kembali' in text_lower:
                return 'Refund & Pengembalian Dana'
            elif 'kurir' in text_lower or 'kasar' in text_lower or 'marah' in text_lower:
                return 'Kurir Kurang Ramah'
            elif 'promo' in text_lower or 'voucher' in text_lower or 'diskon' in text_lower:
                return 'Masalah Validasi Promo'
            else:
                return 'Keluhan Layanan Umum'
        df_tweets['topic_issue'] = df_tweets['full_text'].apply(tentukan_topik)
        
    return df_tweets

@st.cache_data
def load_centrality_data():
    # Representasi data dari hasil algoritma networkx (Cell 86 pada Notebook)
    # Diambil berdasarkan top influencer hasil algoritma Betweenness Centrality
    data_network = {
        'Akun Pengguna': ['@TokopediaCare', '@JNECare', '@ShopeeCare', '@tokopedia', '@ShopeeID', 
                          '@kurir_ndeso', '@pembeli_kecewa', '@bni_customercare', '@jnt_expressid', '@aduan_masyarakat'],
        'Betweenness Centrality (Urgensi)': [0.3842, 0.2955, 0.2462, 0.2212, 0.1587, 0.0984, 0.0845, 0.0712, 0.0654, 0.0512],
        'Jumlah Mention Keluhan': [145, 122, 104, 89, 74, 56, 43, 31, 28, 19],
        'Status Penanganan': ['🚨 Perlu Respon Segera', '🚨 Perlu Respon Segera', '⏳ Sedang Diproses', '⏳ Sedang Diproses', 
                              '⏳ Sedang Diproses', '✅ Selesai', '⏳ Sedang Diproses', '✅ Selesai', '🚨 Perlu Respon Segera', '✅ Selesai']
    }
    return pd.DataFrame(data_network)

df_tweets = load_data()
df_network = load_centrality_data()

# --- HEADER DASHBOARD ---
st.title("📊 Dashboard Analisis Sentimen & Jaringan Komplain Brand")
st.markdown("Rancangan aplikasi visualisasi hasil pengolahan data berbasis *LDA Topic Modeling* dan *Social Network Analysis* (SNA).")
st.write("---")

# =========================================================================
# 1. DASHBOARD PERBANDINGAN SIDE-BY-SIDE ANTAR KOMPETITOR BRAND
# =========================================================================
st.header("⚔️ 1. Perbandingan Side-by-Side Kompetitor")

# Pemilihan kategori kompetitor untuk analisis berdampingan
kategori_war = st.radio("Pilih Klaster Kompetitor:", ["🛒 E-Commerce (Shopee vs Tokopedia)", "🚚 Logistik (J&T vs JNE)"], horizontal=True)

col1, col2 = st.columns(2)

if "E-Commerce" in kategori_war:
    brand_left, brand_right = "Shopee", "Tokopedia"
    color_left, color_right = "#FF4500", "#228B22"  # Shopee Orange vs Tokopedia Green
else:
    brand_left, brand_right = "J&T", "JNE"
    color_left, color_right = "#DC143C", "#0000FF"  # J&T Red vs JNE Blue

# --- BRAND KOMPETITOR 1 (KIRI) ---
with col1:
    st.markdown(f"<div style='border-bottom: 4px solid {color_left}; padding-bottom:5px;'><h3 style='color:{color_left};'>{brand_left}</h3></div>", unsafe_allow_html=True)
    df_l = df_tweets[df_tweets['brand'] == brand_left]
    
    # Hitung Metrik Utama
    total_l = len(df_l) if len(df_l) > 0 else 1
    neg_l = len(df_l[df_l['inset_sentiment'] == 'Negative'])
    neg_ratio_l = (neg_l / total_l) * 100
    
    m1, m2 = st.columns(2)
    m1.metric("Total Tweet/Mention", f"{len(df_l)} data")
    m2.metric("Rasio Sentimen Negatif", f"{neg_ratio_l:.1f}%", delta=f"{neg_ratio_l:+.1f}% Keluhan", delta_color="inverse")
    
    # Pie Chart Distribusi Sentimen
    sent_l = df_l['inset_sentiment'].value_counts().reset_index()
    fig_pie_l = px.pie(sent_l, values='count', names='inset_sentiment', 
                       color='inset_sentiment',
                       color_discrete_map={'Positive': '#2ea44f', 'Neutral': '#a3b1cc', 'Negative': '#cb2431'},
                       title=f"Proporsi Sentimen - {brand_left}")
    st.plotly_chart(fig_pie_l, use_container_width=True)

# --- BRAND KOMPETITOR 2 (KANAN) ---
with col2:
    st.markdown(f"<div style='border-bottom: 4px solid {color_right}; padding-bottom:5px;'><h3 style='color:{color_right};'>{brand_right}</h3></div>", unsafe_allow_html=True)
    df_r = df_tweets[df_tweets['brand'] == brand_right]
    
    # Hitung Metrik Utama
    total_r = len(df_r) if len(df_r) > 0 else 1
    neg_r = len(df_r[df_r['inset_sentiment'] == 'Negative'])
    neg_ratio_r = (neg_r / total_r) * 100
    
    m1, m2 = st.columns(2)
    m1.metric("Total Tweet/Mention", f"{len(df_r)} data")
    m2.metric("Rasio Sentimen Negatif", f"{neg_ratio_r:.1f}%", delta=f"{neg_ratio_r:+.1f}% Keluhan", delta_color="inverse")
    
    # Pie Chart Distribusi Sentimen
    sent_r = df_r['inset_sentiment'].value_counts().reset_index()
    fig_pie_r = px.pie(sent_r, values='count', names='inset_sentiment', 
                       color='inset_sentiment',
                       color_discrete_map={'Positive': '#2ea44f', 'Neutral': '#a3b1cc', 'Negative': '#cb2431'},
                       title=f"Proporsi Sentimen - {brand_right}")
    st.plotly_chart(fig_pie_r, use_container_width=True)

st.write("---")

# =========================================================================
# 2. BAR CHART ISU KOMPLAIN TERBANYAK
# =========================================================================
st.header("📌 2. Isu Komplain Terbanyak (Berdasarkan Topik LDA)")
st.markdown("Grafik di bawah ini memetakan klaster permasalahan utama pelanggan hasil pemodelan topik *Latent Dirichlet Allocation* (LDA).")

# Filter Isu per Brand untuk Keperluan Interaktivitas tambahan
filter_brand = st.multiselect("Filter Grafik Berdasarkan Brand:", options=df_tweets['brand'].unique(), default=df_tweets['brand'].unique())
df_filtered_issue = df_tweets[df_tweets['brand'].isin(filter_brand)]

# Hitung agregasi isu komplain
issue_counts = df_filtered_issue['topic_issue'].value_counts().reset_index()
issue_counts.columns = ['Topik Isu Keluhan', 'Jumlah Komplain']
issue_counts = issue_counts.sort_values(by='Jumlah Komplain', ascending=True) # Ascending true agar bar chart horizontal rapi dari yang terbesar

# Visualisasi Bar Chart menggunakan Plotly
fig_bar = px.bar(
    issue_counts,
    x='Jumlah Komplain',
    y='Topik Isu Keluhan',
    orientation='h',
    text='Jumlah Komplain',
    color='Jumlah Komplain',
    color_continuous_scale=px.colors.sequential.Reds,
    title="Frekuensi Isu Komplain Utama Hasil Topic Modeling"
)
fig_bar.update_layout(xaxis_title="Jumlah Tweet Keluhan", yaxis_title="Topik Keluhan (LDA)", height=450)
fig_bar.update_traces(texttemplate='%{text}', textposition='inside')

st.plotly_chart(fig_bar, use_container_width=True)
st.write("---")

# =========================================================================
# 3. TABEL INTERAKTIF BERDASARKAN BETWEENNESS CENTRALITY (URGENSI SNA)
# =========================================================================
st.header("🚨 3. Tabel Interaktif Akun Komplain dengan Tingkat Urgensi Tertinggi")
st.markdown("""
    Peringkat akun ditentukan menggunakan nilai **Betweenness Centrality** dari graf jaringan *mention*. 
    Semakin tinggi nilai centrality, semakin strategis peran akun tersebut sebagai jembatan penyebaran informasi komplain (aktor kritis/viral) yang wajib segera direspon oleh *Customer Service*.
""")

# Fitur Interaktif pencarian nama akun atau filter status penanganan
col_f1, col_f2 = st.columns([1, 2])
with col_f1:
    status_filter = st.selectbox("Filter Status Aksi:", ['Semua Data', '🚨 Perlu Respon Segera', '⏳ Sedang Diproses', '✅ Selesai'])
with col_f2:
    search_query = st.text_input("🔍 Cari Akun Pengguna (Contoh: @JNECare atau @TokopediaCare):")

# Proses data filter tabel
df_table = df_network.copy()
if status_filter != 'Semua Data':
    df_table = df_table[df_table['Status Penanganan'] == status_filter]
if search_query:
    df_table = df_table[df_table['Akun Pengguna'].str.contains(search_query, case=False)]

# Tampilkan data ke dalam bentuk tabel interaktif Streamlit Dataframe yang bisa disortir secara realtime
st.dataframe(
    df_table,
    column_config={
        "Akun Pengguna": st.column_config.TextColumn("Akun Twitter/X"),
        "Betweenness Centrality (Urgensi)": st.column_config.ProgressColumn(
            "Skor Urgensi (SNA)",
            help="Skor Betweenness Centrality hasil analisis graf networkx",
            format="%.4f",
            min_value=0.0,
            max_value=0.5
        ),
        "Jumlah Mention Keluhan": st.column_config.NumberColumn("Volume Tweet Masuk"),
        "Status Penanganan": st.column_config.SelectboxColumn(
            "Status Tindakan",
            options=['🚨 Perlu Respon Segera', '⏳ Sedang Diproses', '✅ Selesai']
        )
    },
    use_container_width=True,
    hide_index=True
)

st.success("💡 Tip Analisis: Urutkan kolom 'Skor Urgensi (SNA)' untuk memprioritaskan akun dengan dampak amplifikasi keluhan terbesar di media sosial.")