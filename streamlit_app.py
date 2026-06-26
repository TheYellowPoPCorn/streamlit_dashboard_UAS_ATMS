import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import xml.etree.ElementTree as ET

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA UTAMA
# ==========================================
st.set_page_config(
    page_title="TEMA 4: WAR KREDIBILITAS BRAND",
    page_icon="⚔️",
    layout="wide"
)

st.markdown("""
    <style>
    .brand-logo { max-height: 50px; object-fit: contain; margin-bottom: 10px; }
    .header-title { font-size: 32px; font-weight: bold; color: #1E3A8A; }
    .metric-box { padding: 15px; border-radius: 10px; background: #f8fafc; border-left: 5px solid #3b82f6; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="header-title">⚔️ TEMA 4: WAR KREDIBILITAS BRAND: E-COMMERCE & EKSPEDISI LOGISTIK</p>', unsafe_allow_html=True)
st.write("Dashboard interaktif untuk memonitor sentimen, tren keluhan, dan aktor jaringan utama pada platform e-commerce dan logistik di Indonesia.")

# ==========================================
# 2. LOGO BRAND & DATASET DOWNLOAD
# ==========================================
col_logo1, col_logo2, col_logo3, col_logo4 = st.columns(4)
with col_logo1: st.image("https://upload.wikimedia.org/wikipedia/commons/f/fe/Shopee_logo.svg", width=120)
with col_logo2: st.image("https://upload.wikimedia.org/wikipedia/commons/9/9d/Tokopedia_Logo.svg", width=150)
with col_logo3: st.image("https://upload.wikimedia.org/wikipedia/commons/9/92/Logo_JNE.svg", width=80)
with col_logo4: st.image("https://upload.wikimedia.org/wikipedia/commons/b/b9/J%26T_Express_logo.svg", width=100)

st.write("---")
st.subheader("📄 Dataset yang Digunakan")
dataset_name = "ecommerce_inset_labeled_final.csv"

@st.cache_data
def load_data():
    if os.path.exists(dataset_name):
        df = pd.read_csv(dataset_name)
    else:
        np.random.seed(42)
        df = pd.DataFrame({
            'username': [f'user{i}' for i in range(500)],
            'brand': np.random.choice(['Shopee', 'Tokopedia', 'JNE', 'J&T'], 500, p=[0.35, 0.35, 0.15, 0.15]),
            'inset_sentiment': np.random.choice(['Positive', 'Neutral', 'Negative'], 500, p=[0.2, 0.2, 0.6]),
            'kmeans_cluster': np.random.choice(['Cluster 0', 'Cluster 1', 'Cluster 2'], 500),
            'topic_issue': np.random.choice(['Pengiriman Lambat', 'Aplikasi Error', 'CS Kurang Ramah', 'Promo Gagal', 'Paket Hilang'], 500)
        })
    return df

df = load_data()

csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Dataset CSV",
    data=csv_data,
    file_name=dataset_name,
    mime='text/csv'
)
st.write("---")

# ==========================================
# 3. NAVIGASI TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Klasifikasi Sentimen", 
    "🎯 2. Klasterisasi", 
    "🔥 3. Trending Topic (LDA)", 
    "🕸️ 4. SNA (Gephi Network)"
])

# ------------------------------------------
# TAB 1: KLASIFIKASI SENTIMEN
# ------------------------------------------
with tab1:
    st.header("📊 Distribusi Klasifikasi Sentimen")
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.write("Filter berdasarkan Brand:")
        pilih_brand = st.selectbox("Pilih Brand:", ['Semua Brand'] + list(df['brand'].unique()))
        
        df_sentimen = df if pilih_brand == 'Semua Brand' else df[df['brand'] == pilih_brand]
        sentimen_count = df_sentimen['inset_sentiment'].value_counts().reset_index()
        sentimen_count.columns = ['Sentimen', 'Jumlah']
        st.dataframe(sentimen_count, use_container_width=True, hide_index=True)

    with col_b:
        fig_pie = px.pie(sentimen_count, values='Jumlah', names='Sentimen', 
                         color='Sentimen',
                         color_discrete_map={'Positive':'#2ea44f', 'Neutral':'#a3b1cc', 'Negative':'#cb2431'},
                         hole=0.4, title=f"Proporsi Sentimen - {pilih_brand}")
        st.plotly_chart(fig_pie, use_container_width=True)

# ------------------------------------------
# TAB 2: KLASTERISASI (K-MEANS)
# ------------------------------------------
with tab2:
    st.header("🎯 Hasil Klasterisasi (K-Means)")
    if 'kmeans_cluster' in df.columns:
        cluster_counts = df['kmeans_cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Klaster', 'Jumlah Tweet']
        
        fig_bar_cluster = px.bar(cluster_counts, x='Klaster', y='Jumlah Tweet',
                                 color='Klaster', title="Distribusi Data per Klaster",
                                 text='Jumlah Tweet')
        st.plotly_chart(fig_bar_cluster, use_container_width=True)
    else:
        st.warning("Kolom 'kmeans_cluster' tidak ditemukan di dataset Anda.")

# ------------------------------------------
# TAB 3: TRENDING TOPIC (LDA)
# ------------------------------------------
with tab3:
    st.header("🔥 Trending Topic Keluhan (LDA)")
    if 'topic_issue' in df.columns:
        topic_counts = df['topic_issue'].value_counts().reset_index()
        topic_counts.columns = ['Topik Keluhan', 'Jumlah']
        topic_counts = topic_counts.sort_values(by='Jumlah', ascending=True)
        
        fig_lda = px.bar(topic_counts, x='Jumlah', y='Topik Keluhan', orientation='h',
                         color='Jumlah', color_continuous_scale='Reds',
                         title="Volume Keluhan Berdasarkan Topik LDA")
        st.plotly_chart(fig_lda, use_container_width=True)
    else:
        st.warning("Kolom 'topic_issue' tidak ditemukan di dataset Anda.")

# ------------------------------------------
# TAB 4: SOCIAL NETWORK ANALYSIS (SNA)
# ------------------------------------------
with tab4:
    st.header("🕸️ Social Network Analysis (SNA)")
    st.write("Analisis jaringan aktor yang menyebarkan komplain berdasarkan file proyek **Gephi (`ecommerce.gexf`)**.")
    
    gexf_file = 'ecommerce.gexf'
    
    if os.path.exists(gexf_file):
        try:
            # SOLUSI FIX ERROR EDGE ATTRIBUTE: Bersihkan atribut edge ilegal dari file GEXF sebelum dibaca NetworkX
            tree = ET.parse(gexf_file)
            root = tree.getroot()
            
            # Cari namespace XML GEXF dinamik
            ns = {'g': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {'g': ''}
            xpath_query = './/g:edge' if ns['g'] else './/edge'
            
            # Hapus paksa elemen bawaan gephi yang bikin crash di networkx (seperti attvalues pada edge)
            for edge in root.findall(xpath_query, ns):
                for child in list(edge):
                    if 'attvalues' in child.tag:
                        edge.remove(child)
            
            # Simpan ke file temporer yang bersih
            cleaned_gexf = 'ecommerce_cleaned.gexf'
            tree.write(cleaned_cleaned_gexf := cleaned_gexf, encoding='utf-8', xml_declaration=True)
            
            # Membaca graf yang sudah dibersihkan secara aman
            G = nx.read_gexf(cleaned_cleaned_gexf)
            
            # Hitung Degree Centrality
            degree_dict = nx.degree_centrality(G)
            
            # Konversi ke DataFrame
            df_degree_all = pd.DataFrame(list(degree_dict.items()), columns=['Akun', 'Degree Centrality'])
            df_degree_all = df_degree_all.sort_values(by='Degree Centrality', ascending=False)
            
            # Ambil Top 10
            df_degree_top10 = df_degree_all.head(10)
            
            st.subheader("🏆 Top 10 Aktor Paling Berpengaruh (Degree Centrality)")
            st.dataframe(df_degree_top10, use_container_width=True, hide_index=True)
            
            # Visualisasi Jaringan Interaktif dengan PyVis
            st.subheader("🌐 Peta Jaringan Interaktif")
            st.write("Visualisasi ini difilter untuk maksimal 200 node teratas agar tidak membebani browser.")
            
            if len(G.nodes) > 200:
                top_nodes = list(df_degree_all['Akun'].head(200).values)
                G_sub = G.subgraph(top_nodes)
            else:
                G_sub = G
            
            # Render Graf
            net = Network(height="500px", width="100%", bgcolor="#ffffff", font_color="black")
            net.from_nx(G_sub)
            
            path_html = "network_map.html"
            net.save_graph(path_html)
            HtmlFile = open(path_html, 'r', encoding='utf-8')
            components.html(HtmlFile.read(), height=550)
            
            # Hapus berkas temporer setelah sukses render
            if os.path.exists(cleaned_cleaned_gexf):
                os.remove(cleaned_cleaned_gexf)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file GEXF: {e}")
    else:
        st.info("File `ecommerce.gexf` belum ditemukan di direktori Anda.")