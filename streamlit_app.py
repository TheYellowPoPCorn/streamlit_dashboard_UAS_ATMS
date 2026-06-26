import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os

# Konfigurasi Tema
st.set_page_config(page_title="War Kredibilitas Brand", layout="wide", page_icon="⚔️")

st.markdown("""
    <style>
    .header-title { font-size: 32px; font-weight: bold; color: #1E3A8A; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="header-title">⚔️ TEMA 4: WAR KREDIBILITAS BRAND: E-COMMERCE & EKSPEDISI LOGISTIK</p>', unsafe_allow_html=True)

# Dataset Management
@st.cache_data
def load_data():
    # Sesuaikan dengan nama file CSV hasil crawling Anda
    if os.path.exists("ecommerce_inset_labeled_final.csv"):
        return pd.read_csv("ecommerce_inset_labeled_final.csv")
    return pd.DataFrame()

df = load_data()

# Tab Navigasi
tab1, tab2, tab3, tab4 = st.tabs(["📊 Klasifikasi Sentimen", "🎯 Klasterisasi", "🔥 Trending Topic (LDA)", "🕸️ SNA"])

# TAB 1: KLASIFIKASI SENTIMEN
with tab1:
    st.header("📊 Distribusi Sentimen Brand")
    if not df.empty and 'brand' in df.columns:
        brand = st.selectbox("Pilih Brand:", df['brand'].unique())
        df_f = df[df['brand'] == brand]
        fig = px.pie(df_f, names='inset_sentiment', title=f"Sentimen Brand: {brand}")
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: KLASTERISASI
with tab2:
    st.header("🎯 Hasil Klasterisasi (K-Means)")
    if 'kmeans_cluster' in df.columns:
        fig = px.histogram(df, x='kmeans_cluster', color='brand')
        st.plotly_chart(fig)

# TAB 3: TRENDING TOPIC (LDA)
with tab3:
    st.header("🔥 Trending Topic Keluhan")
    if 'topic_issue' in df.columns:
        fig = px.bar(df['topic_issue'].value_counts().reset_index(), x='index', y='topic_issue')
        st.plotly_chart(fig)

# TAB 4: SNA
with tab4:
    st.header("🕸️ Social Network Analysis (SNA)")
    if os.path.exists("ecommerce.gexf"):
        G = nx.read_gexf("ecommerce.gexf")
        net = Network(height="500px", width="100%")
        net.from_nx(G.subgraph(list(G.nodes)[:100]))
        net.save_graph("net.html")
        with open("net.html", 'r', encoding='utf-8') as f:
            components.html(f.read(), height=550)