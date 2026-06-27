import streamlit as st
import os
import config
from utils.theme import set_page_config
from utils.helper import load_network_graph
from utils.analytics_sna import (
    load_graph, network_summary, degree_centrality, 
    betweenness_centrality, eigenvector_centrality, pagerank, community_detection
)
from utils.plot_sna import plot_centrality_bar, render_interactive_network

set_page_config(page_title="Social Network Analysis", layout="wide")

def main():
    st.title("🌐 Social Network Analysis (SNA)")
    st.write("Menganalisis hubungan dan interaksi antar pengguna untuk mendeteksi Influencer dan Komunitas utama.")
    
    network_path = load_network_graph()
    
    if not network_path:
        st.warning(f"File jaringan tidak ditemukan di `{config.NETWORK_PATH}`. Silakan upload file `.gexf` Anda ke folder `data/`.")
        return
        
    try:
        with st.spinner("Memuat struktur jaringan (Mungkin memakan waktu untuk file besar)..."):
            G = load_graph(network_path)
            
        if G is None:
            return
            
        # Tampilkan Network Metrik Dasar
        metrics = network_summary(G)
        st.subheader("📊 Statistik Jaringan")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Nodes (Akun)", f"{metrics['Nodes']:,}")
        col2.metric("Total Edges (Interaksi)", f"{metrics['Edges']:,}")
        col3.metric("Network Density", metrics['Density'])
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["🕸️ Interactive Network", "⭐ Influencer / Centrality", "🏘️ Community Detection"])
        
        # --- TAB 1: INTERACTIVE NETWORK ---
        with tab1:
            st.subheader("Pemetaan Jaringan (Visualisasi Interaktif)")
            
            # Menambahkan Slider untuk kontrol jumlah node
            max_nodes_choice = st.select_slider(
                "Pilih jumlah maksimal node untuk ditampilkan:",
                options=[100, 200, 300, 400, 500],
                value=100
            )
            
            st.info(f"Visualisasi menampilkan {max_nodes_choice} node berdasarkan koneksi terbanyak. Gunakan scroll untuk zoom dan klik untuk drag node.")
            
            # Menggunakan variabel max_nodes_choice yang dipilih pengguna
            render_interactive_network(G, max_nodes=max_nodes_choice)
            
        # --- TAB 2: INFLUENCER & CENTRALITY ---
        with tab2:
            st.subheader("Analisis Pengaruh (Top 10 Influencer)")
            
            with st.spinner("Menghitung metrik sentralitas..."):
                df_dc = degree_centrality(G)
                df_bc = betweenness_centrality(G)
                df_ec = eigenvector_centrality(G)
                df_pr = pagerank(G)
                
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_centrality_bar(df_dc, "Degree Centrality (Aktor Terpopuler)", "Oranges"), use_container_width=True)
                st.plotly_chart(plot_centrality_bar(df_bc, "Betweenness Centrality (Aktor Jembatan Informasi)", "Purples"), use_container_width=True)
            with c2:
                st.plotly_chart(plot_centrality_bar(df_ec, "Eigenvector Centrality (Aktor di Lingkaran Penting)", "Greens"), use_container_width=True)
                st.plotly_chart(plot_centrality_bar(df_pr, "PageRank (Aktor Paling Berpengaruh)", "Reds"), use_container_width=True)

        # --- TAB 3: COMMUNITY DETECTION ---
        with tab3:
            st.subheader("Mendeteksi Klaster Komunitas Pengguna")
            
            with st.spinner("Mendeteksi komunitas dengan Modularity..."):
                df_comm, communities = community_detection(G)
                
            if not df_comm.empty:
                st.write(f"Ditemukan **{len(communities)}** komunitas berbeda dalam jaringan ini.")
                
                c_tbl, c_dtl = st.columns([1, 1.5])
                with c_tbl:
                    st.dataframe(df_comm.head(15), use_container_width=True, hide_index=True)
                with c_dtl:
                    st.write("**Anggota Komunitas Terbesar (Top 1):**")
                    top_community_members = list(communities[0])[:30] # Tampilkan max 30 akun
                    st.code(", ".join(top_community_members))
            else:
                st.info("Tidak dapat mendeteksi komunitas yang signifikan pada jaringan ini.")
                
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses SNA: {e}")

if __name__ == "__main__":
    main()