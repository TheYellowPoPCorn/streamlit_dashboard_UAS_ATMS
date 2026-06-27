import streamlit as st
import pandas as pd
from utils.theme import set_page_config, render_hero_section
import config
import os

# Konfigurasi Halaman (Harus dipanggil paling awal)
set_page_config(page_title="Dashboard Utama - War Kredibilitas", layout="wide")

def check_data_availability():
    """Fungsi pembantu untuk memvalidasi keberadaan file dataset."""
    status = {"dataset": False, "network": False}
    if os.path.exists(config.DATASET_PATH):
        status["dataset"] = True
    if os.path.exists(config.NETWORK_PATH):
        status["network"] = True
    return status

def main():
    # 1. Tampilkan UI Hero Section
    render_hero_section()
    
    st.markdown("---")
    
    # 2. Deskripsi Project
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📌 Project Overview")
        st.write("""
            Dashboard Business Intelligence ini dibangun untuk mengevaluasi sentimen publik dan pola interaksi pengguna 
            terhadap raksasa ekonomi digital di Indonesia.
            
            **Fokus Analisis:**
            * 🛒 **E-Commerce:** Shopee & Tokopedia
            * 📦 **Ekspedisi Logistik:** JNE & J&T Express
            
            Gunakan navigasi di **Sidebar** untuk mengakses menu:
            - **Dashboard Utama:** Ringkasan KPI dan Metrik Agregat.
            - **Sentiment Classification:** Hasil ekstraksi *Lexicon InSet* & Prediksi Model ML (SVM, Naive Bayes, XGBoost).
            - **Clustering & Topic Modeling:** Segmentasi tweet dengan *K-Means* dan deteksi isu sentral dengan *LDA & BERTopic*.
            - **Social Network Analysis:** Pemetaan *Centrality* dan *Community Detection* dengan *NetworkX & Gephi*.
        """)
        
    with col2:
        st.subheader("⚙️ Status Sistem")
        status = check_data_availability()
        
        if status["dataset"]:
            st.success("✅ Dataset CSV Ditemukan")
        else:
            st.error(f"❌ Dataset tidak ditemukan di {config.DATASET_PATH}")
            
        if status["network"]:
            st.success("✅ Network GEXF Ditemukan")
        else:
            st.error(f"❌ Network file tidak ditemukan di {config.NETWORK_PATH}")

    st.markdown("---")
    
    # 3. Preview Dataset (Jika ada)
    st.subheader("📂 Preview Data Mentah")
    if status["dataset"]:
        try:
            df = pd.read_csv(config.DATASET_PATH)
            st.write(f"**Total Baris:** {df.shape[0]:,} | **Total Kolom:** {df.shape[1]}")
            
            with st.expander("Klik untuk melihat cuplikan dataset", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Full Dataset (CSV)",
                data=csv,
                file_name="dataset_kredibilitas.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Gagal memuat dataset: {e}")
    else:
        st.info("Silakan pastikan file `dataset.csv` sudah ada di dalam folder `data/` sebelum melanjutkan ke halaman analisis.")

if __name__ == "__main__":
    main()