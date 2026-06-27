import streamlit as st
from utils.theme import set_page_config

set_page_config(page_title="About Project", layout="wide")

def main():
    st.title("ℹ️ Tentang Proyek")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("War Kredibilitas Brand")
        st.subheader("Analisis Persaingan E-Commerce & Ekspedisi Logistik di Indonesia")
        
        st.markdown("""
        ### 📖 Deskripsi
        Aplikasi **Business Intelligence Dashboard** ini dibangun sebagai platform analitik komprehensif 
        untuk mengukur sentimen publik dan meneliti pola kredibilitas antara raksasa ekonomi digital: 
        **Shopee, Tokopedia, JNE, dan J&T Express**.
        
        ### 🔬 Metodologi Analisis
        Proses pengolahan data mengikuti *pipeline* ketat dari Jupyter Notebook:
        1. **Data Preprocessing & Lexicon Based:** Membersihkan teks (*clean_text*) dan memberikan label awal menggunakan **Lexicon InSet** (Indonesia Sentiment Lexicon).
        2. **Feature Engineering:** Menggunakan **TF-IDF** untuk Model Supervised dan **CountVectorizer** untuk Topic Modeling.
        3. **Supervised Learning (Klasifikasi):** Melatih tiga model Machine Learning (**SVM, Naive Bayes, XGBoost**) untuk dapat memprediksi klasifikasi teks secara otonom.
        4. **Unsupervised Learning (Clustering):** Segmentasi topik menggunakan **K-Means Clustering** yang direduksi dengan SVD, serta mendeteksi isu dominan menggunakan **LDA** dan **BERTopic**.
        5. **Social Network Analysis (SNA):** Membaca graf relasi interaksi antar pengguna Twitter (X) untuk mendeteksi *Top Influencer* (Centrality) dan mendeteksi pembentukan polarisasi (Community Detection).
        
        ### 🚀 Arsitektur Code
        Proyek ini dirancang agar **Production-Ready** dengan penerapan prinsip *Modular, DRY (Don't Repeat Yourself),* dan didukung *caching resource* agresif dari Streamlit demi performa maksimal.
        """)
        
    with col2:
        st.info("### 💻 Teknologi & Stack")
        st.markdown("""
        **Bahasa Pemrograman:**
        * Python 3.11
        
        **Framework & UI:**
        * Streamlit
        * Plotly & PyVis (Visualisasi)
        
        **Machine Learning & NLP:**
        * Scikit-Learn
        * XGBoost
        * BERTopic & SentenceTransformers
        
        **Graph & Data Processing:**
        * Pandas & NumPy
        * NetworkX
        """)
        
        st.success("### 👤 Penulis / Pengembang")
        st.markdown("""
        **[Nama Anda / Tim Anda]** Dashboard ini dibuat untuk memenuhi standardisasi Analisis Big Data tingkat lanjut.
        
        🔗 *Github Repository: [Tautan Repositori]*
        """)

if __name__ == "__main__":
    main()