import streamlit as st
import pandas as pd
from utils.theme import set_page_config
from utils.helper import load_dataset
from utils.feature_engineering import prepare_dataset, prepare_tfidf, prepare_count_vectorizer, prepare_svd
# PASTIKAN get_cluster_titles DI-IMPORT DI SINI
from utils.analytics_cluster import run_kmeans, find_best_cluster, get_cluster_titles, cluster_keywords, prepare_cluster_dataframe
from utils.plot_cluster import plot_cluster_metrics, plot_svd_scatter, plot_cluster_distribution
from utils.analytics_topic import run_lda, get_lda_topics, run_bertopic, generate_wordcloud, generate_ngrams
from utils.plot_topic import plot_ngrams, plot_bertopic_barchart
import config

set_page_config(page_title="Clustering & Topic Modeling", layout="wide")

def main():
    st.title("🗂️ Clustering & Topic Modeling")
    st.write("Mengelompokkan data tanpa pengawasan (Unsupervised Learning) untuk menemukan narasi tersembunyi dari pelanggan.")
    
    try:
        df_raw = load_dataset()
        df = prepare_dataset(df_raw)
        docs = df['clean_text'].tolist()
        
        tab1, tab2, tab3 = st.tabs(["K-Means Clustering", "LDA Topic Modeling", "BERTopic & N-Grams"])
        
        # --- TAB 1: K-MEANS CLUSTERING ---
        with tab1:
            st.subheader("Segmentasi berbasis K-Means")
            
            with st.spinner("Mengekstraksi fitur TF-IDF & SVD..."):
                X_tfidf, tfidf_vectorizer = prepare_tfidf(docs)
                X_svd, svd_model = prepare_svd(X_tfidf)
            
            with st.expander("📊 Evaluasi Metrik (Elbow, Silhouette, Davies-Bouldin)", expanded=False):
                with st.spinner("Menghitung metrik cluster (Mungkin memakan waktu)..."):
                    metrics_df = find_best_cluster(X_tfidf)
                    st.plotly_chart(plot_cluster_metrics(metrics_df), use_container_width=True)
                    st.info("Pilih jumlah Cluster (K) terbaik berdasarkan patahan (Elbow) tertinggi dan Silhouette terbesar.")

            k = st.slider("Pilih Jumlah Cluster (K)", min_value=config.MIN_CLUSTER, max_value=config.MAX_CLUSTER, value=config.DEFAULT_CLUSTER)
            
            with st.spinner(f"Melatih K-Means dengan K={k}..."):
                kmeans_model = run_kmeans(X_tfidf, k)
                
            # BUAT JUDUL TOPIK DINAMIS
            cluster_titles = get_cluster_titles(kmeans_model, tfidf_vectorizer, n_words=3)
                
            # Plot
            df_cluster = prepare_cluster_dataframe(df, kmeans_model.labels_, X_svd, cluster_titles)
            
            c1, c2 = st.columns([1.5, 1])
            with c1:
                st.plotly_chart(plot_svd_scatter(df_cluster), use_container_width=True)
            with c2:
                st.plotly_chart(plot_cluster_distribution(df_cluster), use_container_width=True)
                
            st.markdown("**Top Keywords Tiap Cluster**")
            kw_df = cluster_keywords(kmeans_model, tfidf_vectorizer, cluster_titles, n_words=10)
            st.dataframe(kw_df, use_container_width=True)
            
            csv_cluster = df_cluster.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Hasil Clustering (CSV)", csv_cluster, "kmeans_result.csv", "text/csv")
            
        # --- TAB 2: LDA TOPIC MODELING ---
        with tab2:
            st.subheader("Latent Dirichlet Allocation (LDA)")
            
            with st.spinner("Mengekstraksi CountVectorizer & Melatih LDA..."):
                X_cv, cv = prepare_count_vectorizer(docs)
                lda_model = run_lda(X_cv, n_topics=config.LDA_TOPICS)
            
            # BUAT JUDUL TOPIK DINAMIS
            lda_titles = get_lda_titles(lda_model, cv, n_words=3)
            
            st.write(f"Distribusi **Top 15 Keywords** untuk {config.LDA_TOPICS} Topik")
            
            # MENGGUNAKAN JUDUL DINAMIS
            lda_topics_df = get_lda_topics(lda_model, cv, lda_titles, n_words=15)
            
            st.dataframe(lda_topics_df.style.background_gradient(cmap='Greens', axis=None), use_container_width=True)
            
        # --- TAB 3: BERTOPIC & N-GRAMS ---
        with tab3:
            st.subheader("Deteksi Topik dengan Transformer (BERTopic)")
            
            run_bert = st.checkbox("Jalankan BERTopic (Membutuhkan komputasi tinggi, klik untuk mengeksekusi)")
            if run_bert:
                with st.spinner("Mengunduh/Memuat model SentenceTransformer & Melatih BERTopic..."):
                    topic_model, topics = run_bertopic(docs)
                    
                c_info, c_chart = st.columns([1, 2])
                with c_info:
                    st.write("**Topik yang Terbentuk:**")
                    freq = topic_model.get_topic_info()
                    st.dataframe(freq[['Topic', 'Count', 'Name']].head(10))
                with c_chart:
                    fig_bert = plot_bertopic_barchart(topic_model)
                    if fig_bert:
                        st.plotly_chart(fig_bert, use_container_width=True)
                        
            st.markdown("---")
            st.subheader("Analisis N-Grams & WordCloud")
            
            c_wc, c_ng = st.columns([1, 1])
            with c_wc:
                st.write("**WordCloud Keseluruhan**")
                all_text = " ".join(docs)
                st.pyplot(generate_wordcloud(all_text))
                
            with c_ng:
                st.write("**Bigram (2 Kata)**")
                bigram_df = generate_ngrams(docs, n=2, top_k=10)
                st.plotly_chart(plot_ngrams(bigram_df, "Top 10 Bigram"), use_container_width=True)
                
                st.write("**Trigram (3 Kata)**")
                trigram_df = generate_ngrams(docs, n=3, top_k=10)
                st.plotly_chart(plot_ngrams(trigram_df, "Top 10 Trigram"), use_container_width=True)

    except Exception as e:
        st.error(f"Terjadi kesalahan pada modul Clustering/Topic Modeling: {e}")

if __name__ == "__main__":
    main()