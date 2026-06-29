import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import streamlit as st
import config

@st.cache_resource
def run_kmeans(_X, n_clusters):
    """Menjalankan K-Means Clustering."""
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=config.RANDOM_STATE, 
        n_init='auto'
    )
    kmeans.fit(_X)
    return kmeans

@st.cache_data
def find_best_cluster(_X, min_k=config.MIN_CLUSTER, max_k=config.MAX_CLUSTER):
    """Menghitung metrik evaluasi untuk menemukan jumlah K optimal."""
    metrics = {
        'K': [], 'Inertia': [], 'Silhouette': [], 
        'Davies_Bouldin': [], 'Calinski_Harabasz': []
    }
    
    X_array = _X.toarray() if hasattr(_X, "toarray") else _X
    
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init='auto')
        labels = kmeans.fit_predict(_X)
        
        metrics['K'].append(k)
        metrics['Inertia'].append(kmeans.inertia_)
        metrics['Silhouette'].append(silhouette_score(_X, labels))
        metrics['Davies_Bouldin'].append(davies_bouldin_score(X_array, labels))
        metrics['Calinski_Harabasz'].append(calinski_harabasz_score(X_array, labels))
        
    return pd.DataFrame(metrics)

@st.cache_data
def get_cluster_titles(_model, _vectorizer, n_words=3):
    """Mengekstrak top 3 keywords untuk dijadikan Judul Topik dinamis."""
    order_centroids = _model.cluster_centers_.argsort()[:, ::-1]
    terms = _vectorizer.get_feature_names_out()
    
    titles = {}
    for i in range(_model.n_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :n_words]]
        # Membuat format judul: "Cluster 0: Kata1, Kata2, Kata3"
        titles[i] = f"Cluster {i}: {', '.join(top_words).title()}"
    return titles

@st.cache_data
def cluster_keywords(_model, _vectorizer, cluster_titles, n_words=10):
    """Mengambil Top Keywords dan memetakannya ke Judul Topik."""
    order_centroids = _model.cluster_centers_.argsort()[:, ::-1]
    terms = _vectorizer.get_feature_names_out()
    
    keywords = {}
    for i in range(_model.n_clusters):
        title = cluster_titles[i]
        keywords[title] = [terms[ind] for ind in order_centroids[i, :n_words]]
    return pd.DataFrame(keywords)

def prepare_cluster_dataframe(df, labels, svd_coords, cluster_titles):
    """Menggabungkan Judul Topik cluster dan koordinat SVD ke dalam DataFrame utama."""
    df_c = df.copy()
    # Petakan label angka (0, 1, 2) ke judul teks yang sudah dibuat
    df_c['Cluster'] = [cluster_titles[label] for label in labels]
    df_c['SVD1'] = svd_coords[:, 0]
    df_c['SVD2'] = svd_coords[:, 1]
    return df_c

@st.cache_data
def get_lda_titles(_model, _vectorizer, n_words=3):
    """Mengekstrak top 3 keywords untuk judul topik LDA."""
    titles = {}
    feature_names = _vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(_model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
        titles[topic_idx] = f"Topik {topic_idx+1}: {', '.join(top_words).title()}"
    return titles

@st.cache_data
def get_lda_topics(_model, _vectorizer, lda_titles, n_words=10):
    """Menarik keywords dari model LDA dengan judul yang sudah dipetakan."""
    topics = {}
    feature_names = _vectorizer.get_feature_names_out()
    
    for topic_idx, topic in enumerate(_model.components_):
        top_features_ind = topic.argsort()[:-n_words - 1:-1]
        title = lda_titles[topic_idx]
        topics[title] = [feature_names[i] for i in top_features_ind]
    return pd.DataFrame(topics)