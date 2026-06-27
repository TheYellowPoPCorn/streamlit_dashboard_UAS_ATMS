import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import streamlit as st
import config

__all__ = ['run_kmeans', 'find_best_cluster', 'cluster_keywords', 'prepare_cluster_dataframe']

def run_kmeans(_X, n_clusters):
    """Menjalankan K-Means Clustering."""
    kmeans = KMeans(
        n_clusters=n_clusters, 
        random_state=config.RANDOM_STATE, 
        n_init='auto'
    )
    kmeans.fit(_X)
    return kmeans

def find_best_cluster(_X, min_k=config.MIN_CLUSTER, max_k=config.MAX_CLUSTER):
    """Menghitung metrik evaluasi untuk menemukan jumlah K optimal."""
    metrics = {
        'K': [], 'Inertia': [], 'Silhouette': [], 
        'Davies_Bouldin': [], 'Calinski_Harabasz': []
    }
    
    # Konversi ke array dense jika dataset kecil untuk Davies-Bouldin, 
    # namun demi keamanan memory Streamlit, kita gunakan format aslinya atau array
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

def cluster_keywords(_model, _vectorizer, n_words=10):
    """Mengambil Top Keywords tiap cluster dari pusat K-Means (Centroid)."""
    order_centroids = _model.cluster_centers_.argsort()[:, ::-1]
    terms = _vectorizer.get_feature_names_out()
    
    keywords = {}
    for i in range(_model.n_clusters):
        keywords[f"Cluster {i}"] = [terms[ind] for ind in order_centroids[i, :n_words]]
    return pd.DataFrame(keywords)

def prepare_cluster_dataframe(df, labels, svd_coords):
    """Menggabungkan label cluster dan koordinat SVD ke dalam DataFrame utama."""
    df_c = df.copy()
    df_c['Cluster'] = [f"Cluster {label}" for label in labels]
    df_c['SVD1'] = svd_coords[:, 0]
    df_c['SVD2'] = svd_coords[:, 1]
    return df_c