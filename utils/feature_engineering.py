import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
import config

@st.cache_data
def prepare_dataset(df):
    """Membersihkan dataset dari nilai kosong pada teks atau sentimen."""
    df_clean = df.dropna(subset=['clean_text', 'inset_sentiment']).copy()
    return df_clean

@st.cache_resource
def prepare_tfidf(text_corpus):
    """Mengekstraksi fitur TF-IDF dari korpus teks."""
    tfidf = TfidfVectorizer(
        max_features=config.MAX_FEATURES, 
        ngram_range=config.NGRAM_RANGE
    )
    X_tfidf = tfidf.fit_transform(text_corpus)
    return X_tfidf, tfidf

@st.cache_resource
def prepare_count_vectorizer(text_corpus):
    """Mengekstraksi fitur Bag of Words dari korpus teks (untuk LDA/Topic Modeling)."""
    cv = CountVectorizer(
        max_features=config.MAX_FEATURES, 
        ngram_range=config.NGRAM_RANGE
    )
    X_cv = cv.fit_transform(text_corpus)
    return X_cv, cv

@st.cache_data
def prepare_train_test(_X, y):
    """Membagi data menjadi Training dan Testing (di-cache)."""
    return train_test_split(
        _X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )

@st.cache_resource
def prepare_svd(_X_tfidf):
    """Reduksi dimensi SVD untuk visualisasi Clustering (K-Means scatter plot)."""
    svd = TruncatedSVD(
        n_components=config.SVD_COMPONENTS, 
        random_state=config.RANDOM_STATE
    )
    X_svd = svd.fit_transform(_X_tfidf)
    return X_svd, svd