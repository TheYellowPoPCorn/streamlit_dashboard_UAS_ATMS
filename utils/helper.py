import pandas as pd
import streamlit as st
import config
import os

@st.cache_data
def load_dataset():
    """Memuat dataset utama dan melakukan pra-pemrosesan ringan."""
    if not os.path.exists(config.DATASET_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan di {config.DATASET_PATH}")
        
    df = pd.read_csv(config.DATASET_PATH)
    
    # Validate required columns
    required_columns = ['clean_text', 'inset_sentiment']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset missing required columns: {missing_columns}\n"
            f"Expected columns: {required_columns}\n"
            f"Found columns: {list(df.columns)}"
        )
    
    # Pastikan format tanggal benar
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        
    # Deteksi Brand berdasarkan kata kunci pada clean_text jika kolom Brand tidak ada
    if 'Brand' not in df.columns and 'clean_text' in df.columns:
        def assign_brand(text):
            text = str(text).lower()
            if 'shopee' in text: return 'Shopee'
            elif 'tokopedia' in text or 'toped' in text: return 'Tokopedia'
            elif 'j&t' in text or 'jnt' in text: return 'J&T'
            elif 'jne' in text: return 'JNE'
            else: return 'Unknown'
            
        df['Brand'] = df['clean_text'].apply(assign_brand)
        # Hapus data yang tidak terdeteksi brand utamanya (opsional, agar fokus ke 4 brand)
        df = df[df['Brand'] != 'Unknown'].reset_index(drop=True)
        
    return df

@st.cache_data
def load_network_graph():
    """Mengecek keberadaan file Gephi (.gexf) untuk halaman SNA nantinya."""
    if not os.path.exists(config.NETWORK_PATH):
        return None
    return config.NETWORK_PATH