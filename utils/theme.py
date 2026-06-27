import streamlit as st
import config
import os

def load_css():
    """Membaca file CSS eksternal dari assets/css/style.css."""
    css_path = os.path.join(config.ASSETS_DIR, "css", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        # Fallback jika file CSS belum terbuat
        pass

def set_page_config(page_title="BI Dashboard", layout="wide"):
    """Konfigurasi awal halaman Streamlit dan injeksi CSS."""
    st.set_page_config(
        page_title=page_title,
        page_icon="📊",
        layout=layout,
        initial_sidebar_state="expanded"
    )
    # Memuat stylesheet dari assets
    load_css()

def render_hero_section():
    """Merender Hero Section menggunakan kelas CSS dari style.css."""
    st.markdown('<div class="hero-title">War Kredibilitas Brand</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Business Intelligence Dashboard | E-Commerce & Ekspedisi Logistik</div>', unsafe_allow_html=True)