"""
=========================================================
COMPONENTS
WAR KREDIBILITAS BRAND
=========================================================
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from config import (
    BRANDS, SENTIMENT_COLORS, BRAND_COLORS, LOGO_FOLDER,
    DATASET_PATH, APP_NAME, APP_TITLE
)

# ==========================================================
# HERO SECTION
# ==========================================================

def hero_section():

    st.markdown(
        """
        <div class="hero-container">

            <div class="hero-title">

                ⚔️ WAR KREDIBILITAS BRAND

            </div>

            <div class="hero-subtitle">

                Analisis Sentimen, Clustering,
                Trending Topic,
                dan Social Network Analysis

                <br>

                E-Commerce & Ekspedisi Logistik

            </div>

            <div class="hero-divider"></div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================================
# PAGE HEADER
# ==========================================================

def page_header(title, subtitle=""):

    st.title(title)

    if subtitle:

        st.caption(subtitle)

    st.divider()


# ==========================================================
# SECTION
# ==========================================================

def section(title):

    st.subheader(title)


# ==========================================================
# KPI
# ==========================================================

def metric_row(metrics):

    cols = st.columns(len(metrics))

    for col, metric in zip(cols, metrics):

        label = metric[0]

        value = metric[1]

        delta = metric[2] if len(metric) > 2 else None

        with col:

            st.metric(

                label,

                value,

                delta=delta,

                border=True

            )


# ==========================================================
# BRAND LOGO
# ==========================================================

def brand_logo():

    logos = [

        ("Shopee", "assets/logos/shopee.png"),

        ("Tokopedia", "assets/logos/tokopedia.png"),

        ("JNE", "assets/logos/jne.png"),

        ("J&T", "assets/logos/jt.png"),

    ]

    cols = st.columns(4)

    for col, (brand, logo) in zip(cols, logos):

        with col:

            if Path(logo).exists():

                st.image(
                    logo,
                    width=90
                )

            st.caption(
                brand
            )


# ==========================================================
# DATASET INFO
# ==========================================================

def dataset_info_card(df: pd.DataFrame):

    size = round(

        df.memory_usage(deep=True).sum() / 1024,

        2

    )

    st.markdown(f"""

**Dataset**

`{DATASET_PATH}`

| Informasi | Nilai |
|-----------|------:|
| Jumlah Tweet | {len(df):,} |
| Jumlah Kolom | {df.shape[1]} |
| Ukuran | {size} KB |

""")


# ==========================================================
# QUICK NAVIGATION
# ==========================================================

def quick_navigation():

    st.markdown("### 🚀 Quick Navigation")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.page_link(

            "app.py",

            label="🏠 Dashboard"

        )

    with c2:

        st.page_link(

            "pages/1_Sentiment_Analysis.py",

            label="📊 Sentiment"

        )

    with c3:

        st.page_link(

            "pages/2_Clustering_Topic.py",

            label="🔥 Clustering & Trending Topic"

        )

    with c4:

        st.page_link(

            "pages/3_Social_Network.py",

            label="🌐 SNA"

        )


# ==========================================================
# SIDEBAR
# ==========================================================

def sidebar():

    with st.sidebar:

        st.title(APP_NAME)

        st.caption(APP_TITLE)

        st.divider()

        st.markdown("""

Dashboard ini menyajikan hasil analisis:

- 📊 Sentiment Analysis
- 🔥 Clustering & Trending Topic
- 🌐 Social Network Analysis

""")

        st.divider()

        st.caption("Version 1.0")


# ==========================================================
# FOOTER
# ==========================================================

def footer():

    st.divider()

    st.caption(

        "© 2026 | WAR KREDIBILITAS BRAND"

    )