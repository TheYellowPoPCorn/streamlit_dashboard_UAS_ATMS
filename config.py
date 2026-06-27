import os

# --- APP CONFIGURATION ---
APP_NAME = "War Kredibilitas Brand"
APP_TITLE = "Analisis Sentimen, Clustering, Trending Topic & SNA"

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_FOLDER = os.path.join(ASSETS_DIR, "logos")

# File Data
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
NETWORK_PATH = os.path.join(DATA_DIR, "network.gexf")

# --- MACHINE LEARNING & NLP CONFIGURATION ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# TF-IDF Parameters
MAX_FEATURES = 1500
NGRAM_RANGE = (1, 2)

# --- CLUSTERING CONFIGURATION ---
MIN_CLUSTER = 2
MAX_CLUSTER = 10
DEFAULT_CLUSTER = 3
SVD_COMPONENTS = 2

# --- TOPIC MODELING CONFIGURATION ---
LDA_TOPICS = 5
MIN_TOPIC_SIZE = 10

# --- SOCIAL NETWORK ANALYSIS CONFIGURATION ---
TOP_NODE = 10

# --- UI & VISUALIZATION CONFIGURATION ---
PLOT_TEMPLATE = "plotly_dark"
BRAND_COLORS = {
    "Shopee": "#fb923c",
    "Tokopedia": "#42b549",
    "JNE": "#00529C",
    "J&T": "#E01F26",
    "Netral": "#808080"
}
SENTIMENT_COLORS = {
    "Positive": "#4ade80",
    "Negative": "#ff4747",
    "Neutral": "#9ca3af"
}

# --- SENTIMENT CLASSIFICATION CONFIGURATION ---
LABEL_MAP = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
LABEL_NAMES = ['Negative', 'Neutral', 'Positive']

# --- BRANDS CONFIGURATION ---
BRANDS = ["Shopee", "Tokopedia", "JNE", "J&T"]