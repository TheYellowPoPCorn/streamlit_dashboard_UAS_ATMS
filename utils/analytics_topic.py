from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import config

__all__ = ['run_lda', 'get_lda_topics', 'run_bertopic', 'generate_wordcloud', 'generate_ngrams']

def run_lda(_X_cv, n_topics=config.LDA_TOPICS):
    """Menjalankan algoritma LDA."""
    lda = LatentDirichletAllocation(
        n_components=n_topics, 
        random_state=config.RANDOM_STATE
    )
    lda.fit(_X_cv)
    return lda

def get_lda_topics(_model, _vectorizer, n_words=10):
    """Menarik keywords dari model LDA."""
    topics = {}
    for topic_idx, topic in enumerate(_model.components_):
        top_features_ind = topic.argsort()[:-n_words - 1:-1]
        feature_names = _vectorizer.get_feature_names_out()
        topics[f"Topic {topic_idx+1}"] = [feature_names[i] for i in top_features_ind]
    return pd.DataFrame(topics)

def run_bertopic(docs):
    """Menjalankan BERTopic dengan model transformer bahasa Indonesia (jika tersedia)."""
    if not BERTOPIC_AVAILABLE:
        st.warning("⚠️ BERTopic library is not installed. Please install it to use this feature.")
        return None, None
    
    try:
        # Menggunakan model paraphrase-multilingual ringan
        topic_model = BERTopic(
            language="indonesian", 
            calculate_probabilities=False, 
            min_topic_size=config.MIN_TOPIC_SIZE
        )
        topics, _ = topic_model.fit_transform(docs)
        return topic_model, topics
    except Exception as e:
        st.error(f"❌ Error running BERTopic: {str(e)}")
        return None, None

def generate_wordcloud(text):
    """Membuat gambar WordCloud dari kumpulan teks."""
    wc = WordCloud(
        width=800, height=400, 
        background_color='#1a1a1a', 
        colormap='Oranges',
        max_words=100
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a1a')
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig

def generate_ngrams(docs, n=2, top_k=15):
    """Membuat daftar kemunculan Bigram / Trigram."""
    cv = CountVectorizer(ngram_range=(n, n)).fit(docs)
    bag_of_words = cv.transform(docs)
    sum_words = bag_of_words.sum(axis=0)
    
    words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]
    
    return pd.DataFrame(words_freq, columns=['N-gram', 'Frekuensi'])

# Force reload
