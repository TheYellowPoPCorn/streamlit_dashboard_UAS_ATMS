import plotly.express as px
import streamlit as st
import config

def plot_ngrams(ngram_df, title):
    fig = px.bar(
        ngram_df, x='Frekuensi', y='N-gram', 
        orientation='h', 
        title=title,
        color='Frekuensi',
        color_continuous_scale='Sunset'
    )
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'}, 
        template=config.PLOT_TEMPLATE,
        coloraxis_showscale=False
    )
    return fig
    
def plot_bertopic_barchart(topic_model):
    """Memanggil visualisasi bawaan dari library BERTopic."""
    try:
        if topic_model is None:
            st.warning("⚠️ Topic model is not initialized. Barchart plot skipped.")
            return None
            
        fig = topic_model.visualize_barchart(top_n_topics=6)
        fig.update_layout(template=config.PLOT_TEMPLATE, title="Top Keywords Tiap Topik (BERTopic)")
        return fig
    except Exception as e:
        st.error(f"❌ Error creating BERTopic barchart: {str(e)}")
        return None