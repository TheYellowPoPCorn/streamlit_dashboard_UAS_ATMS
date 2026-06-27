import plotly.express as px
import pandas as pd
import streamlit as st
import config

def plot_sentiment_pie(df):
    sentiment_counts = df['inset_sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig = px.pie(
        sentiment_counts, 
        values='Count', 
        names='Sentiment', 
        color='Sentiment',
        color_discrete_map=config.SENTIMENT_COLORS,
        hole=0.4,
        title="Distribusi Sentimen Keseluruhan"
    )
    fig.update_layout(template=config.PLOT_TEMPLATE, margin=dict(t=50, b=20, l=0, r=0))
    return fig

def plot_sentiment_bar_brand(df):
    summary = df.groupby(['Brand', 'inset_sentiment']).size().reset_index(name='Count')
    fig = px.bar(
        summary, 
        x='Brand', 
        y='Count', 
        color='inset_sentiment',
        barmode='group',
        color_discrete_map=config.SENTIMENT_COLORS,
        title="Perbandingan Sentimen Antar Brand"
    )
    fig.update_layout(template=config.PLOT_TEMPLATE)
    return fig

def plot_timeline(df):
    try:
        if 'created_at' not in df.columns:
            st.warning("⚠️ Column 'created_at' not found in dataset. Timeline plot skipped.")
            return None
        
        df_time = df.copy()
        df_time['Date'] = pd.to_datetime(df_time['created_at']).dt.date
        
        if df_time['Date'].isna().all():
            st.warning("⚠️ No valid dates found in 'created_at' column. Timeline plot skipped.")
            return None
            
        timeline = df_time.groupby(['Date', 'inset_sentiment']).size().reset_index(name='Count')
        
        fig = px.line(
            timeline, 
            x='Date', 
            y='Count', 
            color='inset_sentiment',
            color_discrete_map=config.SENTIMENT_COLORS,
            markers=True,
            title="Tren Sentimen Harian"
        )
        fig.update_layout(template=config.PLOT_TEMPLATE)
        return fig
    except Exception as e:
        st.error(f"❌ Error creating timeline plot: {str(e)}")
        return None

def plot_model_comparison(metrics_dict):
    df_metrics = pd.DataFrame(list(metrics_dict.items()), columns=['Model', 'Accuracy'])
    df_metrics['Accuracy %'] = round(df_metrics['Accuracy'] * 100, 2)
    
    fig = px.bar(
        df_metrics, 
        x='Model', 
        y='Accuracy', 
        text='Accuracy %',
        color='Model',
        title="Komparasi Akurasi Model Klasifikasi"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template=config.PLOT_TEMPLATE, yaxis=dict(range=[0, 1.1]))
    return fig
    
def plot_confusion_matrix_heatmap(cm_df, model_name):
    fig = px.imshow(
        cm_df, 
        text_auto=True, 
        color_continuous_scale='Blues' if model_name != "XGBoost" else 'Greens',
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=cm_df.columns,
        y=cm_df.index,
        title=f"Confusion Matrix: {model_name}"
    )
    fig.update_layout(template=config.PLOT_TEMPLATE)
    return fig