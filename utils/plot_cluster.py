import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config

def plot_cluster_metrics(metrics_df):
    """Membuat subplot metrik (Elbow, Silhouette, Davies-Bouldin)."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Elbow Method (Inertia)", "Silhouette Score", "Davies-Bouldin Score"))
    
    fig.add_trace(go.Scatter(x=metrics_df['K'], y=metrics_df['Inertia'], mode='lines+markers', name="Inertia"), row=1, col=1)
    fig.add_trace(go.Scatter(x=metrics_df['K'], y=metrics_df['Silhouette'], mode='lines+markers', name="Silhouette", marker_color='green'), row=1, col=2)
    fig.add_trace(go.Scatter(x=metrics_df['K'], y=metrics_df['Davies_Bouldin'], mode='lines+markers', name="Davies Bouldin", marker_color='red'), row=1, col=3)
    
    fig.update_layout(template=config.PLOT_TEMPLATE, showlegend=False, title_text="Evaluasi Optimal Cluster (K)")
    return fig

def plot_svd_scatter(df):
    """Memetakan sebaran teks dalam dimensi 2D menggunakan SVD."""
    fig = px.scatter(
        df, x='SVD1', y='SVD2', 
        color='Cluster', 
        hover_data=['Brand', 'inset_sentiment'],
        title="Distribusi Cluster (SVD 2D Projection)",
        opacity=0.7
    )
    fig.update_layout(template=config.PLOT_TEMPLATE)
    return fig
    
def plot_cluster_distribution(df):
    """Visualisasi jumlah dokumen di tiap cluster."""
    dist = df['Cluster'].value_counts().reset_index()
    dist.columns = ['Cluster', 'Count']
    fig = px.bar(
        dist, x='Cluster', y='Count', 
        color='Cluster', 
        title="Jumlah Tweet per Cluster",
        text='Count'
    )
    fig.update_layout(template=config.PLOT_TEMPLATE)
    fig.update_traces(textposition='outside')
    return fig