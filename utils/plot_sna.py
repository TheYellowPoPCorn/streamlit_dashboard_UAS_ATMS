import plotly.express as px
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components
import tempfile
import config
import os
import networkx as nx

def plot_centrality_bar(df, title, color_theme='Blues'):
    """Membuat Bar Chart untuk metrik Centrality."""
    df_sorted = df.sort_values(by='Score', ascending=True)
    fig = px.bar(
        df_sorted, 
        x='Score', 
        y='Akun / Node', 
        orientation='h', 
        title=title,
        color='Score',
        color_continuous_scale=color_theme
    )
    fig.update_layout(template=config.PLOT_TEMPLATE, coloraxis_showscale=False)
    return fig

def render_interactive_network(G, max_nodes=500):
    """
    Merender graf menggunakan PyVis. 
    Dibatasi max_nodes agar browser tidak crash/lambat pada data besar.
    """
    # Membatasi ukuran graf berdasarkan degree untuk visualisasi
    if G.number_of_nodes() > max_nodes:
        degree_dict = dict(G.degree())
        top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:max_nodes]
        G_sub = G.subgraph(top_nodes)
    else:
        G_sub = G

    # Inisialisasi PyVis Network (Dark mode)
    net = Network(height='600px', width='100%', bgcolor='#1a1a1a', font_color='white', directed=nx.is_directed(G_sub))
    net.from_nx(G_sub)
    
    # Opsi physics agar tampilan lebih rapi
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04
        },
        "minVelocity": 0.75
      }
    }
    """)
    
    # Simpan ke HTML sementara lalu render di Streamlit
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
            html_path = tmp_file.name
            net.save_graph(html_path)
            
        with open(html_path, 'r', encoding='utf-8') as f:
            html_data = f.read()
            
        components.html(html_data, height=620)
        os.remove(html_path) # Clean up
    except Exception as e:
        st.error(f"Gagal merender interaktif network: {e}")