import networkx as nx
import pandas as pd
import streamlit as st
import config

__all__ = ['load_graph', 'network_summary', 'degree_centrality', 'betweenness_centrality', 'eigenvector_centrality', 'pagerank', 'community_detection']

def load_graph(file_path):
    """Memuat file graf Gephi (.gexf) menjadi objek NetworkX."""
    try:
        G = nx.read_gexf(file_path)
        return G
    except Exception as e:
        st.error(f"Gagal memuat file graf: {e}")
        return None

def network_summary(_G):
    """Menghitung metrik dasar dari jaringan."""
    if _G is None: return {}
    
    num_nodes = _G.number_of_nodes()
    num_edges = _G.number_of_edges()
    density = nx.density(_G)
    
    return {
        "Nodes": num_nodes,
        "Edges": num_edges,
        "Density": round(density, 5)
    }

def _get_top_nodes(centrality_dict, top_k=config.TOP_NODE):
    """Fungsi pembantu untuk mengurutkan dan mengambil Top K nodes."""
    sorted_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return pd.DataFrame(sorted_nodes, columns=['Akun / Node', 'Score'])

def degree_centrality(_G):
    """Menghitung Degree Centrality (Aktor dengan koneksi terbanyak)."""
    dc = nx.degree_centrality(_G)
    return _get_top_nodes(dc)

@st.cache_data
def betweenness_centrality(_G):
    """Menghitung Betweenness Centrality (Aktor yang menjadi jembatan informasi)."""
    bc = nx.betweenness_centrality(_G, k=100, seed=42)
    return _get_top_nodes(bc)

def eigenvector_centrality(_G):
    """Menghitung Eigenvector Centrality (Aktor yang terhubung dengan aktor penting lainnya)."""
    try:
        ec = nx.eigenvector_centrality(_G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        # Fallback jika tidak konvergen
        ec = nx.eigenvector_centrality_numpy(_G)
    return _get_top_nodes(ec)

def pagerank(_G):
    """Menghitung PageRank (Aktor paling berpengaruh secara algoritma Google)."""
    pr = nx.pagerank(_G)
    return _get_top_nodes(pr)

def community_detection(_G):
    """Mendeteksi komunitas dalam jaringan menggunakan Clauset-Newman-Moore greedy modularity."""
    try:
        # Menggunakan undirected graph untuk community detection
        G_un = _G.to_undirected()
        communities = list(nx.community.greedy_modularity_communities(G_un))
        
        community_sizes = [{"Komunitas": f"Komunitas {i+1}", "Jumlah Akun": len(c)} for i, c in enumerate(communities)]
        df_comm = pd.DataFrame(community_sizes).sort_values(by="Jumlah Akun", ascending=False)
        return df_comm, communities
    except Exception as e:
        return pd.DataFrame(), []

