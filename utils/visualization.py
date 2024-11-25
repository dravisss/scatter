import streamlit as st
import umap
import hdbscan
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

# Disable Numba threading
os.environ["NUMBA_THREADING_LAYER"] = "omp"
os.environ["NUMBA_NUM_THREADS"] = "1"

def reduce_dimensionality(embeddings, n_neighbors, min_dist, n_components, metric):
    """Reduce dimensionality of embeddings using UMAP."""
    # Convert embeddings to float64
    embeddings = np.array(embeddings, dtype=np.float64)
    
    # Ensure single thread operation
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
        random_state=42,
        n_jobs=1,  # Force single thread
        low_memory=True,
        verbose=False  # Disable verbose output
    )
    return reducer.fit_transform(embeddings)

def cluster_data(embeddings, min_cluster_size, min_samples, cluster_selection_epsilon, cluster_selection_method):
    """Cluster the embeddings using HDBSCAN."""
    # Convert embeddings to float64
    embeddings = np.array(embeddings, dtype=np.float64)
    
    # Ensure single thread operation
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=1,  # Force single thread
        algorithm='generic'  # Use more stable algorithm
    )
    return clusterer.fit_predict(embeddings)

def get_cluster_colors(cluster_labels, cluster_names=None):
    """Generate colors and labels for clusters."""
    unique_clusters = sorted(list(set(cluster_labels)))
    colors = []
    hover_texts = []
    
    # Generate a color palette
    n_clusters = len(unique_clusters)
    cmap = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    color_palette = [rgb2hex(color[:3]) for color in cmap]  # Convert RGB to hex, ignoring alpha
    
    # Map clusters to colors
    cluster_to_color = {cluster: color for cluster, color in zip(unique_clusters, color_palette)}
    
    for label in cluster_labels:
        colors.append(cluster_to_color[label])
        if cluster_names and label in cluster_names:
            hover_texts.append(f"Cluster {label}: {cluster_names[label]}")
        else:
            hover_texts.append(f"Cluster {label}")
            
    return colors, hover_texts

def get_visualization_params():
    """Get visualization parameters from sidebar."""
    st.sidebar.subheader("Parâmetros UMAP")
    n_neighbors = st.sidebar.slider("Número de vizinhos", 2, 200, 15, 
                                  help="Tamanho da vizinhança local usada para aproximação do manifold")
    min_dist = st.sidebar.slider("Distância mínima", 0.0, 1.0, 0.1, 
                               help="Distância mínima efetiva entre os pontos incorporados")
    n_components = st.sidebar.slider("Número de dimensões", 2, 3, 2,
                                   help="Dimensão do espaço para incorporação")
    metric = st.sidebar.selectbox("Métrica de distância", 
                                ["euclidean", "manhattan", "cosine", "correlation"],
                                help="Métrica usada para cálculo de distância")
    
    st.sidebar.subheader("Parâmetros HDBSCAN")
    min_cluster_size = st.sidebar.slider("Tamanho mínimo do cluster", 2, 100, 5,
                                       help="Tamanho mínimo dos clusters")
    min_samples = st.sidebar.slider("Amostras mínimas", 1, 100, 5,
                                  help="Número de amostras em uma vizinhança para um ponto ser considerado central")
    cluster_selection_epsilon = st.sidebar.slider("Epsilon de seleção", 0.0, 1.0, 0.0,
                                                help="Limiar de distância para fusão de clusters")
    cluster_selection_method = st.sidebar.selectbox("Método de seleção",
                                                  ["eom", "leaf"],
                                                  help="Método usado para seleção de clusters")
    
    return {
        "umap": {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "n_components": n_components,
            "metric": metric
        },
        "hdbscan": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_epsilon": cluster_selection_epsilon,
            "cluster_selection_method": cluster_selection_method
        }
    }
