import streamlit as st
import umap
import hdbscan
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import column
import pandas as pd

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
    
    # Generate colors using matplotlib's color map
    n_clusters = len(unique_clusters)
    if n_clusters > 0:
        cmap = plt.cm.get_cmap('tab20')  # Using tab20 for more distinct colors
        color_map = {cluster: rgb2hex(cmap(i/n_clusters)) for i, cluster in enumerate(unique_clusters)}
    else:
        color_map = {-1: '#808080'}  # Gray for noise
    
    # Generate colors and hover texts for each point
    for label in cluster_labels:
        colors.append(color_map[label])
        if cluster_names and label != -1 and label < len(cluster_names):
            hover_texts.append(f"Cluster {label}: {cluster_names[label]}")
        else:
            hover_texts.append(f"Cluster {label}")
    
    return colors, hover_texts

def create_bokeh_scatter(umap_embeddings, colors, df, text_col, desc_col, cluster_labels, cluster_names=None, cluster_info=None, is_3d=False):
    """Create an interactive Bokeh scatter plot."""
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool
    
    # Prepare data
    data = {
        'x': umap_embeddings[:, 0],
        'y': umap_embeddings[:, 1],
        'colors': colors,
        'text': df[text_col],
        'description': df[desc_col] if desc_col else [''] * len(df),
        'cluster_info': cluster_info if cluster_info else [f'Cluster {label}' for label in cluster_labels]
    }
    if is_3d and umap_embeddings.shape[1] > 2:
        data['z'] = umap_embeddings[:, 2]
    
    source = ColumnDataSource(data)
    
    # Create figure
    tools = "pan,wheel_zoom,box_zoom,reset,save"
    
    if is_3d:
        # 3D visualization logic would go here
        # Note: Bokeh doesn't support true 3D plots, would need to use alternative
        pass
    else:
        p = figure(width=800, height=600, tools=tools)
        
        # Add points
        circles = p.circle('x', 'y', source=source,
                         size=8,
                         fill_color='colors',
                         fill_alpha=0.6,
                         line_color=None)
        
        # Customize hover tool with HTML template
        hover = HoverTool(
            renderers=[circles],
            tooltips="""
                <div style="background-color: white; opacity: 0.95; border: 1px solid gray; 
                     border-radius: 5px; padding: 10px; width: 300px;">
                    <div style="word-wrap: break-word; white-space: pre-wrap;">
                        <span style="font-weight: bold;">Texto:</span> 
                        <span style="display: block; margin: 5px 0;">@text</span>
                        <span style="font-weight: bold;">Descrição:</span>
                        <span style="display: block; margin: 5px 0;">@description</span>
                        <span style="font-weight: bold;">Cluster:</span>
                        <span style="display: block;">@cluster_info{safe}</span>
                    </div>
                </div>
            """
        )
        
        p.add_tools(hover)
        
        # Customize appearance
        p.grid.grid_line_color = "gray"
        p.grid.grid_line_alpha = 0.1
        p.background_fill_color = "white"
        p.border_fill_color = "white"
        p.axis.axis_label_text_font_size = "12pt"
        p.axis.axis_label_text_font_style = "normal"
        p.title.text_font_size = "14pt"
        
        # Add axis labels
        p.xaxis.axis_label = 'UMAP Dimensão 1'
        p.yaxis.axis_label = 'UMAP Dimensão 2'
        
        return p

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
