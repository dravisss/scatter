import streamlit as st
import umap
import hdbscan
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, TapTool, CDSView, BooleanFilter, IndexFilter
from bokeh.layouts import column
import pandas as pd
from adjustText import adjust_text

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
        algorithm='generic',  # Use more stable algorithm
        prediction_data=True  # Enable prediction data for probabilities
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    
    # Get probabilities for each point
    probabilities = clusterer.probabilities_
    
    return cluster_labels, probabilities

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
    from bokeh.models import ColumnDataSource, HoverTool, CustomJS, CDSView, BooleanFilter, IndexFilter

    # Prepare data
    data = {
        'x': umap_embeddings[:, 0],
        'y': umap_embeddings[:, 1],
        'colors': colors,
        'text': df[text_col],
        'description': df[desc_col] if desc_col else [''] * len(df),
        'cluster_info': cluster_info if cluster_info else [f'Cluster {label}' for label in cluster_labels],
        'cluster_label': cluster_labels,
        'index': list(range(len(cluster_labels)))
    }
    if is_3d and umap_embeddings.shape[1] > 2:
        data['z'] = umap_embeddings[:, 2]
    
    source = ColumnDataSource(data)
    
    # Create initial view with all points visible
    view = CDSView(source=source, filters=[BooleanFilter([True] * len(cluster_labels))])
    
    # Create figure
    tools = "pan,wheel_zoom,box_zoom,reset,save"
    
    if is_3d:
        # 3D visualization logic would go here
        pass
    else:
        p = figure(width=800, height=600, tools=tools)
        
        # Add points with view
        circles = p.circle('x', 'y',
                         source=source,
                         view=view,
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
                        <span style="font-weight: bold;">Título</span> 
                        <span style="display: block; margin: 5px 0;">@text</span>
                        <span style="font-weight: bold;">Narrativa</span>
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
        
        return p, source, view  # Return plot, source and view for interactivity

def create_altair_scatter(umap_embeddings, df, text_col, desc_col, cluster_labels, cluster_names=None):
    """Create an interactive Altair scatter plot with toggles."""
    import altair as alt
    import streamlit as st
    
    # Ensure umap_embeddings is a numpy array
    umap_embeddings = np.array(umap_embeddings)
    cluster_labels = np.array(cluster_labels)
    
    # Prepare cluster names
    def get_cluster_name(label):
        if label == -1:
            return 'Outliers'
        elif cluster_names and label in cluster_names:
            return cluster_names[label]  # Removendo o "Cluster X -" do início
        else:
            return f'Cluster {label}'
    
    # Prepare data with cluster information
    plot_df = pd.DataFrame({
        'x': umap_embeddings[:, 0],
        'y': umap_embeddings[:, 1],
        'text': df[text_col],
        'description': df[desc_col] if desc_col else [''] * len(df),
        'cluster': cluster_labels,
        'cluster_name': [get_cluster_name(label) for label in cluster_labels],
        'size': np.ones(len(cluster_labels))  # Tamanho uniforme por enquanto
    })
    
    # Calculando padding para centralização
    x_min, x_max = plot_df['x'].min(), plot_df['x'].max()
    y_min, y_max = plot_df['y'].min(), plot_df['y'].max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    # Definindo cores para os clusters
    unique_clusters = sorted(list(set(cluster_labels)))
    cluster_options = [get_cluster_name(label) for label in unique_clusters]
    
    # Cores distintas para os clusters
    CLUSTER_COLORS = {
        name: color for name, color in zip(
            cluster_options,
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        )
    }
    
    # Criando a seleção interativa
    click = alt.selection_single(
        fields=['cluster_name'],
        on='click',
        empty='none'
    )
    
    # Configurando a opacidade baseada na seleção
    opacity_condition = alt.condition(
        click,
        alt.value(1),
        alt.value(0.3)
    )
    
    # Calculando centroides dos clusters para labels
    cluster_centroids = plot_df.groupby('cluster_name').agg({
        'x': 'mean',
        'y': 'mean'
    }).reset_index()
    
    # Criando a camada de labels dos clusters
    text_labels = alt.Chart(cluster_centroids).mark_text(
        align='center',
        baseline='middle',
        fontSize=15,  # Fonte menor
        font='Arial',
        fontWeight='normal',  # Não usar negrito
        dx=15,  # Deslocar horizontalmente
        dy=-15   # Deslocar verticalmente
    ).encode(
        x='x:Q',
        y='y:Q',
        text=alt.Text('cluster_name:N'),
        opacity=alt.value(0.7)  # Reduzir opacidade
    )
    
    # Criando o gráfico base
    scatter_base = alt.Chart(plot_df).mark_circle(
        stroke='lightgray',
        strokeWidth=1,
    ).encode(
        x=alt.X('x:Q',
            title=None,  
            scale=alt.Scale(domain=[x_min - x_padding, x_max + x_padding]),
            axis=alt.Axis(ticks=False, labels=False, grid=False)
        ),
        y=alt.Y('y:Q',
            title=None,  
            scale=alt.Scale(domain=[y_min - y_padding, y_max + y_padding]),
            axis=alt.Axis(ticks=False, labels=False, grid=False)
        ),
        size=alt.Size(
            'size:Q',
            scale=alt.Scale(range=[50, 400]),
            legend=None
        ),
        color=alt.Color(
            'cluster_name:N',
            scale=alt.Scale(domain=list(CLUSTER_COLORS.keys()),
                          range=list(CLUSTER_COLORS.values())),
            legend=None  # Removendo a legenda da direita
        ),
        opacity=opacity_condition,
        tooltip=[
            alt.Tooltip('text:N', title='Título'),
            alt.Tooltip('description:N', title='Descrição'),
            alt.Tooltip('cluster_name:N', title='Cluster')
        ]
    ).add_selection(click)
    
    # Combinando as camadas
    final_chart = (scatter_base + text_labels).properties(
        width=1200,  # Aumentando ainda mais a largura
        height=800   # Aumentando a altura também
    ).configure_view(
        strokeWidth=0
    ).interactive()
    
    return final_chart

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
