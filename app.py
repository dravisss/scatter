import os
# Disable Numba threading
os.environ["NUMBA_THREADING_LAYER"] = "omp"
os.environ["NUMBA_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
from datetime import datetime
import glob
from utils.embeddings import get_embeddings, save_embeddings, load_embeddings, get_saved_embeddings
from utils.visualization import reduce_dimensionality, cluster_data, get_visualization_params, get_cluster_colors
from utils.cluster_naming import generate_cluster_names
import textwrap

# Constants
EMBEDDINGS_DIR = "embeddings"

# Create embeddings directory if it doesn't exist
if not os.path.exists(EMBEDDINGS_DIR):
    try:
        os.makedirs(EMBEDDINGS_DIR)
    except FileExistsError:
        pass

def handle_new_file(uploaded_file):
    """Process a new uploaded file."""
    try:
        # Clear session state
        if hasattr(st.session_state, 'additional_columns'):
            del st.session_state.additional_columns
        
        # Read CSV with the correct settings for your exported file
        try:
            df = pd.read_csv(
                uploaded_file,
                sep=';',  # Use semicolon as separator
                encoding='utf-8',  # UTF-8 encoding
                index_col=False,  # Don't use any column as index
                dtype=str  # Read all columns as strings
            )
            
            # Ensure we have the right columns
            if len(df.columns) >= 2:
                # Keep only the first two columns if there are more
                df = df.iloc[:, :2]
                
                # Rename columns to ensure they match what we expect
                df.columns = ['Label', 'Description']
                
                st.write("Dados carregados com sucesso!")
                st.write("Número de linhas:", len(df))
                st.write("Colunas:", list(df.columns))
            else:
                st.error("O arquivo precisa ter pelo menos 2 colunas")
                return None, None, None
            
            # Store the raw text columns
            text_col1 = df.iloc[:, 0].astype(str)
            text_col2 = df.iloc[:, 1].astype(str)
            
            # Generate embeddings using concatenated text
            texts = text_col1 + " | " + text_col2
            with st.spinner("Generating embeddings..."):
                embeddings = get_embeddings(texts.tolist())
                if embeddings is None:
                    return None, None, None
                
                # Save embeddings
                save_path = save_embeddings(embeddings, df, uploaded_file.name, EMBEDDINGS_DIR)
                if save_path:
                    st.success(f"Embeddings saved successfully!")
            
            # Store original columns in session state
            st.session_state.text_columns = {
                'col1': text_col1.tolist(),
                'col2': text_col2.tolist()
            }
            
            # Store additional columns
            if len(df.columns) > 2:
                st.session_state.additional_columns = {
                    col: df[col].fillna('').astype(str).tolist() 
                    for col in df.columns[2:]
                }
            
            return embeddings, texts, df
        except Exception as e:
            st.error(f"Error reading CSV file. Please ensure: 1. All rows have the same number of columns 2. Remove the trailing semicolon at the end of each line 3. Save the file in UTF-8 encoding Original error: {str(e)}")
            return None, None, None
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None, None, None

def handle_existing_file(filepath):
    """Load and process existing embeddings."""
    try:
        # Clear session state
        if hasattr(st.session_state, 'additional_columns'):
            del st.session_state.additional_columns
        if hasattr(st.session_state, 'text_columns'):
            del st.session_state.text_columns
            
        # Load embeddings
        result = load_embeddings(filepath)
        if result is None:
            return None, None, None
            
        embeddings, columns_data, column_names = result
        
        # Create DataFrame for export
        df = pd.DataFrame({
            'Label': columns_data[0],
            'Description': columns_data[1]
        })
        
        # Store original text columns
        st.session_state.text_columns = {
            'col1': columns_data[0],
            'col2': columns_data[1],
            'col1_name': column_names[0],
            'col2_name': column_names[1]
        }
        
        # Create display text
        texts = [f"{str(col1)} | {str(col2)}" for col1, col2 in zip(columns_data[0], columns_data[1])]
        
        # Store additional columns
        if len(columns_data) > 2:
            st.session_state.additional_columns = {
                name: data for name, data in zip(column_names[2:], columns_data[2:])
            }
        
        return embeddings, texts, df
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None, None, None

def export_results(df, reduced_data, clusters):
    """Export results as CSV."""
    # Create a new dataframe with original data and clusters
    result_df = pd.DataFrame({
        'Label': df['Label'],
        'Description': df['Description'],
        'Cluster': clusters,
        'X': reduced_data[:, 0],
        'Y': reduced_data[:, 1]
    })
    
    # Convert to CSV
    csv = result_df.to_csv(index=False)
    
    # Create download button
    st.download_button(
        label="Baixar resultados como CSV",
        data=csv,
        file_name="resultados_cluster.csv",
        mime="text/csv"
    )

# Cache para UMAP e clustering
@st.cache_data(show_spinner=False)
def process_embeddings(embeddings, params):
    """Process embeddings with UMAP and HDBSCAN with caching."""
    with st.spinner("Reduzindo dimensionalidade..."):
        umap_embeddings = reduce_dimensionality(embeddings, **params["umap"])
    
    with st.spinner("Agrupando dados..."):
        cluster_labels = cluster_data(umap_embeddings, **params["hdbscan"])
    
    return umap_embeddings, cluster_labels

def main():
    # Page config
    st.set_page_config(page_title="Visualização de Dados Interativa", layout="wide")
    
    # Title and description
    st.title("Visualização de Dados Interativa")
    st.write("Gere embeddings, agrupe dados e crie visualizações interativas a partir dos seus arquivos CSV.")
    
    # First, let user choose between new file or existing embeddings
    saved_embeddings = get_saved_embeddings(EMBEDDINGS_DIR)
    if saved_embeddings:
        option = st.radio(
            "Escolha uma opção:",
            ["Usar embeddings existentes", "Carregar novo arquivo"],
            help="Selecione se deseja usar embeddings salvos anteriormente ou carregar um novo arquivo"
        )
    else:
        option = "Carregar novo arquivo"
        st.info("Nenhum embedding encontrado. Por favor, carregue um novo arquivo.")
    
    embeddings = None
    texts = None
    df = None
    
    # Handle file selection based on user choice
    if option == "Usar embeddings existentes":
        # Clear session state when switching to existing embeddings
        if hasattr(st.session_state, 'additional_columns'):
            del st.session_state.additional_columns
            
        st.subheader("Selecionar Embeddings Salvos")
        selected_embedding = st.selectbox(
            "Escolha os embeddings salvos:",
            list(saved_embeddings.keys()),
            help="Selecione embeddings salvos anteriormente para visualizar"
        )
        if selected_embedding:
            embeddings, texts, df = handle_existing_file(saved_embeddings[selected_embedding])
    else:
        # Clear session state when switching to new file
        if hasattr(st.session_state, 'additional_columns'):
            del st.session_state.additional_columns
            
        st.subheader("Carregar Novo Arquivo")
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
        if uploaded_file:
            embeddings, texts, df = handle_new_file(uploaded_file)
    
    # Process and visualize data if we have valid embeddings
    if embeddings is not None and texts is not None:
        # Get visualization parameters
        params = get_visualization_params()
        
        # Create index mapping to track original order
        original_indices = np.arange(len(embeddings))
        
        # Process embeddings with UMAP and HDBSCAN
        umap_embeddings, cluster_labels = process_embeddings(embeddings, params)
        
        # Generate cluster names using Claude
        with st.spinner("Gerando nomes para os clusters..."):
            try:
                cluster_names = generate_cluster_names(df, cluster_labels)
                st.success("Nomes dos clusters gerados com sucesso!")
            except Exception as e:
                st.error(f"Erro ao gerar nomes dos clusters: {str(e)}")
                cluster_names = None
        
        # Calculate outlier percentage
        n_outliers = sum(1 for label in cluster_labels if label == -1)
        outlier_percentage = (n_outliers / len(cluster_labels)) * 100
        
        # Generate colors and hover texts
        colors, hover_texts = get_cluster_colors(cluster_labels, cluster_names)
        
        # Get the text column name dynamically
        text_columns = df.select_dtypes(include=['object']).columns
        text_col = text_columns[0] if len(text_columns) > 0 else None
        desc_col = text_columns[1] if len(text_columns) > 1 else None
        
        if text_col is None:
            raise ValueError("No text column found in DataFrame")
        
        # Create hover texts with all available information
        def create_hover_text(hover, text, desc=None, cluster_label=None):
            hover_parts = [hover]  # Start with cluster name
            
            # Add title
            hover_parts.append(f"<b>Título:</b><br>{textwrap.fill(text, width=50)}")
            
            # Add description if available
            if desc is not None:
                hover_parts.append(f"<b>Descrição:</b><br>{textwrap.fill(desc, width=50)}")
            
            # Add cluster information
            if cluster_label is not None:
                if cluster_label == -1:
                    hover_parts.append("<b>Status:</b> Outlier")
                else:
                    hover_parts.append(f"<b>Cluster:</b> {cluster_label}")
            
            return "<br><br>".join(hover_parts)
        
        # Create scatter plot
        if params["umap"]["n_components"] == 2:
            fig = go.Figure(data=go.Scatter(
                x=umap_embeddings[:, 0],
                y=umap_embeddings[:, 1],
                mode='markers',
                marker=dict(
                    color=colors,
                    size=8,
                    opacity=0.6
                ),
                text=[create_hover_text(hover, text, 
                                      df[desc_col].iloc[i] if desc_col else None,
                                      cluster_labels[i])
                      for i, (hover, text) in enumerate(zip(hover_texts, df[text_col]))],
                hoverinfo='text'
            ))
        else:
            fig = go.Figure(data=go.Scatter3d(
                x=umap_embeddings[:, 0],
                y=umap_embeddings[:, 1],
                z=umap_embeddings[:, 2],
                mode='markers',
                marker=dict(
                    color=colors,
                    size=8,
                    opacity=0.6
                ),
                text=[create_hover_text(hover, text, 
                                      df[desc_col].iloc[i] if desc_col else None,
                                      cluster_labels[i])
                      for i, (hover, text) in enumerate(zip(hover_texts, df[text_col]))],
                hoverinfo='text'
            ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics
        st.write(f"Porcentagem de outliers: {outlier_percentage:.2f}%")
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        st.write(f"Número de clusters: {n_clusters}")
        
        # Add export functionality
        st.subheader("Exportar Resultados")
        if df is not None:
            export_results(df, umap_embeddings, cluster_labels)
        else:
            st.warning("Dados originais não disponíveis para exportação.")

if __name__ == "__main__":
    main()
