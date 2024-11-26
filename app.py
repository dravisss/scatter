import os
# Disable Numba threading
os.environ["NUMBA_THREADING_LAYER"] = "omp"
os.environ["NUMBA_NUM_THREADS"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
import glob
from utils.embeddings import get_embeddings, save_embeddings, load_embeddings, get_saved_embeddings
from utils.visualization import reduce_dimensionality, cluster_data, get_visualization_params, get_cluster_colors, create_bokeh_scatter, create_altair_scatter
from utils.cluster_naming import generate_cluster_names
import textwrap
from bokeh.models import BooleanFilter, CDSView
import altair as alt

# Constants
EMBEDDINGS_DIR = "embeddings"

# Create embeddings directory if it doesn't exist
if not os.path.exists(EMBEDDINGS_DIR):
    try:
        os.makedirs(EMBEDDINGS_DIR)
    except FileExistsError:
        pass

# Configuração da página deve ser a primeira chamada Streamlit
st.set_page_config(
    page_title="Narrative Clustering",
    layout="wide",  # Usar layout wide
    initial_sidebar_state="expanded"
)

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
        cluster_labels, cluster_probabilities = cluster_data(umap_embeddings, **params["hdbscan"])
    
    return umap_embeddings, cluster_labels, cluster_probabilities

def main():
    st.title("Visualizador de Narrativas")
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
            st.session_state.embeddings = embeddings
            st.session_state.df = df
    else:
        # Clear session state when switching to new file
        if hasattr(st.session_state, 'additional_columns'):
            del st.session_state.additional_columns
            
        st.subheader("Carregar Novo Arquivo")
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=['csv'])
        if uploaded_file:
            embeddings, texts, df = handle_new_file(uploaded_file)
            st.session_state.embeddings = embeddings
            st.session_state.df = df

    # Process and visualize data if we have valid embeddings
    if 'embeddings' in st.session_state and 'df' in st.session_state:
        # Get visualization parameters
        params = get_visualization_params()
        
        # Process embeddings (with caching)
        umap_embeddings, cluster_labels, cluster_probabilities = process_embeddings(st.session_state.embeddings, params)
        
        # Generate cluster names using Claude
        cluster_info = {}
        with st.spinner("Gerando nomes e resumos para os clusters..."):
            try:
                cluster_data = generate_cluster_names(st.session_state.df, cluster_labels, cluster_probabilities)
                if cluster_data is not None:
                    cluster_names = cluster_data["names"]
                    cluster_summaries = cluster_data["summaries"]
                    cluster_stats = cluster_data["stats"]
                    st.success("Informações dos clusters geradas com sucesso!")
                else:
                    st.warning("Não foi possível gerar informações para os clusters. Usando números como identificadores.")
                    cluster_names = {}
                    cluster_summaries = {}
                    cluster_stats = {}
            except Exception as e:
                st.error(f"Erro ao gerar informações dos clusters: {str(e)}")
                cluster_names = {}
                cluster_summaries = {}
                cluster_stats = {}

        # Toggle between visualizations
        viz_type = st.radio("Escolha o tipo de visualização:", ["Interativa (Altair)", "Bokeh"])
        
        if viz_type == "Interativa (Altair)":
            # Create and display Altair plot
            chart = create_altair_scatter(
                umap_embeddings,
                st.session_state.df,
                st.session_state.df.columns[0],
                st.session_state.df.columns[1] if len(st.session_state.df.columns) > 1 else None,
                cluster_labels,
                cluster_names
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            # Bokeh visualization code...
            pass

        # Display cluster information after the visualization
        st.markdown("""---""")  # Divisor
        st.subheader("Resumo dos Clusters")
        st.markdown("""
        <style>
        .cluster-table {
            font-size: 14px;
            margin: 10px 0;
            width: 100%;
        }
        .cluster-table th {
            background-color: #f0f2f6;
            padding: 8px;
            text-align: left;
        }
        .cluster-table td {
            padding: 8px;
            border-bottom: 1px solid #e6e9ef;
        }
        .cluster-table tr:hover {
            background-color: #f5f7fa;
        }
        </style>
        """, unsafe_allow_html=True)

        for cluster_id in sorted(cluster_names.keys()):
            with st.expander(f"{cluster_names[cluster_id]} ({cluster_stats[cluster_id]['count']} narrativas - {cluster_stats[cluster_id]['percentage']:.1f}%)"):
                # Resumo do cluster
                st.markdown("**Resumo Temático:**")
                st.write(cluster_summaries[cluster_id])
                
                # Tabela de narrativas
                st.markdown("**Narrativas do Cluster:**")
                
                # Filtrar narrativas deste cluster
                cluster_mask = cluster_labels == cluster_id
                cluster_df = st.session_state.df[cluster_mask].copy()
                
                # Se temos scores de pertencimento, ordenar por eles
                if cluster_probabilities is not None:
                    cluster_df['score'] = cluster_probabilities[cluster_mask]
                    cluster_df = cluster_df.sort_values('score', ascending=False)
                
                # Criar HTML da tabela
                table_html = "<table class='cluster-table'>"
                table_html += "<tr><th>Título</th><th>Narrativa</th></tr>"
                
                for _, row in cluster_df.iterrows():
                    title = row[st.session_state.df.columns[0]]
                    narrative = row[st.session_state.df.columns[1]]
                    table_html += f"<tr><td>{title}</td><td>{narrative}</td></tr>"
                
                table_html += "</table>"
                st.markdown(table_html, unsafe_allow_html=True)

        # Add export functionality
        st.subheader("Exportar Resultados")
        if st.session_state.df is not None:
            export_results(st.session_state.df, umap_embeddings, cluster_labels)
        else:
            st.warning("Dados originais não disponíveis para exportação.")

if __name__ == "__main__":
    main()
