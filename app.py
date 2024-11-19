import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import openai
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import glob
from utils.embeddings import get_embeddings, save_embeddings, load_embeddings, get_saved_embeddings
from utils.visualization import reduce_dimensionality, cluster_data, get_visualization_params
import textwrap

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        
        # Read CSV
        df = pd.read_csv(uploaded_file)
        if len(df.columns) < 2:
            st.error("File must have at least 2 columns")
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
        
        return embeddings, texts, None
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None, None, None

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
            embeddings, texts, _ = handle_existing_file(saved_embeddings[selected_embedding])
    else:
        # Clear session state when switching to new file
        if hasattr(st.session_state, 'additional_columns'):
            del st.session_state.additional_columns
            
        st.subheader("Carregar Novo Arquivo")
        uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")
        if uploaded_file:
            embeddings, texts, _ = handle_new_file(uploaded_file)
    
    # Process and visualize data if we have valid embeddings
    if embeddings is not None and texts is not None:
        # Get visualization parameters
        params = get_visualization_params()
        
        # Create index mapping to track original order
        original_indices = np.arange(len(embeddings))
        
        # Reduce dimensionality
        with st.spinner("Reduzindo dimensionalidade..."):
            umap_embeddings = reduce_dimensionality(
                embeddings,
                **params["umap"]
            )
        
        # Cluster the reduced embeddings
        with st.spinner("Agrupando dados..."):
            cluster_labels = cluster_data(
                umap_embeddings,
                **params["hdbscan"]
            )
        
        # Calculate outlier percentage
        n_outliers = sum(cluster_labels == -1)
        outlier_percentage = (n_outliers / len(cluster_labels)) * 100
        
        # Prepare data for plotting
        plot_df = pd.DataFrame({
            **{f"UMAP{i+1}": umap_embeddings[:, i] for i in range(umap_embeddings.shape[1])},
            "Cluster": [f"Cluster {l}" if l >= 0 else "Outliers" for l in cluster_labels],
            "Text": texts,
            "original_index": original_indices
        })
        
        # Format text for tooltip
        hover_template = (
            "<b>%{customdata[2]}</b><br>"  # Column 1 name
            "%{customdata[0]}<br><br>"     # Column 2 name
            "<b>%{customdata[3]}</b><br>"  # Column 1 value
            "%{customdata[1]}"             # Column 2 value
        )
        
        # Add additional columns if available
        tooltip_columns = []
        if hasattr(st.session_state, 'additional_columns'):
            for col_name, col_data in st.session_state.additional_columns.items():
                if len(col_data) == len(plot_df):
                    plot_df[col_name] = col_data
                    tooltip_columns.append(col_name)
        
        # Create plot
        st.subheader("Visualização")
        point_size = st.slider("Tamanho dos pontos", 1, 20, 5)
        
        # Prepare custom data for hover using original indices
        if hasattr(st.session_state, 'text_columns'):
            col1_data = np.array(st.session_state.text_columns['col1'])[plot_df['original_index']]
            col2_data = np.array(st.session_state.text_columns['col2'])[plot_df['original_index']]
            col1_name = st.session_state.text_columns.get('col1_name', 'Column 1')
            col2_name = st.session_state.text_columns.get('col2_name', 'Column 2')
            
            customdata = np.array([
                col1_data,  # Column 1 values
                col2_data,  # Column 2 values
                [col1_name] * len(plot_df),  # Column 1 name
                [col2_name] * len(plot_df),  # Column 2 name
            ]).T
        else:
            # Fallback to using the concatenated text
            texts_array = np.array(texts)[plot_df['original_index']]
            parts = [text.split(" | ", 1) for text in texts_array]
            customdata = np.array([
                [p[0] for p in parts],  # First part
                [p[1] if len(p) > 1 else "" for p in parts],  # Second part
                ["Column 1"] * len(plot_df),
                ["Column 2"] * len(plot_df)
            ]).T
        
        # Add additional metadata to hover template if available
        if tooltip_columns:
            for i, col_name in enumerate(tooltip_columns):
                hover_template += f"<br><b>{col_name}:</b> %{{customdata[{i+4}]}}"
                col_data = np.array(plot_df[col_name])[plot_df['original_index']].tolist()
                customdata = np.c_[customdata, col_data]
        
        hover_template += "<extra></extra>"
        
        # Calculate figure dimensions for 16:9 aspect ratio
        width = 1200
        height = int(width * 9/16)
        
        if umap_embeddings.shape[1] == 3:
            fig = px.scatter_3d(
                plot_df,
                x="UMAP1",
                y="UMAP2",
                z="UMAP3",
                color="Cluster",
                title="Visualização UMAP 3D",
                size=[point_size] * len(plot_df),
                custom_data=customdata
            )
        else:
            fig = px.scatter(
                plot_df,
                x="UMAP1",
                y="UMAP2",
                color="Cluster",
                title="Visualização UMAP 2D",
                size=[point_size] * len(plot_df),
                custom_data=customdata
            )
        
        # Update layout and traces
        fig.update_layout(
            width=width,
            height=height,
            showlegend=True,
            legend=dict(
                title="Clusters",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            hovermode='closest'
        )
        
        # Update hover template for all traces
        fig.update_traces(hovertemplate=hover_template)
        
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Porcentagem de outliers: {outlier_percentage:.2f}%")

if __name__ == "__main__":
    main()
