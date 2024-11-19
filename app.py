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
        plot_df = pd.DataFrame(umap_embeddings, columns=[f"UMAP{i+1}" for i in range(umap_embeddings.shape[1])])
        plot_df["Cluster"] = [f"Cluster {l}" if l >= 0 else "Outliers" for l in cluster_labels]
        plot_df["Text"] = texts
        
        # Format text for tooltip with line breaks
        def clean_text_for_tooltip(idx):
            if hasattr(st.session_state, 'text_columns'):
                col1 = st.session_state.text_columns['col1'][idx]
                col2 = st.session_state.text_columns['col2'][idx]
                col1_name = st.session_state.text_columns.get('col1_name', 'Column 1')
                col2_name = st.session_state.text_columns.get('col2_name', 'Column 2')
                return f"<b>{col1_name}:</b> {col1}<br><b>{col2_name}:</b> {col2}"
            return plot_df["Text"].iloc[idx]

        plot_df["Tooltip_Text"] = [clean_text_for_tooltip(i) for i in range(len(plot_df))]
        
        # Add additional columns if available
        tooltip_columns = []
        if hasattr(st.session_state, 'additional_columns'):
            for col_name, col_data in st.session_state.additional_columns.items():
                # Ensure col_data has the same length as plot_df
                if len(col_data) == len(plot_df):
                    plot_df[col_name] = col_data
                    # Format additional columns for tooltip
                    if isinstance(col_data[0], (int, float)):
                        tooltip_col = f"Tooltip_{col_name}"
                        plot_df[tooltip_col] = [f"{col_name}: {val:.2f}" for val in col_data]
                        tooltip_columns.append(tooltip_col)
                    else:
                        tooltip_col = f"Tooltip_{col_name}"
                        plot_df[tooltip_col] = [f"{col_name}: {str(val)}" for val in col_data]
                        tooltip_columns.append(tooltip_col)
                else:
                    st.warning(f"Column {col_name} has different length than the main data and will be skipped.")
        
        # Create plot
        st.subheader("Visualização")
        point_size = st.slider("Tamanho dos pontos", 1, 20, 5)
        
        # Prepare hover data template
        hover_template = (
            "<b>Texto:</b><br>%{customdata[0]}<br><br>"
            "<b>Cluster:</b> %{customdata[1]}"
        )
        
        # Add additional metadata to hover template if available
        if tooltip_columns:
            for i, col_name in enumerate(tooltip_columns, start=2):
                hover_template += f"<br>%{{customdata[{i}]}}"
        
        hover_template += "<extra></extra>"
        
        # Calculate figure dimensions for 16:9 aspect ratio
        width = 1200
        height = int(width * 9/16)
        
        # Prepare custom data for hover
        customdata = [plot_df["Tooltip_Text"].tolist(), plot_df["Cluster"].tolist()]
        if tooltip_columns:
            for col_name in tooltip_columns:
                customdata.append(plot_df[col_name].tolist())
        customdata = np.array(customdata).T  # Transpose to match Plotly's expected format
        
        if umap_embeddings.shape[1] == 3:
            fig = px.scatter_3d(
                plot_df,
                x="UMAP1",
                y="UMAP2",
                z="UMAP3",
                color="Cluster",
                title="Visualização UMAP 3D",
                size=[point_size] * len(plot_df)
            )
        else:
            fig = px.scatter(
                plot_df,
                x="UMAP1",
                y="UMAP2",
                color="Cluster",
                title="Visualização UMAP 2D",
                size=[point_size] * len(plot_df)
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
            )
        )
        
        # Update hover template for all traces
        for trace in fig.data:
            trace.customdata = customdata
            trace.hovertemplate = hover_template
        
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Porcentagem de outliers: {outlier_percentage:.2f}%")

if __name__ == "__main__":
    main()
