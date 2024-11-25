import os
import json
import glob
import streamlit as st
import numpy as np
from datetime import datetime
from .config import client

def get_embeddings(texts):
    """Generate embeddings using OpenAI's API with batch processing."""
    try:
        # Initialize list to store all embeddings
        all_embeddings = []
        
        # Process in batches of 100 texts to stay well within the 8000 token limit
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = client.embeddings.create(
                input=batch_texts,
                model="text-embedding-3-small"
            )
            batch_embeddings = [embedding.embedding for embedding in response.data]
            all_embeddings.extend(batch_embeddings)
        
        # Convert to float64 numpy array
        return np.array(all_embeddings, dtype=np.float64)
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

def save_embeddings(embeddings, df, filename, embeddings_dir):
    """Save embeddings and complete metadata to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(embeddings_dir, f"{filename}_{timestamp}.json")
    
    # Store all columns data and names
    columns_data = {}
    for i, col_name in enumerate(df.columns):
        # Convert data to serializable format
        col_data = df.iloc[:, i]
        if col_data.dtype == 'datetime64[ns]':
            col_data = col_data.astype(str)
        elif col_data.dtype == 'object':
            col_data = col_data.fillna('').astype(str)
        else:
            col_data = col_data.fillna(0).tolist()
            
        columns_data[f"column_{i+1}_data"] = col_data.tolist() if hasattr(col_data, 'tolist') else list(col_data)
        columns_data[f"column_{i+1}_name"] = str(col_name)
    
    data = {
        "embeddings": embeddings.tolist(),
        "n_columns": len(df.columns),
        "filename": str(filename),
        "timestamp": timestamp,
        **columns_data
    }
    
    try:
        # Validate JSON before saving
        json_str = json.dumps(data)
        json.loads(json_str)  # Test if it can be loaded back
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
        
        return save_path
    except Exception as e:
        st.error(f"Error saving embeddings: {str(e)}")
        return None

def load_embeddings(filepath):
    """Load embeddings and complete metadata from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        n_columns = data["n_columns"]
        
        # Extract all columns data
        columns_data = []
        column_names = []
        for i in range(n_columns):
            columns_data.append(data[f"column_{i+1}_data"])
            column_names.append(data[f"column_{i+1}_name"])
        
        # Ensure embeddings are float64
        embeddings = np.array(data["embeddings"], dtype=np.float64)
        
        return (embeddings, 
                columns_data,
                column_names)
    except json.JSONDecodeError as e:
        st.error(f"Error reading embedding file {filepath}: Invalid JSON format")
        return None
    except Exception as e:
        st.error(f"Error reading embedding file {filepath}: {str(e)}")
        return None

def get_saved_embeddings(embeddings_dir):
    """Get list of saved embedding files."""
    if not os.path.exists(embeddings_dir):
        return {}
    
    files = glob.glob(os.path.join(embeddings_dir, "*.json"))
    embeddings_dict = {}
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                # Create a more informative display name
                display_name = f"{data['filename']} ({data['timestamp']})"
                embeddings_dict[display_name] = f
        except json.JSONDecodeError:
            # Skip invalid JSON files
            continue
        except Exception as e:
            # Skip files with other errors
            continue
    
    return embeddings_dict
