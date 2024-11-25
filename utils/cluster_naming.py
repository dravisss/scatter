import os
import anthropic
import pandas as pd
import numpy as np
from collections import defaultdict

def get_cluster_samples(df, cluster_labels, max_samples=5):
    """Get representative samples from each cluster."""
    cluster_samples = defaultdict(list)
    
    # Get the text column name dynamically
    text_columns = df.select_dtypes(include=['object']).columns
    text_col = text_columns[0] if len(text_columns) > 0 else None
    
    if text_col is None:
        raise ValueError("No text column found in DataFrame")
    
    for text, label in zip(df[text_col], cluster_labels):
        if label != -1:  # Ignore noise points
            cluster_samples[label].append(text)
    
    # For each cluster, get up to max_samples random samples
    return {
        cluster: np.random.choice(texts, min(len(texts), max_samples), replace=False).tolist()
        for cluster, texts in cluster_samples.items()
    }

def generate_cluster_names(df, cluster_labels):
    """Generate names for each cluster using Claude API."""
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    try:
        # Get samples for each cluster
        cluster_samples = get_cluster_samples(df, cluster_labels)
        cluster_names = {}
        
        for cluster_id, samples in cluster_samples.items():
            # Create prompt
            prompt = f"""Analise os seguintes textos que pertencem ao mesmo cluster:

{chr(10).join([f'- {text}' for text in samples])}

Com base nesses textos, gere um nome conciso (máximo 3-4 palavras) que capture o tema ou assunto principal comum entre eles.
Responda APENAS com o nome, sem explicações adicionais."""

            # Get response from Claude
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=8000,
                temperature=0,
                system="Você é um assistente especializado em análise de texto e categorização. Gere nomes concisos e precisos para grupos de texto relacionados.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Store the generated name, ensuring we get a string
            response_content = message.content
            if isinstance(response_content, list):
                response_content = response_content[0].text if response_content else "Cluster sem nome"
            
            cluster_names[cluster_id] = response_content.strip()
        
        return cluster_names
    except Exception as e:
        print(f"Error generating cluster names: {str(e)}")
        return None
