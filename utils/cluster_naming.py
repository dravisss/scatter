import os
import anthropic
import pandas as pd
import numpy as np
from collections import defaultdict

def get_cluster_samples(df, cluster_labels, cluster_scores=None, max_samples=8):
    """Get representative samples from each cluster based on membership scores if available."""
    cluster_samples = defaultdict(list)
    
    # Get the text column name dynamically
    text_columns = df.select_dtypes(include=['object']).columns
    text_col = text_columns[0] if len(text_columns) > 0 else None
    
    if text_col is None:
        raise ValueError("No text column found in DataFrame")
    
    if cluster_scores is not None:
        # If we have scores, use them to get top samples
        for text, label, score in zip(df[text_col], cluster_labels, cluster_scores):
            if label != -1:  # Ignore noise points
                cluster_samples[label].append((text, score))
        
        # For each cluster, get top max_samples based on scores
        return {
            cluster: [text for text, _ in sorted(texts, key=lambda x: x[1], reverse=True)[:max_samples]]
            for cluster, texts in cluster_samples.items()
        }
    else:
        # If no scores, use random sampling
        for text, label in zip(df[text_col], cluster_labels):
            if label != -1:  # Ignore noise points
                cluster_samples[label].append(text)
        
        return {
            cluster: np.random.choice(texts, min(len(texts), max_samples), replace=False).tolist()
            for cluster, texts in cluster_samples.items()
        }

def generate_cluster_summary(client, samples):
    """Generate a thematic summary for a cluster based on its samples."""
    prompt = f"""Analise as seguintes narrativas que pertencem ao mesmo grupo temático:

{chr(10).join([f'- {text}' for text in samples])}

Gere um resumo temático em 200 palavras que sintetize o assunto principal dessas narrativas.
O resumo deve ser escrito em português, em um parágrafo único, e deve capturar os elementos mais importantes e recorrentes nas narrativas.
Responda APENAS com o resumo, sem introduções ou explicações adicionais."""

    # Get response from Claude
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        temperature=0,
        system="Você é um especialista em análise temática e síntese de narrativas.",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Get the response content
    response_content = message.content
    if isinstance(response_content, list):
        response_content = response_content[0].text if response_content else "Não foi possível gerar resumo"
    
    return response_content.strip()

def generate_cluster_names(df, cluster_labels, cluster_scores=None):
    """Generate names and summaries for each cluster using Claude API."""
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    
    try:
        # Get samples for each cluster
        cluster_samples = get_cluster_samples(df, cluster_labels, cluster_scores)
        cluster_names = {}
        cluster_summaries = {}
        cluster_stats = {}
        
        # Calculate cluster statistics
        total_points = len(cluster_labels)
        cluster_counts = {}
        for label in cluster_labels:
            if label != -1:  # Exclude noise points
                cluster_counts[label] = cluster_counts.get(label, 0) + 1
        
        for cluster_id, samples in cluster_samples.items():
            # Generate cluster name
            name_prompt = f"""Analise os seguintes textos que pertencem ao mesmo cluster:

{chr(10).join([f'- {text}' for text in samples])}
Não use as palavras: Cultura, Clima, Estratégia, gestão de tempo. 
Com base nesses textos, gere um nome conciso (máximo 3-4 palavras) que capture o tema ou assunto principal comum entre eles.
Responda APENAS com o nome, sem explicações adicionais."""

            # Get name from Claude
            name_message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                temperature=0,
                system="Você é um assistente especializado em análise de texto e categorização. Gere nomes concisos e precisos para grupos de texto relacionados.",
                messages=[{"role": "user", "content": name_prompt}]
            )
            
            # Store the generated name
            response_content = name_message.content
            if isinstance(response_content, list):
                response_content = response_content[0].text if response_content else "Cluster sem nome"
            cluster_names[cluster_id] = response_content.strip()
            
            # Generate cluster summary
            cluster_summaries[cluster_id] = generate_cluster_summary(client, samples)
            
            # Store cluster statistics
            count = cluster_counts.get(cluster_id, 0)
            percentage = (count / total_points) * 100
            cluster_stats[cluster_id] = {
                "count": count,
                "percentage": percentage
            }
        
        return {
            "names": cluster_names,
            "summaries": cluster_summaries,
            "stats": cluster_stats
        }
    
    except Exception as e:
        print(f"Error generating cluster names and summaries: {str(e)}")
        return None
