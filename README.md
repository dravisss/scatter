# Interactive Data Visualization App

This Streamlit application processes CSV files to create interactive visualizations using OpenAI embeddings, UMAP dimensionality reduction, and HDBSCAN clustering.

## Features
- Upload CSV files
- Generate embeddings using OpenAI's text-embedding-ada-002 model
- Reduce dimensionality with UMAP
- Cluster data points using HDBSCAN
- Interactive scatter plot visualization with Plotly

## Setup
1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```
4. Run the app:
```bash
streamlit run app.py
```

## Usage
1. Upload your CSV file (ensure the first two columns contain the text data you want to analyze)
2. Wait for the processing to complete
3. Interact with the generated visualization

## Requirements
- Python 3.8+
- OpenAI API key
- See requirements.txt for full list of dependencies
