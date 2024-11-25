# Interactive Data Visualization App

This Streamlit application processes CSV files to create interactive visualizations using OpenAI embeddings, UMAP dimensionality reduction, and HDBSCAN clustering. The app is designed for exploratory data analysis, helping users discover patterns and relationships in textual data through advanced visualization techniques.

## Overview
The application takes CSV files as input and processes them through several sophisticated steps:
1. Text Processing: Uses the first two columns of your CSV as primary text data
2. Embedding Generation: Creates numerical representations of text using OpenAI's text-embedding-3-small model
3. Dimensionality Reduction: Applies UMAP to make the data visualizable
4. Clustering: Uses HDBSCAN to identify natural groupings in your data
5. Interactive Visualization: Creates an interactive scatter plot using Plotly

## Features
- Upload and process CSV files
- Generate embeddings using OpenAI's text-embedding-3-small model (with batch processing)
- Reduce dimensionality with UMAP for better visualization
- Cluster data points using HDBSCAN
- Interactive scatter plot visualization with Plotly
- Save and reload previously generated embeddings
- Support for additional columns beyond the primary text data
- Batch processing for improved performance

## Use Cases
- Text Data Analysis: Understand relationships between different text entries
- Document Clustering: Group similar documents or text entries
- Pattern Discovery: Identify trends and patterns in your textual data
- Data Exploration: Interactively explore your dataset through visual representation

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
2. Wait for the processing to complete (embedding generation and clustering)
3. Interact with the generated visualization:
   - Hover over points to see detailed information
   - Explore clusters and relationships
   - Use additional columns for enhanced analysis

## CSV File Format Requirements
Your CSV file should follow these requirements:

1. **Minimum Columns**: The file must have at least 2 columns
2. **First Two Columns**: 
   - The first two columns are used as the primary text data for analysis
   - These columns will be concatenated to generate embeddings
   - Any type of text data is accepted (will be converted to string)
3. **Additional Columns** (Optional):
   - You can include any number of additional columns after the first two
   - These will be available for additional analysis and visualization
   - All additional columns will be converted to strings
4. **File Format**:
   - Standard CSV format
   - UTF-8 encoding recommended
   - Headers should be included
   - Can use either comma (,) or semicolon (;) as delimiter
   - **Important**: If your text contains commas, you must enclose it in quotes
   - All rows must have the same number of columns

Example CSV with semicolon delimiter:
```csv
Label;Description
A Saga da Nota Fiscal;Eu tenho que cancelar uma nota. Eu sei que eu tenho que cancelar essa nota, não tem outra solução.
Quem Reporta para Quem?;Às vezes me pergunto: 'Com quem eu tenho que reportar isso?'
```

### Common CSV Issues and Solutions
1. **Text Contains Commas**: 
   - If using comma as delimiter, enclose text in quotes: `"Text, with comma"`
   - Or use semicolon as delimiter instead
2. **Inconsistent Columns**: 
   - Ensure all rows have the same number of columns
   - Check for missing or extra delimiters
   - Remove trailing delimiters (like semicolon at the end of lines)
3. **Special Characters**:
   - Use UTF-8 encoding
   - Escape special characters if needed
4. **Trailing Delimiters**:
   - Some CSV editors might add an extra delimiter at the end of each line
   - You can safely remove these trailing delimiters
   - The app will automatically handle trailing delimiters

## Requirements
- Python 3.8+
- OpenAI API key
- See requirements.txt for full list of dependencies

## Technical Details
- Uses OpenAI's text-embedding-3-small model for text embedding
- Implements UMAP for dimensionality reduction
- Utilizes HDBSCAN for clustering
- Built with Streamlit for the web interface
- Plotly for interactive visualizations
- Pandas for data handling
