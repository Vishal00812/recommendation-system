import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import joblib
import os

# --- Config ---
CSV_PATH = 'Active Product List  Admin 02.04.2025.csv'
MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDINGS_PATH = 'product_embeddings.npy'
INDEX_PATH = 'faiss_index.index'
DF_PATH = 'product_df.pkl'

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_data():
    if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(INDEX_PATH) and os.path.exists(DF_PATH):
        df = joblib.load(DF_PATH)
        index = faiss.read_index(INDEX_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
    else:
        df = pd.read_csv(CSV_PATH)
        df.fillna('', inplace=True)
        df['search_text'] = df['Name'] + ' ' + df['Type'] + ' ' + df['Category'] + ' ' + df['SubCategory']
        model = load_model()
        embeddings = model.encode(df['search_text'].tolist(), show_progress_bar=True)
        np.save(EMBEDDINGS_PATH, embeddings)
        joblib.dump(df, DF_PATH)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        faiss.write_index(index, INDEX_PATH)
    return df, index

def recommend_products(query, top_n=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_n)
    return df.iloc[indices[0]][['Product Id', 'Name', 'Category', 'SubCategory', 'Sale Price']]

# --- Streamlit UI ---
st.set_page_config(page_title="🛒 Product Recommender", layout="centered")
st.title("🔍 Semantic Product Recommender")

# Load model and data
with st.spinner("Loading model and product data..."):
    model = load_model()
    df, index = load_data()

# Query input
query = st.text_input("Enter a product search query (e.g., 'dal', 'toothpaste')", "")
top_n = st.slider("Number of recommendations", 1, 10, 3)

# Show results
if query:
    with st.spinner("Searching..."):
        results = recommend_products(query, top_n)
    st.success(f"Top {top_n} product recommendations for: '{query}'")
    st.dataframe(results.reset_index(drop=True))
