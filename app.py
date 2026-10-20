import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer, CrossEncoder

# ============================================================
# RESOURCE LOADING & INITIALIZATION
# ============================================================
# Define the directory where the pre-processed artifacts are stored
save_path = r"D:\Fall 2026 (4)\CS 1st Semester\CS492 Project\Chat Bot"

# Load the main dataset containing disease information and metadata
df = pd.read_pickle(os.path.join(save_path, "df.pkl"))

# Load the BM25 object for keyword-based (sparse) retrieval
with open(os.path.join(save_path, "bm25.pkl"), "rb") as f:
    bm25 = pickle.load(f)

# Load the FAISS index for high-performance vector (dense) similarity search
index = faiss.read_index(os.path.join(save_path, "faiss.index"))

# Load the pre-computed document embeddings for the FAISS index
embeddings = np.load(os.path.join(save_path, "embeddings.npy"))

# Load the raw text corpus used for indexing and retrieval
with open(os.path.join(save_path, "texts.pkl"), "rb") as f:
    texts = pickle.load(f)

# Load the configuration file (e.g., model names, hyperparameters)
with open(os.path.join(save_path, "config.pkl"), "rb") as f:
    config = pickle.load(f)

# ============================================================
# DEEP LEARNING MODEL LOADER
# ============================================================
@st.cache_resource
def load_models():
    """
    Loads transformer models once and caches them to save memory and startup time.
    - SentenceTransformer: Generates dense vector representations of text.
    - CrossEncoder: Performs high-accuracy re-ranking of retrieved candidates.
    """
    embedding_model = SentenceTransformer(config["embedding_model"])
    reranker = CrossEncoder(config["reranker_model"])
    return embedding_model, reranker

embedding_model, reranker = load_models()

# ============================================================
# CORE LOGIC FUNCTIONS
# ============================================================

def detect_plant(query):
    """
    Analyzes the query string to identify the specific plant mentioned.
    This is used to narrow down the search space and improve precision.
    """
    plants = ["apple","tomato","grape","bell pepper","potato","blueberry","strawberry","sugercane","watermelon",
              "corn","cassava","coffee","rice","cherry","rose","orange","raspberry","squash","soybean","peach"]
    q = query.lower()
    for p in plants:
        if p in q:
            return p
    return None

def reasoning_answer(query, best):
    """
    Expert System Layer: Intercepts potential user errors in chemical treatment.
    Prevents the model from recommending fungicides for viruses or antibiotics for fungi.
    """
    q = query.lower()
    disease_type = str(best.get("type", "")).lower()
    name = best.get("clean_name", "this disease")

    # Misuse Case 1: Checking if antibiotics are suggested for non-bacterial diseases
    if any(word in q for word in ["antibiotic", "antibacterial"]):
        if disease_type in ["fungal", "oomycete", "virus", "pest"]:
            return f"**Warning:** Antibiotics do NOT work on {disease_type} diseases like {name}. \n\n**Recommended:** Use appropriate fungicides or remove infected plants."

    # Misuse Case 2: Checking if fungicides/sprays are suggested for viruses or bacteria
    if any(word in q for word in ["fungicide", "spray", "chemical"]):
        if disease_type == "virus":
            return f"**Warning:** Fungicides do NOT work on viral diseases like {name}. \n\n**Recommended:** Remove and burn infected plants. Control insect vectors (e.g., whiteflies)."
        elif disease_type == "bacterial":
            return f"**Warning:** Standard fungicides do NOT work on bacterial diseases like {name}. \n\n**Recommended:** Use copper-based bactericides."

    return None

def hybrid_search(query, top_k=5):
    """
    Retrieval Layer: Combines BM25 (lexical) and FAISS (semantic) results.
    This ensures the system finds exact keywords and conceptual matches.
    """
    # 1. Sparse Retrieval (BM25)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # 2. Dense Retrieval (FAISS)
    q_emb = embedding_model.encode([query], normalize_embeddings=True)
    _, faiss_indices = index.search(q_emb, top_k)
    faiss_indices = faiss_indices[0]

    # 3. Combine indices and remove duplicates
    candidate_indices = list(set(
        faiss_indices.tolist() +
        np.argsort(bm25_scores)[-top_k:].tolist()
    ))

    # 4. Filter results based on the detected plant name for higher accuracy
    results = df.iloc[candidate_indices].copy()
    plant = detect_plant(query)
    if plant:
        filtered = results[results["clean_name"].str.lower().str.contains(plant)]
        if len(filtered) > 0:
            results = filtered
    return results

def rerank_full(query, results):
    """
    Refinement Layer: Uses a Cross-Encoder to rank candidates by actual relevance.
    Cross-Encoders are slower but significantly more accurate than vector similarity.
    """
    pairs = [[query, row["combined_text"]] for _, row in results.iterrows()]
    scores = reranker.predict(pairs)
    results["score"] = scores
    results = results.sort_values(by="score", ascending=False)
    return results

def answer_question(query):
    """
    Orchestration Layer: Coordinates search, ranking, and logic to generate the final response.
    """
    # Step 1: Retrieve and Rank candidates
    results = hybrid_search(query)
    results = rerank_full(query, results)
    
    if len(results) == 0:
        return "Sorry, I couldn't find information on that."

    # Step 2: Extract top candidate and validate relevance via score threshold
    best = results.iloc[0]
    if best['score'] < -4.0: 
        return "Sorry, I couldn't find information on that."

    # Step 3: Handle Comparison queries (e.g., 'Difference between X and Y')
    comparison_keywords = ["difference", "compare", "vs", "versus", "between"]
    is_comparison = any(word in query.lower() for word in comparison_keywords)

    if is_comparison and len(results) >= 2:
        d1 = results.iloc[0]
        d2 = results.iloc[1]
        
        # Sort diseases by their order of appearance in the user's query
        n1, n2 = d1['clean_name'].lower(), d2['clean_name'].lower()
        if query.lower().find(n2) < query.lower().find(n1) and query.lower().find(n2) != -1:
            d1, d2 = d2, d1

        return f"""
        Comparison: {d1['clean_name']} vs {d2['clean_name']}
        
        Type:
        - ({d1['clean_name']}) : {d1['type']}  
        - ({d2['clean_name']}) : {d2['type']}

        Symptoms:
        - ({d1['clean_name']}) : {d1['Disease Definition']}
        - ({d2['clean_name']}) : {d2['Disease Definition']}

        Active Ingredient:
        - ({d1['clean_name']}) : {d1['Active Ingredient']} 
        - ({d2['clean_name']}) : {d2['Active Ingredient']} 

        Dosage:
        - ({d1['clean_name']}) : {d1['Dosage']}  
        - ({d2['clean_name']}) : {d2['Dosage']} 

        Treatment:
        - ({d1['clean_name']}) : {d1['Treatment']}
        - ({d2['clean_name']}) : {d2['Treatment']}
        """

    # Step 4: Validate single-result logic through the Reasoning Layer
    reasoning = reasoning_answer(query, best)
    if reasoning:
        return reasoning

    # Step 5: Format and return the standard disease report
    return f"""
**Disease:** {best['clean_name']} ({best['type']})

**Definition:**
{best['Disease Definition']}

**Treatment:**
{best['Treatment']}

**Active Ingredient:**
{best['Active Ingredient']}

**Dosage:**
{best['Dosage']}

**Steps:**
{best['Treatment Steps']}
"""

# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================
st.set_page_config(page_title="🌱 Plant Disease Detection Chatbot", layout="centered")
st.title("🌱 Plant Disease Chatbot")

# User input text box
query = st.text_input("Ask about plant disease treatment (e.g., 'What is treatment for tomato early blight?')")

# Execution trigger
if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please enter a question")
    else:
        # Process query and display the formatted markdown response
        answer = answer_question(query)
        st.info(answer)