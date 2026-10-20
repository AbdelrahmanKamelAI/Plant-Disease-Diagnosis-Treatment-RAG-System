# Plant-Disease-Diagnosis-Treatment-RAG-System
Built an end-to-end RAG system for plant disease diagnosis and treatment recommendations using Streamlit, FAISS, BM25, and Cross-Encoder re-ranking. Includes hybrid retrieval, disease comparison, and treatment validation to prevent inappropriate chemical recommendations.


# 🌱 Plant Disease Diagnosis & Treatment RAG Chatbot

An intelligent, safety-conscious Retrieval-Augmented Generation (RAG) system and interactive Streamlit dashboard designed to assist farmers, agronomists, and researchers in identifying plant diseases, retrieving accurate treatment protocols, comparing pathologies, and preventing chemical misuse.

---

## 📁 Repository Structure & Components

| File | Type | Description |
| :--- | :--- | :--- |
| **`Plant Recommendations.xlsx`** | Knowledge Base | Comprehensive dataset containing 71 plant disease records with structured fields for symptoms, treatments, active ingredients, dosages, and step-by-step guidelines. |
| **`Chat Bot plant disease detection.ipynb`** | Training & Indexing Notebook | Development notebook for data preprocessing, dense/sparse index generation (FAISS + BM25), Cross-Encoder setup, expert safety rules design, and artifact serialization. |
| **`app.py`** | Web Application | Production-ready Streamlit interface providing real-time disease Q&A, side-by-side pathology comparisons, and treatment verification warnings. |

---

## 📊 Dataset Overview (`Plant Recommendations.xlsx`)

The dataset serves as the grounded knowledge base for the RAG retriever. It contains **71 agricultural disease entries** across 20 host plants (e.g., Tomato, Potato, Apple, Grape, Corn, Cassava, Coffee, Sugarcane) structured into 10 key attributes:

- **`Disease`**: Standardized disease identifier (e.g., `Apple___alternaria_leaf_sp`).
- **`Disease Definition`**: Detailed description of causal organisms, visual symptoms, and environmental triggers.
- **`Treatment`**: Recommended agricultural and chemical management strategies.
- **`Active Ingredient`**: Specific active chemicals or biological control agents (e.g., *Mancozeb*, *Chlorothalonil*, *Copper hydroxide*).
- **`Brand Name` & `Alternative`**: Commercial product names and alternative organic control methods.
- **`Dosage`**: Precise chemical application rates and dilution ratios.
- **`Duration of Treatment`**: Application frequency and seasonal timing rules.
- **`Treatment Steps`**: Sequential, step-by-step instructions for field execution.
- **`Medical Consultation`**: Follow-up inspection schedules and disease reduction benchmarks (e.g., *"Recheck after 7–10 days"*).

---

## 🧠 Model Architecture & Pipeline (`notebook` & `app.py`)
