# 🤖 Restaurant Intelligence System (RIS)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/Powered%20by-LangChain-orange)](https://python.langchain.com/)
[![Ollama](https://img.shields.io/badge/Local%20LLM-Ollama-black)](https://ollama.ai/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)

The **Restaurant Intelligence System (RIS)** is a professional-grade, local-first Retrieval-Augmented Generation (RAG) platform. It transforms unstructured restaurant reviews into structured, actionable business insights—ensuring 100% data privacy by running entirely on your local hardware.

---

## 🌟 Key Features

- **Hybrid Retrieval**: Combines semantic meaning (Vector Search) with exact keyword matching (BM25).
- **Intelligent Filtering**: Automatically extracts restaurant names, ratings, and filters from natural language.
- **Precision Reranking**: Uses `FlashRank` (Cross-Encoder) to ensure the most relevant reviews are analyzed by the LLM.
- **Quantitative Evaluation**: Scientific performance tracking via `RAGAS` (Faithfulness, Answer Relevance).
- **Executive Dashboard**: Modern Streamlit UI with real-time "Behind the Scenes" inspection.

---

## 🏗 System Architecture

The project is designed with modularity and scalability in mind:

```text
Restaurant-Intelligence-System
├── data
│   ├── raw                # Source CSV data
│   └── chroma_db          # Persistent Vector DB storage
├── src
│   ├── core               # RAG Orchestration (Chains & Prompts)
│   ├── data_eng           # Ingestion (Loading, Validation, Chunking)
│   ├── retrieval          # Query Translation & Hybrid Search Logic
│   └── utils              # System Infrastructure (Ollama Helpers)
├── eval                   # Evaluation Framework (Test Sets & Reports)
├── app.py                 # Streamlit Web Dashboard
├── cli_prototype.py       # Command-Line Utility
└── pull_models.py         # Automated Environment Setup
```

---

## 🚀 Quick Start

### 1. Prerequisities
- **Python 3.9+**
- **Ollama**: Ensure Ollama is installed and running on your system.

### 2. Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd Restaurant-Intelligence-System

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Models & Data
```bash
# Pull required local models (Llama 3.2 & mxbai-embed-large)
python pull_models.py

# Ingest the restaurant reviews into the vector database
python cli_prototype.py --ingest
```

### 4. Run the Platform
Choose your preferred interface:

**A. Executive Dashboard (Recommended)**
```bash
streamlit run app.py
```

**B. Developer CLI**
```bash
python cli_prototype.py
```

---

## 🛠 Advanced Configuration

### LLM Inference
*Settings in `src/utils/ollama_helpers.py`*
- **Temperature**: Controls creativity vs. factual grounding (Default: 0 for RAG).
- **Model Selection**: Switch between `llama3.2`, `mistral`, or `phi3`.

### Retrieval Tuning
*Settings in `src/retrieval/hybrid_retriever.py`*
- **K-Value**: Number of document chunks retrieved for context.
- **Hybrid Weighting**: Ratio between Keyword (BM25) and Semantic search.

---

## ⚖️ License
This project is open-source and available under the MIT License.
