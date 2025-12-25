# Restaurant Intelligence System (RIS)

The **Restaurant Intelligence System (RIS)** is a high-performance, local-first Retrieval-Augmented Generation (RAG) platform. It transforms unstructured restaurant reviews into structured, actionable business insights.

## 🚀 Phase 1: MVP - Semantic Baseline

This phase establishes the backbone of the system using local LLMs for inference and embeddings, ensuring privacy and cost-efficiency.

### 🛠 Tech Stack
- **LLM**: Llama 3.2 (via Ollama)
- **Embeddings**: mxbai-embed-large (via Ollama)
- **Vector Database**: ChromaDB (Persistent storage)
- **Orchestration**: LangChain (Modular LCEL architecture)
- **Data Integrity**: Pydantic models for validation

### 📂 Project Structure
```text
Restaurant-Intelligence-System
├── data
│   ├── raw                # Source CSV data (Restaurant reviews.csv)
│   └── chroma_db          # Persistent vector indices and document storage
├── src
│   ├── core               # Main AI/RAG orchestration
│   │   ├── chains.py      # LangChain LCEL pipeline definitions
│   │   └── prompts.py     # System instructions and grounding logic
│   ├── data_eng           # Ingestion and processing
│   │   ├── loader.py      # Pydantic validation and CSV loading
│   │   └── ingestor.py    # Chunking and embedding workflows
│   └── utils              # System utilities
│       └── ollama_helpers.py # Model initialization and configuration
├── cli_prototype.py       # Interactive command-line interface
├── pull_models.py         # Utility to prepare local enviroment
└── requirements.txt       # Project dependencies
```

### ⚡ Quick Start

1. **Environment Setup**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare Models**:
   ```bash
   python pull_models.py
   ```

3. **Data Ingestion** (Run once to build the vector store):
   ```bash
   python cli_prototype.py --ingest
   ```

4. **Interact with the System**:
   ```bash
   # Start the interactive loop
   python cli_prototype.py
   
   # Or run a single query
   python cli_prototype.py --query "Summarize the complaints about customer service."
   ```

### 📐 Scalability & Maintenance Design
- **Validated Ingestion**: Uses Pydantic to ensure that only well-formatted data enters the system.
- **Persistent Storage**: ChromaDB ensures vector embeddings are only computed once, saving resources.
- **LCEL Chains**: The LangChain Expression Language allows for easy modification of the RAG pipeline without rewriting core logic.
- **Modular Utilities**: Ollama helpers abstract model management, making it easy to swap LLMs (e.g., Llama 3.2 to Mistral) as needed.

## 📖 Configuration Guide

The system's behavior can be customized by adjusting parameters in the following modules.

### 1. Document Processing (Chunking)
*Defined in: `src/data_eng/ingestor.py`*
- **Chunk Size (512)**: Determines the character count for each text segment. Smaller values ensure more granular retrieval, while larger values provide more comprehensive context for the LLM.
- **Chunk Overlap (51)**: Defines the character overlap between adjacent segments. This ensures semantic continuity and prevents context loss at chunk boundaries.

### 2. Information Retrieval
*Defined in: `cli_prototype.py`*
- **K Value (5)**: Specifies the number of relevant document segments retrieved for each query. 
    - *Higher values* provide more context to the LLM but increase inference latency.
    - *Lower values* optimize for speed but may omit secondary supporting details.

### 3. LLM Inference Settings
*Defined in: `src/utils/ollama_helpers.py`*
- **Temperature (0)**: Controls the randomness of the LLM's output.
    - **0 (Default)**: Optimized for RAG. Ensures deterministic, evidence-based responses.
    - **Higher values**: Increases creativity but introduces risk of factual hallucinations.
- **Model Selection**: Currently optimized for `Llama 3.2`. Compatible with other local models (e.g., `mistral`, `phi3`) via Ollama.

## 🛠 Version Control (Git)

This repository follows standard Git workflows for version management.

### 1. Operations
- **Status Verification**: `git status`
- **Commit Workflow**:
  ```bash
  git add .
  git commit -m "Brief description of changes"
  ```
- **Synchronization**:
  ```bash
  git push origin main
  git pull origin main
  ```

### 2. Environment Exclusion (`.gitignore`)
The project includes a `.gitignore` configuration to exclude:
- **Environment Artifacts**: `venv/` (Local-only dependencies).
- **Persistent Storage**: `data/chroma_db/` (Local vector indices, regenerated via `--ingest`).
- **Security Artifacts**: `.env` and other potential secrets.

---

## ⏭ Road Ahead
- **Phase 2**: Advanced metadata filtering and self-querying capability.
- **Phase 3**: Hybrid search (BM25 + Vector) and result reranking.
- **Phase 4**: Quantitative evaluation via RAGAS and Streamlit dashboard deployment.

