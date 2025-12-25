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

## 📖 Configuration Guide (Plain English)

If you want to tune the "brain" of this system, here are the levers you can pull. No coding knowledge required!

### 1. How the AI reads (Chunking)
*Located in: `src/data_eng/ingestor.py`*
- **Chunk Size (512)**: Think of this as the size of the "sticky note" the AI uses to take notes. If it's too small, the AI loses context. If it's too large, it might get confused by too much information.
- **Chunk Overlap (51)**: This is how much two "sticky notes" share. It's like re-reading the last sentence of a page when you flip to the next one so you don't lose your train of thought.

### 2. How much the AI remembers (Retrieval)
*Located in: `cli_prototype.py`*
- **K Value (5)**: When you ask a question, the AI looks for the top 5 most relevant reviews. 
    - *Increasing this* gives the AI more evidence but might make it slower.
    - *Decreasing this* makes it faster but it might miss a crucial detail.

### 3. How the AI behaves (LLM Settings)
*Located in: `src/utils/ollama_helpers.py`*
- **Temperature (0)**: This is the "Creativity vs. Accuracy" slider.
    - **0 (Current)**: The AI is a strict librarian. It only says what's in the text.
    - **1**: The AI is a storyteller. It might start guessing or being "creative" (which we call hallucinations).
## 🛠 Version Control (Git)

This project is managed with Git. Follow these steps to collaborate or back up your changes.

### 1. Basic Git Commands
- **Check Status**: `git status` (See what has changed)
- **Save Changes**:
  ```bash
  git add .
  git commit -m "Describe your changes here"
  ```
- **Send to GitHub**: `git push origin main`
- **Get Latest Changes**: `git pull origin main`

### 2. Why use `.gitignore`?
We have configured a `.gitignore` file to ensure that:
- The **Virtual Environment** (`venv/`) is not uploaded (it's too big and unique to your PC).
- The **Local Database** (`data/chroma_db/`) is not uploaded (it will be recreated when you run `--ingest`).
- **Secret Keys** are kept safe.

---

## ⏭ Road Ahead
- **Phase 2**: Advanced metadata filtering and self-querying capability.
- **Phase 3**: Hybrid search (BM25 + Vector) and result reranking.
- **Phase 4**: Quantitative evaluation via RAGAS and Streamlit dashboard deployment.
- **Model (Llama 3.2)**: This is the version of the AI's brain. You can change this to `mistral` or `phi3` if you have them installed in Ollama.

