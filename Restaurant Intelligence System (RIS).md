Project Plan Restaurant Intelligence System (RIS)
1. Project Overview
The Restaurant Intelligence System (RIS) is a high-performance, local-first Retrieval-Augmented Generation (RAG) platform. It is designed to transform unstructured restaurant reviews into structured, actionable business insights. This project serves as a showcase of ML engineering, advanced retrieval strategies, and LLM orchestration for a Master’s level portfolio.

Core Tech Stack
LLM Llama 3.2 (Local via Ollama)

Embeddings xbai-embed-large

Orchestration LangChain (LCEL)

Vector Database ChromaDB

Evaluation RAGAS Framework

Deployment Streamlit & Docker

Dataset avaliable at: https://github.com/manthanpatel98/Restaurant-Review-Sentiment-Analysis/blob/master/Restaurant%20reviews.csv

2. Proposed Project Structure
Plaintext

Restaurant-Intelligence-System
├── data
│   ├── raw                # restaurant_reviews.csv
│   └── chroma_db          # Persistent vector indices
├── src
│   ├── core               # Main RAG logic
│   │   ├── chains.py       # LangChain LCEL definitions
│   │   └── prompts.py      # Grounding and system prompts
│   ├── data_eng           # Data cleaning and ingestion
│   │   ├── loader.py       # CSV loading & Pydantic validation
│   │   └── ingestor.py     # Chunking and embedding logic
│   ├── retrieval          # Search logic
│   │   ├── hybrid.py       # BM25 + Vector fusion (RRF)
│   │   └── filters.py      # Metadata filter builders
│   └── utils              # Ollama helpers and logging
├── eval                   # Evaluation scripts
│   ├── test_set.json       # Ground truth Q&A pairs
│   └── run_ragas.py        # RAGAS metric calculations
├── app.py                  # Streamlit Dashboard
├── requirements.txt        # Project dependencies
├── Dockerfile              # Containerization
└── README.md               # Technical documentation
3. Implementation Phases
Phase 1 MVP - The Semantic Baseline
Objective Establish a stable Text-In, Answer-Out loop to verify the basic RAG pipeline.

Data Preparation Load CSV and sanitize Review_Text.

Chunking Strategy Use RecursiveCharacterTextSplitter with a chunk size of 512 and 10% overlap to maintain semantic continuity.

Vector Initialization Embed text using xbai-embed-large and store in ChromaDB.

Grounded Generation Implement a system prompt that forces Llama 3.2 to answer only based on the provided context to minimize hallucinations.

Outcome A functional CLI prototype capable of answering general questions about the review corpus.

Phase 2 Metadata & Data Robustness
Objective Leverage structured data columns to move from General Search to Intelligent Filtering.

Metadata Mapping Inject Restaurant, Reviewer, and Review Rating into the ChromaDB metadata layer.

Imputation Strategy Address missing Time values using a Boolean flag (has_timestamp) to prevent errors during temporal queries.

Self-Querying Build a logical router where the LLM parses natural language into metadata filters (e.g., Find all 1-star reviews for Burger King).

Outcome Precision-targeted retrieval based on specific restaurant attributes and ratings.

Phase 3 Advanced Retrieval (Hybrid & Rerank)
Objective Solve the Proper Noun problem and optimize context quality.

Hybrid Search Combine xbai-embed-large (Dense) with BM25 (SparseKeyword) search to capture exact dish names and entities.

Reciprocal Rank Fusion (RRF) Merge results from both search methods using the RRF algorithm to prioritize documents found by both.

Reranking Implement a secondary scoring step (Cross-Encoder) to re-order the top 10 results, passing only the most relevant 3-5 chunks to the LLM.

Outcome A high-precision engine that handles niche queries (e.g., specific menu items) with high accuracy.

Phase 4 Quantitative Evaluation (RAGAS)
Objective Validate the system using industry-standard ML metrics.

Golden Dataset Create a set of 20+ ground-truth Q&A pairs.

RAGAS Implementation Calculate Faithfulness, Answer Relevance, and Context Precision.

Deployment Wrap the application in a Streamlit dashboard and containerize using Docker for reproducibility.

Outcome A production-ready portfolio piece with documented performance metrics (e.g., Faithfulness Score 0.92).

4. Engineering Standards & Validation
Data Integrity Use Pydantic for schema enforcement during ingestion.

Performance Log latency (Time-to-First-Token) for local inference.

Persistence Ensure the vector store is persistent to avoid redundant embedding coststime.