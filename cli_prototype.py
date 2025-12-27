import argparse
import sys
import os

from src.data_eng.ingestor import ReviewIngestor
from src.core.chains import create_rag_chain, create_intelligent_rag_chain
from langchain_chroma import Chroma
from src.utils.ollama_helpers import OllamaProvider

def run_query(rag_chain, query):
    """
    Executes a single query against the RAG chain and prints the response.
    """
    try:
        response = rag_chain.invoke(query)
        print("\nResponse:")
        print("-" * 20)
        print(response)
        print("-" * 20)
    except Exception as e:
        print(f"Error during query: {e}")

def main():
    """
    Main entry point for the CLI. Handles argument parsing and interactive loop.
    """
    parser = argparse.ArgumentParser(description="Restaurant Intelligence System (RIS) CLI")
    parser.add_argument("--ingest", action="store_true", help="Ingest data from CSV to Vector DB")
    parser.add_argument("--query", type=str, help="Single query to the RAG system")
    
    args = parser.parse_args()
    
    persist_dir = "data/chroma_db"
    
    # Data Ingestion Routine
    if args.ingest:
        print("Starting ingestion...")
        ingestor = ReviewIngestor(persist_directory=persist_dir)
        ingestor.ingest("data/raw/Restaurant reviews.csv")
        print("Ingestion complete.")
        return

    # Check for existing vector store before querying
    if not os.path.exists(persist_dir):
        print("Vector store not found. Please run with --ingest first.")
        return

    # Initialization of system components
    print("Loading Restaurant Intelligence System...")
    embeddings = OllamaProvider.get_embeddings()
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    # Configure retriever (k=5 for balance between context richness and LLM context window)
    # We now pass the vector_store to the intelligent chain which handles retrieval internally
    rag_chain = create_intelligent_rag_chain(vector_store)

    # Mode selection: Single Query vs Interactive
    if args.query:
        run_query(rag_chain, args.query)
    else:
        print("\n" + "="*40)
        print("  Restaurant Intelligence System (RIS)")
        print("="*40)
        print("Type your question and press Enter.")
        print("Type 'q' or 'quit' to exit.")
        print("-" * 40 + "\n")
        
        # Interactive session loop
        while True:
            try:
                user_input = input("Question: ").strip()
                if user_input.lower() in ['q', 'quit']:
                    print("Goodbye!")
                    break
                if not user_input:
                    continue
                
                run_query(rag_chain, user_input)
                print("\n(Type 'q' to quit)")
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                print("\nExiting...")
                break

if __name__ == "__main__":
    main()
