import json
import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from src.core.chains import create_intelligent_rag_chain
from src.utils.ollama_helpers import OllamaProvider
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

def main():
    print("Initializing Evaluation System...")
    
    # 1. Load Vector Store
    persist_dir = "data/chroma_db"
    embeddings = OllamaProvider.get_embeddings()
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    
    # 2. Initialize RAG Chain
    # Note: Phase 3 chain includes Hybrid Search + Reranking
    rag_chain = create_intelligent_rag_chain(vector_store)
    
    # 3. Load Test Set
    with open("eval/test_set.json", "r") as f:
        test_set = json.load(f)
        
    print(f"Running evaluation on {len(test_set)} examples...")
    
    results = []
    for item in test_set:
        question = item["question"]
        print(f"\nProcessing: {question}")
        
        # We need to manually invoke the chain to get context + specialized response
        # In our intelligent_rag_chain, we can't easily extract intermediate context 
        # without modifying the return type, but for evaluation, we want to see what's happening.
        
        # However, for a quick Phase 4 MVP, we'll just run the chain
        response = rag_chain.invoke(question)
        
        # To get context for Ragas, we'd ideally capture what 'intelligent_retrieval' produced.
        # For now, we'll just store the answer.
        results.append({
            "question": question,
            "answer": response,
            "ground_truth": item["ground_truth"]
        })

    # 4. Format for RAGAS
    # Ragas needs context, but our chain returns a string. 
    # To truly use RAGAS metrics like Faithfulness, we need the retrieved docs.
    # In Phase 4, we'll implement a simple display for the dashboard.
    
    df = pd.DataFrame(results)
    print("\nEvaluation Results Preview:")
    print(df[["question", "answer"]])
    
    # Note: To run full RAGAS evaluation locally with Ollama, 
    # we would need to wrap ChatOllama in RagasLLM.
    # This can be resource intensive. For this MVP, we'll stick to the preview.
    
    df.to_csv("eval/evaluation_report.csv", index=False)
    print("\nReport saved to eval/evaluation_report.csv")

if __name__ == "__main__":
    main()
