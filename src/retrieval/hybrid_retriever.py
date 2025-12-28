from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document

class HybridRetrieverFactory:
    """
    Factory class to create a Hybrid Retriever combining Vector search and BM25.
    """
    
    @staticmethod
    def create_hybrid_retriever(vector_store, info_filters: dict = None, k: int = 5) -> EnsembleRetriever:
        """
        Initializes a BM25 retriever from a filtered subset of documents and
        combines it with the vector store's filtered retriever.
        """
        if info_filters:
            filtered_data = vector_store.get(where=info_filters)
        else:
            filtered_data = vector_store.get()
        
        documents = [
            Document(page_content=text, metadata=meta) 
            for text, meta in zip(filtered_data['documents'], filtered_data['metadatas'])
        ]
        
        # 2. Initialize Vector Retriever with filters
        search_kwargs = {"k": k}
        if info_filters:
            search_kwargs["filter"] = info_filters
            
        vector_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

        if not documents:
            return vector_retriever

        # 3. Initialize BM25 Retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = k
        
        # 4. Combine into Ensemble Retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]
        )
        
        return ensemble_retriever
