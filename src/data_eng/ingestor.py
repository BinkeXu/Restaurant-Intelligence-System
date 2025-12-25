from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from src.data_eng.loader import ReviewDataLoader
from langchain_core.documents import Document
import os

class ReviewIngestor:
    """
    Handles the ingestion process: loading data, chunking text, and storing in a vector database.
    Designed to be scalable and maintainable for local RAG environments.
    """
    def __init__(self, persist_directory: str, embedding_model: str = "mxbai-embed-large"):
        """
        Initializes the ingestor with a persistence directory and embedding model.
        
        Args:
            persist_directory: Folder path for ChromaDB storage.
            embedding_model: Name of the Ollama embedding model to use.
        """
        self.persist_directory = persist_directory
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        # Using RecursiveCharacterTextSplitter for optimal semantic boundary detection
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=51, # ~10% overlap to maintain context between chunks
            length_function=len,
            is_separator_regex=False,
        )

    def ingest(self, reviews_csv_path: str):
        """
        Runs the full ingestion pipeline.
        
        1. Loads reviews via ReviewDataLoader.
        2. Wraps review text in Document objects with relevant metadata.
        3. Splits documents into manageable chunks.
        4. Embeds and stores chunks in ChromaDB.
        """
        loader = ReviewDataLoader(reviews_csv_path)
        reviews = loader.load_reviews()
        
        documents = []
        for review in reviews:
            # Metadata injection is key for Phase 2 intelligent filtering.
            # We keep context in page_content and structured data in metadata.
            doc = Document(
                page_content=review.review_text,
                metadata={
                    "restaurant": review.restaurant,
                    "reviewer": review.reviewer,
                    "rating": review.rating,
                    "time": review.time
                }
            )
            documents.append(doc)
        
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        
        # Initialize vector store and persist documents
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"Ingested {len(chunks)} chunks into ChromaDB at {self.persist_directory}.")
        return vector_store

if __name__ == "__main__":
    ingestor = ReviewIngestor(persist_directory="data/chroma_db")
    ingestor.ingest("data/raw/Restaurant reviews.csv")
