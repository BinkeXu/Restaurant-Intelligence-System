from langchain_ollama import ChatOllama, OllamaEmbeddings

class OllamaProvider:
    """
    Centralized provider for Ollama-based LLMs and Embeddings.
    Facilitates easy swapping of models and configuration.
    """
    @staticmethod
    def get_llm(model: str = "llama3.2", temperature: float = 0):
        """
        Returns a ChatOllama instance. 
        """
        return ChatOllama(model=model, temperature=temperature)

    @staticmethod
    def get_embeddings(model: str = "mxbai-embed-large"):
        """
        Returns an OllamaEmbeddings instance for text vectorization.
        """
        return OllamaEmbeddings(model=model)
