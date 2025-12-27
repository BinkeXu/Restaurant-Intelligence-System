from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from src.core.prompts import get_rag_prompt, get_query_translation_prompt
from src.utils.ollama_helpers import OllamaProvider
from src.retrieval.filters import ChromaFilterBuilder

def format_docs(docs):
    """
    Converts a list of documents into a single string of combined text.
    Separates documents with double newlines for clear context demarcation.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever):
    """
    Orchestrates the RAG chain using LangChain Expression Language (LCEL).
    
    The chain follows these steps:
    1. Retrieve relevant docs based on the question.
    2. Format docs into context.
    3. Pass context and question to the prompt.
    4. Execute LLM inference.
    5. Parse output to string.
    """
    llm = OllamaProvider.get_llm()
    prompt = get_rag_prompt()
    
    # LCEL allows for clear, declarative definition of the data flow.
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

def create_intelligent_rag_chain(vector_store):
    """
    Creates an advanced RAG chain that performs query translation for intelligent filtering.
    """
    llm = OllamaProvider.get_llm()
    translator_prompt = get_query_translation_prompt()
    rag_prompt = get_rag_prompt()
    
    # 1. Translator Chain: Question -> Structured JSON (filters, clean_query)
    translator_chain = translator_prompt | llm | JsonOutputParser()
    
    def intelligent_retrieval(input_data):
        """
        Executes retrieval using the extracted filters.
        """
        translation = input_data["translation"]
        clean_query = translation.get("clean_query", input_data["question"])
        filters_raw = translation.get("filters", {})
        
        # Convert simple filters to ChromaDB format
        chroma_filter = ChromaFilterBuilder.build_filter(filters_raw)
        
        # Search with filters
        docs = vector_store.similarity_search(clean_query, k=5, filter=chroma_filter)
        return {"context": format_docs(docs), "question": clean_query}

    # 2. Complete Chain
    full_chain = (
        {"translation": translator_chain, "question": RunnablePassthrough()}
        | RunnableLambda(intelligent_retrieval)
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    return full_chain
