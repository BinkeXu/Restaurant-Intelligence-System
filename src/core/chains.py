from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.core.prompts import get_rag_prompt
from src.utils.ollama_helpers import OllamaProvider

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
