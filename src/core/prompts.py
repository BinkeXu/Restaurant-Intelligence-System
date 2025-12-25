from langchain_core.prompts import ChatPromptTemplate

# System prompt designed to enforce grounding and prevent hallucinations.
# LLMs are instructed to strictly follow the provided context.
RAG_SYSTEM_PROMPT = """
You are a Restaurant Intelligence Assistant. Use the provided context to answer the user's question about restaurant reviews.
If the answer is not in the context, say that you don't know based on the provided reviews. Do not hallucinate or use external knowledge.
Keep your answers professional and concise.

Context:
{context}
"""

def get_rag_prompt():
    """
    Constructs the ChatPromptTemplate for the RAG chain.
    """
    return ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
