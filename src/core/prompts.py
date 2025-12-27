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

# Prompt for translating natural language queries into structured metadata filters.
QUERY_TRANSLATION_PROMPT = """
Your task is to translate a user question about restaurant reviews into a structured JSON filter for a vector database.

Metadata fields available:
- restaurant (string): The name of the restaurant.
- rating (float): The star rating of the review (1.0 to 5.0).
- has_timestamp (boolean): Whether the review has a valid timestamp.

Operators available: $eq, $ne, $gt, $gte, $lt, $lte.

Return ONLY a JSON object with two keys:
1. "filters": A dictionary of metadata filters. Use the operators above if needed.
2. "clean_query": The original question stripped of the filter-specific parts to use for semantic search.

Example 1:
Question: "What are the 1-star reviews saying for Beyond Flavours?"
Response: {{
    "filters": {{
        "restaurant": "Beyond Flavours",
        "rating": {{"$eq": 1.0}}
    }},
    "clean_query": "What are the reviews saying?"
}}

Example 2:
Question: "General sentiment about food quality?"
Response: {{
    "filters": {{}},
    "clean_query": "General sentiment about food quality?"
}}

Question: {question}
Response:
"""

def get_query_translation_prompt():
    """
    Constructs the prompt for query translation.
    """
    return ChatPromptTemplate.from_messages([
        ("system", "You are a Query Translator. Output JSON only."),
        ("human", QUERY_TRANSLATION_PROMPT),
    ])
