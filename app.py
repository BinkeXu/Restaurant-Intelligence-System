import streamlit as st
import os
import json
from src.core.chains import create_intelligent_rag_chain
from src.utils.ollama_helpers import OllamaProvider
from src.retrieval.filters import ChromaFilterBuilder
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from src.core.prompts import get_query_translation_prompt

# --- Page Config ---
st.set_page_config(
    page_title="Restaurant Intelligence System (RIS)",
    page_icon="🤖",
    layout="wide"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #fcfcfc;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .source-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 10px;
        font-size: 0.9em;
    }
    .metric-label {
        font-weight: bold;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🤖 Restaurant Intelligence System (RIS)")
st.caption("A production-ready RAG platform for restaurant review analysis. Built with LangChain, Ollama, and ChromaDB.")
st.markdown("---")

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3448/3448610.png", width=80) 
    st.header("⚙️ System Config")
    model_choice = st.selectbox("LLM Model", ["llama3.2", "mistral", "phi3"], index=0)
    top_k = st.slider("Retrieval Depth (K)", 1, 10, 5)
    temp = st.slider("Creativity (Temp)", 0.0, 1.0, 0.0, 0.1)
    
    st.markdown("---")
    st.subheader("🔍 Inspection Mode")
    show_filters = st.checkbox("Show Query Translation", value=True)
    show_sources = st.checkbox("Show Source Documents", value=True)
    
    st.markdown("---")
    st.success("✅ Hybrid Search: Active")
    st.success("✅ FlashRank: Active")

# --- System Initialization ---
@st.cache_resource
def get_vector_store():
    persist_dir = "data/chroma_db"
    if not os.path.exists(persist_dir):
        return None
        
    embeddings = OllamaProvider.get_embeddings()
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

vector_store = get_vector_store()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ex: What do people say about the fish at Beyond Flavours?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if vector_store is None:
            st.warning("⚠️ **Vector store not found!** Please run the ingestion process first.")
            st.code("python cli_prototype.py --ingest")
        else:
            # Inspection flags from sidebar
            inspection_enabled = show_filters or show_sources
            
            with st.spinner("Analyzing reviews..." if not inspection_enabled else None):
                try:
                    # 1. Internal Logic (Always runs, but only displays if requested)
                    translator_prompt = get_query_translation_prompt()
                    llm = OllamaProvider.get_llm(temperature=0)
                    translator_chain = translator_prompt | llm | JsonOutputParser()
                    translation = translator_chain.invoke({"question": prompt})

                    # Show Translation if requested
                    if show_filters:
                        with st.expander("🔄 Query Translation Details", expanded=True):
                            cols = st.columns(2)
                            with cols[0]:
                                st.json(translation.get("filters", {}))
                            with cols[1]:
                                st.info(f"Clean Query: {translation.get('clean_query', prompt)}")

                    # 2. Execute RAG Chain
                    dynamic_chain = create_intelligent_rag_chain(
                        vector_store, 
                        k=top_k, 
                        temperature=temp
                    )
                    
                    response = dynamic_chain.invoke(prompt)
                    
                    # Display Final Answer
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # 3. Show Source Documents if requested
                    if show_sources:
                        with st.expander("📚 Source Documents", expanded=False):
                            from src.retrieval.hybrid_retriever import HybridRetrieverFactory
                            from langchain.retrievers import ContextualCompressionRetriever
                            from langchain.retrievers.document_compressors import FlashrankRerank
                            
                            chroma_filter = ChromaFilterBuilder.build_filter(translation.get("filters", {}))
                            hybrid_retriever = HybridRetrieverFactory.create_hybrid_retriever(
                                vector_store, info_filters=chroma_filter, k=top_k*2
                            )
                            compressor = FlashrankRerank(top_n=top_k)
                            compression_retriever = ContextualCompressionRetriever(
                                base_compressor=compressor, base_retriever=hybrid_retriever
                            )
                            docs = compression_retriever.invoke(translation.get("clean_query", prompt))
                            
                            for i, doc in enumerate(docs):
                                st.markdown(f"""
                                <div class="source-card">
                                    <b>Source #{i+1} - {doc.metadata.get('restaurant', 'Unknown')}</b><br>
                                    <i>Reviewer: {doc.metadata.get('reviewer', 'Anonymous')} ({doc.metadata.get('rating')} stars)</i><br><br>
                                    "{doc.page_content}"
                                </div>
                                """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# --- Footer ---
st.markdown("---")
footer_col1, footer_col2 = st.columns([2, 1])
with footer_col1:
    st.markdown("Designed for Business Intelligence & Restaurant Management.")
with footer_col2:
    st.markdown("Local RAG Stack v1.0.0")
