import streamlit as st
import os
from dotenv import load_dotenv
import tempfile
import traceback
from utils.advanced_data_ingest import load_and_process_advanced, search_arxiv_papers, download_arxiv_paper
from utils.vector_store import build_vector_db, save_vector_db, load_vector_db, add_to_vector_db, get_source_documents
from utils.advanced_rag_chain import create_advanced_rag_chain, process_specialized_query
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Advanced RAG Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
    }

    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px #aaa;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .chat-message {
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 1rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in-out;
        position: relative;
    }

    .user-message {
        background-color: #DCF8C6;
        margin-left: auto;
        width: 80%;
        border-radius: 1rem 1rem 0 1rem;
    }

    .bot-message {
        background-color: #ECE5DD;
        width: 80%;
        border-radius: 1rem 1rem 1rem 0;
    }
    
    .chat-message strong {
        font-weight: 700;
    }

    .sidebar .st-emotion-cache-10oheav {
        padding-top: 2rem;
    }
    
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }

</style>
""", unsafe_allow_html=True)

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state["conversations"] = {} # { "conversation_name": [ (q,a), ... ] }
if "current_conversation" not in st.session_state:
    st.session_state["current_conversation"] = "new"
    st.session_state["conversations"]["new"] = []

if "rag_chain" not in st.session_state:
    st.session_state["rag_chain"] = None
if "document_processed" not in st.session_state:
    st.session_state["document_processed"] = False
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None

# Directory to store vector DBs
VECTOR_DB_DIR = "vector_dbs"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)
DOWNLOADS_DIR = "downloads"
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

# Auto-load existing vector DB on startup
COMBINED_DB_PATH = os.path.join(VECTOR_DB_DIR, "combined_vector_db")
if os.path.exists(os.path.join(COMBINED_DB_PATH, "chroma-collections.parquet")):
    try:
        st.session_state["vectorstore"] = load_vector_db(COMBINED_DB_PATH)
        st.session_state["rag_chain"] = create_advanced_rag_chain(st.session_state["vectorstore"])
        st.session_state["document_processed"] = True
        st.sidebar.success("📚 Loaded existing vector database.")
        
        # Display loaded documents
        loaded_docs = get_source_documents(st.session_state["vectorstore"])
        if loaded_docs:
            st.sidebar.markdown("---")
            st.sidebar.markdown('<h3 style="color: #4A4A4A; font-size: 1.2rem; margin-bottom: 1rem;">Loaded Documents</h3>', unsafe_allow_html=True)
            for doc_name in loaded_docs:
                st.sidebar.markdown(f"- {os.path.basename(doc_name)}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading existing vector database: {str(e)}")
        st.session_state["document_processed"] = False

# Sidebar
st.sidebar.markdown('<h2 style="color: #4A4A4A; font-size: 1.8rem; margin-bottom: 0.5rem;">QA-AI-Agent Pro</h2>', unsafe_allow_html=True)
st.sidebar.markdown('<p style="color: #888888; font-size: 0.9rem; margin-bottom: 2rem;">Your intelligent research companion</p>', unsafe_allow_html=True)

# API Key check
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
if not openrouter_api_key:
    st.sidebar.error(f"⚠️ OPENROUTER_API_KEY not found in environment variables!")
    st.sidebar.info(f"Please add your OPENROUTER_API_KEY to the .env file")

# File uploader
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key="doc_uploader",
    help="Add PDF research papers and DOCX documents"
)

st.sidebar.markdown("---")

# Arxiv Search Section
st.sidebar.markdown('<h3 style="color: #4A4A4A; font-size: 1.2rem; margin-bottom: 1rem;">Arxiv Paper Search</h3>', unsafe_allow_html=True)
arxiv_query = st.sidebar.text_input("Search Arxiv", placeholder="Enter keywords or paper title")
if st.sidebar.button("Search Papers"):
    if arxiv_query:
        try:
            with st.spinner("Searching Arxiv..."):
                papers = search_arxiv_papers(arxiv_query, max_results=5)
                st.session_state["arxiv_results"] = papers
        except Exception as e:
            st.sidebar.error(f"Error searching Arxiv: {str(e)}")
    else:
        st.sidebar.warning("Please enter a search query")

# Display Arxiv results if available
if "arxiv_results" in st.session_state and st.session_state["arxiv_results"]:
    st.sidebar.markdown("---")
    st.sidebar.markdown('<h3 style="color: #4A4A4A; font-size: 1.2rem; margin-bottom: 1rem;">Search Results</h3>', unsafe_allow_html=True)
    
    for i, paper in enumerate(st.session_state["arxiv_results"]):
        with st.sidebar.expander(f"📄 {paper['title'][:50]}..."):
            st.markdown(f"**Authors:** {', '.join(paper['authors'][:3])}")
            st.markdown(f"**Published:** {paper['published']}")
            st.markdown(f"**Abstract:** {paper['abstract'][:200]}...")
            
            if st.button(f"Download Paper {i+1}", key=f"download_{i}"):
                try:
                    with st.spinner("Downloading paper..."):
                        filepath = download_arxiv_paper(paper['entry_id'].split('/')[-1], DOWNLOADS_DIR)
                        st.success(f"Downloaded: {os.path.basename(filepath)}")
                        # Add to uploaded files to process
                        st.session_state["downloaded_paper"] = filepath
                        st.rerun()
                except Exception as e:
                    st.error(f"Error downloading paper: {str(e)}")

st.sidebar.markdown("---")

st.sidebar.markdown('<h3 style="color: #4A4A4A; font-size: 1.2rem; margin-bottom: 1rem;">Recent Conversations</h3>', unsafe_allow_html=True)

# Display recent conversations
for convo_name in st.session_state["conversations"]:
    if convo_name != "new":
        if st.sidebar.button(convo_name, key=f"convo_{convo_name}"):
            st.session_state["current_conversation"] = convo_name

st.sidebar.markdown("---")

# New Conversation button
if st.sidebar.button("+ New Conversation", key="new_convo"):
    # Create a placeholder for a new conversation
    st.session_state["current_conversation"] = "new"
    if "new" not in st.session_state["conversations"]:
        st.session_state["conversations"]["new"] = []

# Clear Conversations button
if st.sidebar.button("🗑️ Clear Conversations", key="clear_chat"):
    st.session_state["conversations"] = {"new": []}
    st.session_state["current_conversation"] = "new"

# Document processing
if uploaded_files or "downloaded_paper" in st.session_state:
    # Handle downloaded papers
    all_files = []
    if "downloaded_paper" in st.session_state:
        all_files.append(st.session_state["downloaded_paper"])
        del st.session_state["downloaded_paper"]
    
    # Handle uploaded files
    temp_file_paths = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to temporary location
        # Preserve the original file extension
        file_extension = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
            temp_file_paths.append(tmp_path)
            all_files.append(tmp_path)
    
    # Use a fixed directory for combined vector DB for simplicity
    persist_directory = os.path.join(VECTOR_DB_DIR, "combined_vector_db")
    
    try:
        with st.sidebar.container():
            st.info("📄 Processing documents...")
            progress_bar = st.progress(0)
            
            all_docs = []
            
            for i, file_path in enumerate(all_files):
                progress_bar.progress(int((i + 1) / len(all_files) * 50))
                
                docs = load_and_process_advanced(file_path)
                all_docs.extend(docs)
            
            progress_bar.progress(75)
            
            # Check if ChromaDB already exists for this combined set
            chroma_exists = os.path.exists(os.path.join(persist_directory, "chroma-collections.parquet"))
            if chroma_exists:
                vectorstore = load_vector_db(persist_directory)
                st.sidebar.info("♻️ Loaded existing vector database.")
                progress_bar.progress(85)
            else:
                vectorstore = build_vector_db(all_docs, persist_directory)
                save_vector_db(vectorstore) # Save the vector DB
                st.sidebar.info("💾 Vector database built and saved.")
                progress_bar.progress(85)
            
            st.session_state["vectorstore"] = vectorstore
            rag_chain = create_advanced_rag_chain(vectorstore)
            st.session_state["rag_chain"] = rag_chain
            progress_bar.progress(100)
            
            # Clean up temporary files
            for tmp_path in temp_file_paths:
                os.unlink(tmp_path)
            
            st.session_state["document_processed"] = True
            st.sidebar.success(f"✅ {len(all_files)} document(s) processed and added to the database!")
            
    except Exception as e:
        st.sidebar.error(f"❌ Error processing documents: {str(e)}")
        st.sidebar.error("Please check your files and try again.")
        # Clean up temporary files in case of error
        for tmp_path in temp_file_paths:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

# Main content area
if not st.session_state["document_processed"]:
    st.markdown("""
    <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; min-height: 65vh;">
        <h1 style="color: #4A4A4A; margin-bottom: 0.5rem; font-size: 1.5rem;">Welcome to <span style="color: #6c5ce7;">QA-AI-Agent Pro</span></h1>
        <p style="color: #888888; font-size: 0.95rem; max-width: 420px; text-align: center; margin-bottom: 1.2rem;">
            Your intelligent research companion powered by advanced AI.<br>
            Upload research papers, search Arxiv, ask questions, and get personalized insights.
        </p>
        <div style="margin: 1.2rem 0;">
            <iframe src="https://lottie.host/embed/d9f7f1ad-e638-4f69-9786-ab3c7a6f4d27/4nGGcizM77.lottie" style="width:180px;height:180px;border:none;display:block;margin:auto;"></iframe>
        </div>
        <input type="text" placeholder="Ask me anything about your research..." disabled style="margin-top:1rem; width:280px; padding:0.5rem; border-radius:8px; border:1px solid #ccc; background:#f5f5f5; text-align:center; font-size:0.9rem;">
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
        .stTextInput [data-testid="stFormSubmitButton"] {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Display Arxiv results on main screen if available
    if "arxiv_results" in st.session_state and st.session_state["arxiv_results"]:
        st.markdown("---")
        st.markdown('<h2 style="color: #4A4A4A; font-size: 1.5rem; margin-bottom: 1rem;">Arxiv Search Results</h2>', unsafe_allow_html=True)
        
        for i, paper in enumerate(st.session_state["arxiv_results"]):
            with st.expander(f"📄 {paper['title']}", expanded=False):
                st.markdown(f"**Authors:** {', '.join(paper['authors'][:3])}")
                st.markdown(f"**Published:** {paper['published']}")
                st.markdown(f"**Abstract:** {paper['abstract']}")
                
                if st.button(f"Download Paper {i+1}", key=f"main_download_{i}"):
                    try:
                        with st.spinner("Downloading paper..."):
                            filepath = download_arxiv_paper(paper['entry_id'].split('/')[-1], DOWNLOADS_DIR)
                            st.success(f"Downloaded: {os.path.basename(filepath)}")
                            # Add to uploaded files to process
                            st.session_state["downloaded_paper"] = filepath
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error downloading paper: {str(e)}")
    
else:
    st.markdown("""
    <div style="display: flex; flex-direction: column; justify-content: flex-start; align-items: center; min-height: 60vh; width: 100%;">
        <h2 style="color: #4A4A4A; font-size: 1.8rem; margin-bottom: 0.5rem;">Research Assistant</h2>
        <p style="color: #888888; font-size: 0.9rem; margin-bottom: 2rem;">Ask me anything about your research papers</p>
        <div style="width: 100%; max-width: 700px; flex-grow: 1; display: flex; flex-direction: column;">
    """, unsafe_allow_html=True)
    
    # Display chat history
    chat_history = st.session_state["conversations"].get(st.session_state["current_conversation"], [])
    chat_container = st.container()
    with chat_container:
        for i, (question, answer) in enumerate(chat_history):
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>🙋 You:</strong> {question}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant:</strong> {answer}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Query input at the bottom, centered
    user_question = st.chat_input(
        "Ask me anything about your research papers...",
        key="chat_input"
    )
    
    # Process query
    if user_question and st.session_state.get("rag_chain"):
        # If this is the first message in a new conversation, use it as the name
        if st.session_state["current_conversation"] == "new":
            # Rename the conversation with the first question
            st.session_state["conversations"][user_question] = st.session_state["conversations"].pop("new")
            st.session_state["current_conversation"] = user_question
        
        try:
            with st.spinner("🤔 Thinking..."):
                result = process_specialized_query(st.session_state["rag_chain"], user_question)
                answer = result["result"]
                
                # Add to chat history
                st.session_state["conversations"][st.session_state["current_conversation"]].append((user_question, answer))
                
                # Add conversation to vector store
                from langchain.schema import Document
                qa_docs = [
                    Document(page_content=f"User Question: {user_question}"),
                    Document(page_content=f"Assistant Answer: {answer}")
                ]
                add_to_vector_db(st.session_state["vectorstore"], qa_docs)
                
                # Display latest response
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🙋 You:</strong> {user_question}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Assistant:</strong> {answer}
                </div>
                """, unsafe_allow_html=True)
                
                # Show source documents if available
                if "source_documents" in result and result["source_documents"]:
                    with st.expander("📖 View Sources"):
                        for i, doc in enumerate(result["source_documents"][:3]):
                            st.markdown(f"**Source {i + 1}:**")
                            st.text(doc.page_content[:300] + "...")
        except Exception as e:
            st.error(f"❌ Error processing your question: {str(e)}")
            st.error("Please try rephrasing your question or check your API key.")
