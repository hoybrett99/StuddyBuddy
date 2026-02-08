# frontend/streamlit_app.py

"""
Study Buddy - Streamlit Web Interface
A beautiful, interactive UI for your RAG-based study assistant.
"""

import streamlit as st
import requests
from pathlib import Path
import time
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
API_URL = "http://127.0.0.1:8000"  # Your FastAPI backend

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Study Buddy 📚",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS for better styling
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-box {
        padding: 2rem;
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        text-align: center;
        background-color: #f0f8ff;
    }
    .stat-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
    }
    .source-box {
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #52c41a;
        background-color: #f6ffed;
        border-radius: 4px;
        color: #000000 !important;  /* Force black text */
    }
    .source-box strong {
        color: #000000 !important;  /* Force black text for bold */
    }
    /* Fix for Streamlit's default text color in expanders */
    .streamlit-expanderContent {
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Helper Functions
# ============================================================================

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def upload_file(file):
    """Upload a file to the API."""
    files = {"file": (file.name, file, file.type)}
    response = requests.post(f"{API_URL}/upload", files=files)
    return response.json()

def query_api(question, num_contexts=4):
    """Send a query to the API."""
    data = {
        "question": question,
        "num_contexts": num_contexts
    }
    response = requests.post(f"{API_URL}/query", json=data)
    return response.json()

def get_stats():
    """Get system statistics."""
    response = requests.get(f"{API_URL}/stats")
    return response.json()

# ============================================================================
# Session State Initialization
# ============================================================================
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

# ============================================================================
# Sidebar
# ============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # API Health Check
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Not Running")
        st.info("Start the API with:\n```\nuvicorn app.main:app --reload\n```")
    
    st.markdown("---")
    
    # Number of context chunks
    num_contexts = st.slider(
        "Context Chunks",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of relevant chunks to retrieve for each query"
    )
    
    st.markdown("---")
    
    # System Stats
    st.markdown("## 📊 Statistics")
    if st.button("Refresh Stats"):
        try:
            stats = get_stats()
            st.metric("Total Documents", stats.get('total_documents', 0))
            st.metric("Total Chunks", stats.get('total_chunks', 0))
            st.metric("Queries Processed", stats.get('total_queries_processed', 0))
        except:
            st.error("Could not fetch statistics")
    
    st.markdown("---")
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.info("📚 Study Buddy uses RAG (Retrieval-Augmented Generation) to help you study from your uploaded materials.")

# ============================================================================
# Main Content
# ============================================================================

# Header
st.markdown('<h1 class="main-header">📚 Study Buddy</h1>', unsafe_allow_html=True)
st.markdown("### Your AI-powered study assistant")

# ============================================================================
# Tab Layout
# ============================================================================
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📤 Upload Documents", "📖 Library"])

# ============================================================================
# TAB 1: Chat Interface
# ============================================================================
with tab1:
    st.markdown("### Ask questions about your study materials")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📎 View Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        # Create columns for better layout
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**Source {i}:** `{source['document_name']}`")
                        
                        with col2:
                            st.metric("Relevance", f"{source['relevance_score']:.3f}")
                        
                        # Display the chunk text
                        if source.get('chunk_text'):
                            st.text_area(
                                f"Text from chunk",
                                value=source['chunk_text'],
                                height=150,
                                key=f"chunk_{source['chunk_id']}_{i}",
                                disabled=True  # Read-only
                            )
                        
                        st.divider()  # Add separator between sources
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your study materials..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    response = query_api(prompt, num_contexts)
                    answer = response['answer']
                    sources = response.get('sources', [])
                    query_time = response.get('query_time_seconds', 0)
                    
                    st.markdown(answer)
                    st.caption(f"⏱️ Answered in {query_time}s")
                    
                    # Display sources with chunk text
                    if sources:
                        with st.expander("📎 View Sources", expanded=False):
                            for i, source in enumerate(sources, 1):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**Source {i}:** `{source['document_name']}`")
                                
                                with col2:
                                    st.metric("Relevance", f"{source['relevance_score']:.3f}")
                                
                                # Display the chunk text
                                if source.get('chunk_text'):
                                    st.text_area(
                                        f"Text from chunk",
                                        value=source['chunk_text'],
                                        height=150,
                                        key=f"chunk_new_{source['chunk_id']}_{i}",
                                        disabled=True
                                    )
                                
                                st.divider()
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

# ============================================================================
# TAB 2: Upload Documents
# ============================================================================
with tab2:
    st.markdown("### Upload your study materials")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'txt', 'docx'],
            help="Upload PDF, TXT, or DOCX files"
        )
        
        if uploaded_file is not None:
            # Show file details
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            st.write(f"**Type:** {uploaded_file.type}")
            
            # Upload button
            if st.button("📤 Upload and Process", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        # Upload to API
                        result = upload_file(uploaded_file)
                        
                        if result.get('success'):
                            st.success(f"✅ {result['message']}")
                            st.info(f"Created {result['chunks_created']} chunks")
                            
                            # Add to uploaded files list
                            st.session_state.uploaded_files.append({
                                'filename': result['filename'],
                                'document_id': result['document_id'],
                                'chunks': result['chunks_created'],
                                'uploaded_at': datetime.now()
                            })
                        else:
                            st.error(f"❌ Upload failed: {result.get('message', 'Unknown error')}")
                    
                    except Exception as e:
                        st.error(f"❌ Error uploading file: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.info("""
        **Supported formats:**
        - 📄 PDF (.pdf)
        - 📝 Text (.txt)
        - 📃 Word (.docx)
        
        **Tips:**
        - Upload textbooks, notes, or study guides
        - Files are automatically chunked and indexed
        - Ask questions after uploading
        """)

# ============================================================================
# TAB 3: Library
# ============================================================================
with tab3:
    st.markdown("### Your Document Library")
    
    if st.session_state.uploaded_files:
        for i, doc in enumerate(st.session_state.uploaded_files):
            with st.expander(f"📄 {doc['filename']}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Chunks", doc['chunks'])
                
                with col2:
                    st.write(f"**ID:** `{doc['document_id'][:12]}...`")
                
                with col3:
                    st.write(f"**Uploaded:** {doc['uploaded_at'].strftime('%Y-%m-%d %H:%M')}")
    else:
        st.info("📭 No documents uploaded yet. Go to the 'Upload Documents' tab to get started!")

# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using FastAPI, Claude, and Streamlit | Study Buddy v0.0.1"
    "</div>",
    unsafe_allow_html=True
)