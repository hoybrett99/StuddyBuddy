# frontend/streamlit_app.py

import streamlit as st
import requests
from datetime import datetime
import time
import pandas as pd
import re

# ============================================================================
# Configuration
# ============================================================================

API_URL = "http://127.0.0.1:8000"

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Study Buddy - AI Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
    <style>
    /* Use full viewport width */
    .main .block-container {
        max-width: none;
        padding: 2rem 3rem;
    }
    
    /* Upload box styling */
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: rgba(76, 175, 80, 0.1);
    }
    
    /* Stat card styling */
    .stat-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Conversation starter buttons */
    .stButton button {
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# Session State Initialization
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Agent conversation history for conversational memory
if 'agent_conversation' not in st.session_state:
    st.session_state.agent_conversation = []

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

def upload_document(file):
    """Upload a document to the API."""
    files = {"file": (file.name, file.getvalue(), file.type)}
    response = requests.post(f"{API_URL}/upload", files=files)
    return response

def query_api(question, num_contexts=4, use_agent_mode=False, conversation_history=None):
    """
    Query the API with optional agent mode and conversation history.
    
    Args:
        question: User's question
        num_contexts: Number of context chunks to retrieve
        use_agent_mode: Whether to use AI agent (smarter) or basic RAG
        conversation_history: Previous conversation for context (agent mode only)
        
    Returns:
        dict: Response with answer, sources, etc.
    """
    if use_agent_mode:
        endpoint = f"{API_URL}/agent/query"
    else:
        endpoint = f"{API_URL}/query"
    
    payload = {
        "question": question,
        "num_contexts": num_contexts,
        "conversation_history": conversation_history if use_agent_mode else None
    }
    
    response = requests.post(endpoint, json=payload)
    return response.json()

def get_stats():
    """Get system statistics from the API."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Stats endpoint returned status {response.status_code}")
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "total_queries": 0
            }
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "total_queries": 0
        }

# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.title("📚 Study Buddy")
    st.markdown("---")
    
    # API Health Check
    api_status = check_api_health()
    if api_status:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.info("Make sure the FastAPI server is running:\n```\nuvicorn app.main:app --reload\n```")
    
    st.markdown("---")
    
    # System Stats
    st.subheader("📊 System Stats")
    stats = get_stats()
    if stats:
        st.metric("Total Documents", stats.get('total_documents', 0))
        st.metric("Total Chunks", stats.get('total_chunks', 0))
        st.metric("Total Queries", stats.get('total_queries', 0))
    else:
        st.warning("Unable to load stats")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    num_contexts = st.slider(
        "Context chunks per query",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of relevant text chunks to use for answering questions"
    )
    
    st.markdown("---")
    
    # AI Agent Toggle
    st.subheader("🤖 AI Agent")
    
    use_agent = st.toggle(
        "Enable Smart Agent",
        value=True,
        help="AI agent can handle complex questions, comparisons, practice questions, and remembers conversation context"
    )
    
    if use_agent:
        st.success("✨ Agent Mode: ON")
        st.caption("✓ Conversational memory")
        st.caption("✓ Smart comparisons")
        st.caption("✓ Practice questions")
        
        # Show conversation stats
        if len(st.session_state.agent_conversation) > 0:
            turns = len(st.session_state.agent_conversation) // 2
            st.metric("Conversation turns", turns)
            
            if turns > 3:
                st.info("💡 Tip: Clear chat when switching topics for better focus!")
    else:
        st.info("📚 Basic Mode: ON")
        st.caption("Faster, simpler RAG search")
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About")
    st.markdown("""
    **Study Buddy** is an AI-powered learning assistant that helps you study more effectively.
    
    **Features:**
    - 💬 Conversational Q&A
    - 📚 Document upload (PDF, TXT, DOCX)
    - 🔍 Smart search with citations
    - 🤖 AI agent for complex queries
    - 📝 Practice question generation
    
    Built with Claude, FastAPI, and ChromaDB.
    """)

# ============================================================================
# Main Content
# ============================================================================

st.title("📚 Study Buddy - AI Learning Assistant")
st.markdown("Upload your study materials and have natural conversations about what you're learning!")

# ============================================================================
# Tabs
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📤 Upload", "📖 Library", "🔍 Preview"])

# ============================================================================
# TAB 1: Chat Interface
# ============================================================================

with tab1:
    st.markdown("### 💬 Chat with Study Buddy")
    
    # Show conversation starters if chat is empty
    if not st.session_state.messages:
        st.markdown("**👋 Hi! I'm Study Buddy, your AI learning assistant.**")
        st.markdown("Ask me anything about your uploaded study materials. I can help you:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📖 Learn**")
            st.markdown("- Explain concepts")
            st.markdown("- Define terms")
            st.markdown("- Clarify confusion")
        
        with col2:
            st.markdown("**🔍 Compare**")
            st.markdown("- Find differences")
            st.markdown("- Identify similarities")
            st.markdown("- Contrast topics")
        
        with col3:
            st.markdown("**✏️ Practice**")
            st.markdown("- Generate quizzes")
            st.markdown("- Test knowledge")
            st.markdown("- Review material")
        
        st.markdown("---")
        
        st.markdown("**💡 Try these conversation starters:**")
        
        starter_col1, starter_col2 = st.columns(2)
        
        with starter_col1:
            st.button("📝 What are mitochondria?", key="starter_1", disabled=True, use_container_width=True)
            st.button("🔬 Explain photosynthesis simply", key="starter_2", disabled=True, use_container_width=True)
        
        with starter_col2:
            st.button("🆚 Compare plant and animal cells", key="starter_3", disabled=True, use_container_width=True)
            st.button("📚 Quiz me on cell organelles", key="starter_4", disabled=True, use_container_width=True)
        
        st.markdown("---")
        st.info("💡 **Tip:** If using Agent Mode, I'll remember our conversation! You can ask follow-up questions like 'tell me more' or 'what about chloroplasts?'")
    
    # Display chat messages
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if message["role"] == "assistant" and "sources" in message:
                if message["sources"]:
                    with st.expander("📎 View Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            # Source header
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"### 📄 Source {i}")
                                st.caption(f"`{source['document_name']}`")
                            
                            with col2:
                                st.metric("Relevance", f"{source['relevance_score']:.3f}")
                            
                            with col3:
                                st.caption("Chunk ID")
                                st.code(source['chunk_id'][-8:], language=None)
                            
                            # Display chunk text if available
                            if source.get('chunk_text'):
                                with st.expander("📖 View chunk content", expanded=False):
                                    st.text_area(
                                        "Chunk text:",
                                        value=source['chunk_text'],
                                        height=200,
                                        label_visibility="collapsed",
                                        disabled=True,
                                        key=f"chunk_msg{msg_idx}_src{i}_{source['chunk_id'][:8]}"
                                    )
                            
                            # Add divider between sources
                            if i < len(message["sources"]):
                                st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your study materials..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Track in agent conversation history if agent mode is on
        if use_agent:
            st.session_state.agent_conversation.append({
                "role": "user",
                "content": prompt
            })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    # Prepare conversation history for agent
                    conv_history = None
                    if use_agent and len(st.session_state.agent_conversation) > 1:
                        # Send conversation history (exclude current message)
                        conv_history = st.session_state.agent_conversation[:-1]
                    
                    # Call API with conversation history
                    response = query_api(
                        prompt, 
                        num_contexts, 
                        use_agent_mode=use_agent,
                        conversation_history=conv_history
                    )
                    
                    answer = response['answer']
                    sources = response.get('sources', [])
                    query_time = response.get('query_time_seconds', 0)
                    
                    st.markdown(answer)
                    
                    # Show query time and conversation indicator
                    time_col1, time_col2 = st.columns([1, 3])
                    with time_col1:
                        st.caption(f"⏱️ {query_time:.2f}s")
                    with time_col2:
                        if use_agent and conv_history:
                            st.caption(f"💬 Using {len(conv_history)//2} previous turns for context")
                    
                    # Display sources
                    if sources:
                        with st.expander("📎 View Sources", expanded=False):
                            for i, source in enumerate(sources, 1):
                                col1, col2, col3 = st.columns([2, 1, 1])
                                
                                with col1:
                                    st.markdown(f"### 📄 Source {i}")
                                    st.caption(f"`{source['document_name']}`")
                                
                                with col2:
                                    st.metric("Relevance", f"{source['relevance_score']:.3f}")
                                
                                with col3:
                                    st.caption("Chunk ID")
                                    st.code(source['chunk_id'][-8:], language=None)
                                
                                # Display chunk text
                                if source.get('chunk_text'):
                                    with st.expander("📖 View chunk content", expanded=False):
                                        st.text_area(
                                            "Chunk text:",
                                            value=source['chunk_text'],
                                            height=200,
                                            label_visibility="collapsed",
                                            disabled=True,
                                            key=f"chunk_{source['chunk_id']}_{i}"
                                        )
                                
                                if i < len(sources):
                                    st.divider()
                    
                    # Save to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Track assistant response in agent conversation
                    if use_agent:
                        st.session_state.agent_conversation.append({
                            "role": "assistant",
                            "content": answer
                        })
                
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.session_state.messages:
        st.markdown("---")
        
        clear_col1, clear_col2, clear_col3 = st.columns([2, 1, 1])
        
        with clear_col1:
            if use_agent and len(st.session_state.agent_conversation) > 0:
                turns = len(st.session_state.agent_conversation) // 2
                st.caption(f"💬 Conversation: {turns} turn{'s' if turns != 1 else ''}")
        
        with clear_col2:
            if st.button("🔄 New Topic", use_container_width=True, help="Start fresh conversation on a new topic"):
                if use_agent:
                    st.session_state.agent_conversation = []
                st.success("✓ Conversation reset! Agent will treat next question as new topic.")
        
        with clear_col3:
            if st.button("🗑️ Clear All", use_container_width=True, help="Clear entire chat history"):
                st.session_state.messages = []
                st.session_state.agent_conversation = []
                st.rerun()

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
            help="Upload PDF, TXT, or DOCX files (max 50MB)"
        )
        
        if uploaded_file is not None:
            st.write(f"**📄 Filename:** {uploaded_file.name}")
            st.write(f"**📦 Size:** {uploaded_file.size / 1024:.2f} KB")
            st.write(f"**📋 Type:** {uploaded_file.type}")
            
            if st.button("📤 Upload and Process", type="primary", use_container_width=True):
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Upload file
                    status_text.text("⬆️ Uploading file...")
                    progress_bar.progress(10)
                    uploaded_file.seek(0)
                    
                    status_text.text("📄 Extracting text (this may take 30-90 seconds for large PDFs)...")
                    progress_bar.progress(20)
                    
                    # Estimate time based on file size
                    size_mb = uploaded_file.size / (1024 * 1024)
                    if size_mb > 20:
                        status_text.warning(f"⏳ Large file ({size_mb:.1f}MB) - processing may take 2-5 minutes...")
                    
                    # Make request
                    response = upload_document(uploaded_file)
                    
                    progress_bar.progress(50)
                    status_text.text("🧠 Generating embeddings...")
                    
                    # Wait for response
                    if response.status_code == 200:
                        progress_bar.progress(100)
                        status_text.text("✅ Complete!")
                        
                        result = response.json()
                        st.success("✅ Document uploaded successfully!")
                        
                        # Display results
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Document ID", result['document_id'][:8] + "...")
                        with col_b:
                            st.metric("Chunks Created", result['chunks_created'])
                        with col_c:
                            st.metric("Status", "✓ Ready")
                        
                        st.info("💡 You can now ask questions about this document in the Chat tab!")
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Upload failed: {error_detail}")
                
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📝 Supported Formats")
        st.markdown("""
        - **PDF** (.pdf)
        - **Text** (.txt)
        - **Word** (.docx)
        
        ### 💡 Tips
        - Files should be text-based
        - Max file size: 50MB
        - Clear, readable text works best
        - Multiple uploads are supported
        
        ### 🚀 After Upload
        - Documents are automatically chunked
        - Embeddings are generated
        - Ready for instant querying
        """)

# ============================================================================
# TAB 3: Library
# ============================================================================

with tab3:
    st.markdown("### 📖 Document Library")
    
    stats = get_stats()
    
    if stats:
        total_docs = stats.get('total_documents', 0)
        total_chunks = stats.get('total_chunks', 0)
        total_queries = stats.get('total_queries', 0)
        
        if total_docs > 0:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Total Documents", total_docs)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Total Chunks", total_chunks)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Queries Processed", total_queries)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.info("📚 All your uploaded documents are stored and ready for querying!")
            
            # Show average chunks per document
            if total_docs > 0:
                avg_chunks = total_chunks / total_docs
                st.caption(f"📊 Average chunks per document: {avg_chunks:.1f}")
        else:
            st.info("📭 No documents uploaded yet. Go to the Upload tab to add your first document!")
    else:
        st.warning("⚠️ Unable to connect to API. Please check if the backend is running.")

# ============================================================================
# TAB 4: Preview Document Extraction
# ============================================================================

with tab4:
    st.markdown("### 🔍 Preview Document Extraction & Chunking")
    st.info("📋 Upload a document to see the extracted text and how it will be chunked for RAG processing.")
    
    preview_file = st.file_uploader(
        "Choose a file to preview",
        type=['pdf', 'txt', 'docx'],
        key="preview_uploader",
        help="Upload a file to see how text is extracted and chunked"
    )
    
    if preview_file is not None:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**📄 File:** {preview_file.name}")
            st.write(f"**📦 Size:** {preview_file.size / 1024:.2f} KB")
        
        with col2:
            preview_button = st.button("🔍 Extract & Chunk", type="primary", use_container_width=True)
        
        if preview_button:
            with st.spinner("Extracting and chunking document..."):
                try:
                    # Reset file pointer
                    preview_file.seek(0)
                    
                    # Call preview endpoint
                    files = {"file": (preview_file.name, preview_file.getvalue(), preview_file.type)}
                    response = requests.post(f"{API_URL}/preview", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display extraction stats
                        st.markdown("---")
                        st.markdown("### 📊 Extraction Statistics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Characters", f"{result['extracted_length']:,}")
                        with col2:
                            st.metric("Words", f"{result['word_count']:,}")
                        with col3:
                            st.metric("Lines", f"{result['line_count']:,}")
                        with col4:
                            st.metric("File Type", result['file_type'].upper())
                        
                        # Display chunking stats
                        st.markdown("---")
                        st.markdown("### 🧩 Chunking Statistics")
                        
                        chunk_stats = result['chunk_stats']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Chunks", chunk_stats['total_chunks'])
                        with col2:
                            st.metric("Avg Size", f"{chunk_stats['avg_chunk_size']:.0f} chars")
                        with col3:
                            st.metric("Min Size", f"{chunk_stats['min_chunk_size']} chars")
                        with col4:
                            st.metric("Max Size", f"{chunk_stats['max_chunk_size']} chars")
                        
                        # Quality check
                        st.markdown("---")
                        st.markdown("### ✅ Quality Check")
                        
                        quality_issues = []
                        text = result['full_text']
                        
                        # Check for common OCR issues
                        single_letters = len([w for w in text.split() if len(w) == 1 and w.isupper() and w not in ['A', 'I']])
                        if single_letters > 50:
                            quality_issues.append(f"⚠️ {single_letters} stray single capital letters detected")
                        
                        consonant_clusters = len(re.findall(r'[bcdfghjklmnpqrstvwxyz]{6,}', text.lower()))
                        if consonant_clusters > 20:
                            quality_issues.append(f"⚠️ {consonant_clusters} unusual consonant clusters")
                        
                        excessive_spaces = len(re.findall(r'\s{4,}', text))
                        if excessive_spaces > 10:
                            quality_issues.append(f"⚠️ {excessive_spaces} instances of excessive spacing")
                        
                        # Check chunking quality
                        chunks = result['chunks']
                        very_small_chunks = sum(1 for c in chunks if c['length'] < 200)
                        very_large_chunks = sum(1 for c in chunks if c['length'] > 1500)
                        
                        if very_small_chunks > len(chunks) * 0.2:
                            quality_issues.append(f"⚠️ {very_small_chunks} very small chunks (< 200 chars)")
                        
                        if very_large_chunks > 0:
                            quality_issues.append(f"⚠️ {very_large_chunks} very large chunks (> 1500 chars)")
                        
                        if quality_issues:
                            for issue in quality_issues:
                                st.warning(issue)
                        else:
                            st.success("✅ Text extraction and chunking look good!")
                        
                        # Display chunks
                        st.markdown("---")
                        st.markdown("### 📦 Chunk Preview")
                        
                        # Create tabs for different views
                        chunk_tab1, chunk_tab2, chunk_tab3 = st.tabs([
                            f"📚 All Chunks ({len(chunks)})",
                            "📄 Full Text",
                            "📊 Chunk Analysis"
                        ])
                        
                        with chunk_tab1:
                            st.caption(f"Showing how the document will be split into {len(chunks)} chunks for RAG processing")
                            
                            # Add chunk size filter
                            show_all = st.checkbox("Show all chunks", value=False)
                            
                            chunks_to_show = chunks if show_all else chunks[:10]
                            
                            if not show_all and len(chunks) > 10:
                                st.info(f"Showing first 10 of {len(chunks)} chunks. Check 'Show all chunks' to see everything.")
                            
                            # Display each chunk
                            for i, chunk in enumerate(chunks_to_show, 1):
                                with st.expander(
                                    f"📦 Chunk {chunk['chunk_index'] + 1} - {chunk['length']} chars, {chunk['word_count']} words",
                                    expanded=(i <= 3)  # First 3 chunks expanded
                                ):
                                    # Chunk metadata
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.caption(f"**Index:** {chunk['chunk_index']}")
                                    with col2:
                                        st.caption(f"**Characters:** {chunk['length']}")
                                    with col3:
                                        st.caption(f"**Words:** {chunk['word_count']}")
                                    
                                    # Chunk content
                                    st.markdown("**Content:**")
                                    st.text_area(
                                        "Chunk text:",
                                        value=chunk['text'],
                                        height=200,
                                        label_visibility="collapsed",
                                        disabled=True,
                                        key=f"preview_chunk_{chunk['chunk_id']}"
                                    )
                                    
                                    # Show where chunk starts
                                    st.caption(f"**Starts with:** {chunk['first_line'][:100]}...")
                        
                        with chunk_tab2:
                            st.caption("Complete extracted text")
                            
                            # Add search functionality
                            search_term = st.text_input(
                                "🔍 Search in text:", 
                                key="search_preview_full",
                                placeholder="Enter text to search..."
                            )
                            
                            if search_term:
                                count = result['full_text'].lower().count(search_term.lower())
                                st.success(f"✨ Found **{count}** occurrence(s)")
                                
                                highlighted_text = re.sub(
                                    f'({re.escape(search_term)})',
                                    r'>>> \1 <<<',
                                    result['full_text'],
                                    flags=re.IGNORECASE
                                )
                                st.text_area(
                                    "Content:",
                                    value=highlighted_text,
                                    height=500,
                                    label_visibility="collapsed",
                                    disabled=True,
                                    key="preview_full_search"
                                )
                            else:
                                st.text_area(
                                    "Content:",
                                    value=result['full_text'],
                                    height=500,
                                    label_visibility="collapsed",
                                    disabled=True,
                                    key="preview_full"
                                )
                        
                        with chunk_tab3:
                            st.caption("Analysis of chunk distribution and quality")
                            
                            # Chunk size distribution
                            st.markdown("**Chunk Size Distribution:**")
                            
                            chunk_sizes = [c['length'] for c in chunks]
                            
                            # Create a simple histogram using metrics
                            size_ranges = [
                                ("0-200", sum(1 for s in chunk_sizes if s < 200)),
                                ("200-400", sum(1 for s in chunk_sizes if 200 <= s < 400)),
                                ("400-600", sum(1 for s in chunk_sizes if 400 <= s < 600)),
                                ("600-800", sum(1 for s in chunk_sizes if 600 <= s < 800)),
                                ("800-1000", sum(1 for s in chunk_sizes if 800 <= s < 1000)),
                                ("1000+", sum(1 for s in chunk_sizes if s >= 1000)),
                            ]
                            
                            cols = st.columns(6)
                            for col, (range_label, count) in zip(cols, size_ranges):
                                with col:
                                    st.metric(f"{range_label} chars", count)
                            
                            st.markdown("---")
                            
                            # Chunk details table
                            st.markdown("**Chunk Details:**")
                            
                            chunk_data = []
                            for chunk in chunks[:20]:  # First 20 chunks
                                chunk_data.append({
                                    "Index": chunk['chunk_index'] + 1,
                                    "Characters": chunk['length'],
                                    "Words": chunk['word_count'],
                                    "First Line": chunk['first_line'][:50] + "..." if len(chunk['first_line']) > 50 else chunk['first_line']
                                })
                            
                            df = pd.DataFrame(chunk_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            
                            if len(chunks) > 20:
                                st.caption(f"Showing first 20 of {len(chunks)} chunks")
                        
                        # Download options
                        st.markdown("---")
                        st.markdown("### 💾 Export Options")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                label="📄 Download Full Text",
                                data=result['full_text'],
                                file_name=f"{preview_file.name}_extracted.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Create chunked text file
                            chunked_text = ""
                            for i, chunk in enumerate(chunks, 1):
                                chunked_text += f"\n{'='*80}\n"
                                chunked_text += f"CHUNK {i}/{len(chunks)}\n"
                                chunked_text += f"Characters: {chunk['length']} | Words: {chunk['word_count']}\n"
                                chunked_text += f"{'='*80}\n\n"
                                chunked_text += chunk['text']
                                chunked_text += "\n\n"
                            
                            st.download_button(
                                label="📦 Download Chunks",
                                data=chunked_text,
                                file_name=f"{preview_file.name}_chunks.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col3:
                            # Create JSON export
                            import json
                            json_export = {
                                "filename": preview_file.name,
                                "extracted_at": datetime.now().isoformat(),
                                "statistics": {
                                    "characters": result['extracted_length'],
                                    "words": result['word_count'],
                                    "lines": result['line_count'],
                                    "chunks": len(chunks)
                                },
                                "chunk_stats": chunk_stats,
                                "chunks": chunks
                            }
                            
                            st.download_button(
                                label="🔧 Download JSON",
                                data=json.dumps(json_export, indent=2),
                                file_name=f"{preview_file.name}_analysis.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                    else:
                        error_detail = response.json().get('detail', 'Unknown error')
                        st.error(f"❌ Error: {error_detail}")
                
                except Exception as e:
                    st.error(f"❌ Error previewing file: {str(e)}")
                    import traceback
                    with st.expander("🐛 Debug Info"):
                        st.code(traceback.format_exc())

# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Made with ❤️ using Claude, FastAPI, and Streamlit<br>
        Study Buddy v1.0 | Powered by AI
    </div>
    """,
    unsafe_allow_html=True
)