"""
Streamlit Web Interface for Multi-Modal RAG System.
Professional interface for document upload, processing, and querying.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import tempfile
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, format_file_size, ProgressTracker
from src.rag_engine import RAGEngine

# Page configuration
st.set_page_config(
    page_title="Academic Multi-Modal RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-card {
        background: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .source-tag {
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.2rem 0.5rem;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with caching."""
    try:
        config = load_config()
        rag_engine = RAGEngine(config)
        return rag_engine, None
    except Exception as e:
        return None, str(e)

def display_header():
    """Display the main header."""
    st.markdown('<div class="main-header">🔍 Academic Multi-Modal RAG System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Process PDFs, Images, and Videos • Ask Questions • Get Intelligent Answers</div>', unsafe_allow_html=True)

def display_system_stats(rag_engine):
    """Display system statistics in the sidebar."""
    with st.sidebar:
        st.header("📊 System Status")
        
        try:
            stats = rag_engine.get_system_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats.get('total_documents', 0))
            with col2:
                status = stats.get('system_status', 'unknown')
                st.metric("Status", "🟢 Online" if status == 'operational' else "🔴 Error")
            
            with st.expander("📋 Model Information"):
                st.write(f"**Text Embedding:** {stats.get('embedding_model', 'N/A')}")
                st.write(f"**Vision Model:** {stats.get('vision_model', 'N/A')}")
                st.write(f"**Vector Store:** ChromaDB")
            
            with st.expander("📁 Supported Formats"):
                formats = stats.get('supported_formats', [])
                for fmt in formats:
                    st.write(f"• {fmt.upper()}")
                    
        except Exception as e:
            st.error(f"Error loading system stats: {e}")

def handle_file_upload(rag_engine):
    """Handle file upload and processing."""
    st.header("📤 Document Upload & Processing")
    
    uploaded_files = st.file_uploader(
        "Upload your documents (PDF, Images, Videos)",
        type=['pdf', 'png', 'jpg', 'jpeg', 'mp4', 'avi'],
        accept_multiple_files=True,
        help="Supported formats: PDF (with text and images), PNG, JPG, MP4, AVI"
    )
    
    if uploaded_files:
        st.subheader(f"📋 Uploaded Files ({len(uploaded_files)})")
        
        # Display file information
        for file in uploaded_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📄 {file.name}")
            with col2:
                st.write(format_file_size(file.size))
            with col3:
                st.write(f".{file.name.split('.')[-1].upper()}")
        
        # Process files button
        if st.button("🚀 Process Documents", type="primary"):
            return process_uploaded_files(rag_engine, uploaded_files)
    
    return None

def process_uploaded_files(rag_engine, uploaded_files):
    """Process uploaded files."""
    try:
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Save uploaded files temporarily
        temp_files = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_files.append(tmp_file.name)
        
        # Process files
        results = []
        for i, temp_file in enumerate(temp_files):
            progress = (i + 1) / len(temp_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_files[i].name}...")
            
            result = rag_engine.ingest_document(temp_file)
            results.append({
                'filename': uploaded_files[i].name,
                'result': result
            })
            
            time.sleep(0.5)  # Small delay for UX
        
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        # Display results
        progress_bar.progress(1.0)
        status_text.success("✅ Processing completed!")
        
        successful = sum(1 for r in results if r['result']['status'] == 'success')
        st.success(f"🎉 Successfully processed {successful}/{len(results)} documents!")
        
        # Show detailed results
        with st.expander("📋 Processing Details"):
            for r in results:
                result = r['result']
                if result['status'] == 'success':
                    st.success(f"✅ {r['filename']}: {result['items_processed']} items processed")
                else:
                    st.error(f"❌ {r['filename']}: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        st.error(f"Error processing files: {e}")
        return False

def handle_querying(rag_engine):
    """Handle document querying interface."""
    st.header("💬 Ask Questions")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        placeholder="e.g., What are the main findings in the research paper? What does the image show?",
        height=100
    )
    
    # Query options
    col1, col2 = st.columns(2)
    with col1:
        max_results = st.slider("Max Results", 1, 10, 5)
    with col2:
        include_sources = st.checkbox("Include Sources", value=True)
    
    # Advanced filters
    with st.expander("🔧 Advanced Filters"):
        content_type = st.selectbox(
            "Content Type",
            ["All", "Text", "Images", "Videos"],
            help="Filter results by content type"
        )
        
        source_filter = st.text_input(
            "Source Filter",
            placeholder="Filter by filename (optional)",
            help="Enter part of filename to filter results"
        )
    
    if st.button("🔍 Search", type="primary") and query.strip():
        return execute_query(rag_engine, query, max_results, include_sources, content_type, source_filter)
    
    return None

def execute_query(rag_engine, query, max_results, include_sources, content_type, source_filter):
    """Execute the query and display results."""
    try:
        with st.spinner("🔍 Searching through your documents..."):
            # Prepare filters
            filters = {}
            if content_type != "All":
                type_mapping = {
                    "Text": "text",
                    "Images": "image", 
                    "Videos": "video_frame"
                }
                filters['type'] = type_mapping[content_type]
            
            if source_filter.strip():
                # Note: This would need enhancement for partial matching in ChromaDB
                pass
            
            # Execute query
            results = rag_engine.generate_response(
                query_text=query,
                max_context_results=max_results,
                filters=filters if filters else None
            )
        
        # Display results
        st.subheader("🎯 Answer")
        
        # Confidence indicator
        confidence = results.get('confidence', 0.0)
        confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
        st.write(f"**Confidence:** {confidence_color} {confidence:.1%}")
        
        # Main answer
        st.markdown(f"""
        <div class="result-card">
            {results.get('answer', 'No answer generated.')}
        </div>
        """, unsafe_allow_html=True)
        
        # Sources
        if include_sources and results.get('sources'):
            st.subheader("📚 Sources")
            
            for i, source in enumerate(results['sources'], 1):
                with st.expander(f"Source {i}: {Path(source['source']).name}"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Type:** {source['type'].title()}")
                        if 'page' in source:
                            st.write(f"**Page:** {source['page']}")
                        if 'timestamp' in source:
                            st.write(f"**Timestamp:** {source['timestamp']:.1f}s")
                    
                    with col2:
                        similarity = source.get('similarity_score', 0.0)
                        st.metric("Relevance", f"{similarity:.1%}")
        
        # Query metadata
        st.caption(f"Query processed at {results.get('timestamp', 'Unknown time')}")
        
        return True
        
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return False

def main():
    """Main application function."""
    # Initialize system
    rag_engine, error = initialize_rag_system()
    
    if error:
        st.error(f"🚨 System initialization failed: {error}")
        st.info("Please check your configuration and try again.")
        return
    
    # Display header
    display_header()
    
    # Display system stats in sidebar
    display_system_stats(rag_engine)
    
    # Add system management in sidebar
    with st.sidebar:
        st.header("🛠️ System Management")
        
        if st.button("🗑️ Clear All Documents", help="Remove all documents from the system"):
            if st.checkbox("I confirm I want to clear all documents"):
                with st.spinner("Clearing documents..."):
                    success = rag_engine.clear_all_documents()
                    if success:
                        st.success("✅ All documents cleared!")
                        st.experimental_rerun()
                    else:
                        st.error("❌ Failed to clear documents")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "💬 Query Documents", "📊 System Info"])
    
    with tab1:
        handle_file_upload(rag_engine)
    
    with tab2:
        handle_querying(rag_engine)
    
    with tab3:
        st.header("📋 System Information")
        
        try:
            stats = rag_engine.get_system_stats()
            
            # System overview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Documents", stats.get('total_documents', 0))
            with col2:
                st.metric("System Status", "Operational" if stats.get('system_status') == 'operational' else "Error")
            with col3:
                st.metric("Vector Store", "ChromaDB")
            
            # Technical details
            st.subheader("🔧 Technical Configuration")
            
            config_data = {
                "Embedding Model": stats.get('embedding_model', 'N/A'),
                "Vision Model": stats.get('vision_model', 'N/A'),
                "Supported Formats": ', '.join(stats.get('supported_formats', [])),
                "Vector Store Type": "ChromaDB",
                "Collection Name": stats.get('vector_store', {}).get('collection', 'N/A')
            }
            
            for key, value in config_data.items():
                st.write(f"**{key}:** {value}")
                
        except Exception as e:
            st.error(f"Error loading system information: {e}")

if __name__ == "__main__":
    main()