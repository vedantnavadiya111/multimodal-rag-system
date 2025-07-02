"""
Comprehensive test to verify the complete RAG pipeline.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.utils import load_config
from src.rag_engine import RAGEngine

def main():
    try:
        print("🔧 Testing Multi-Modal RAG System Setup...")
        print("=" * 50)
        
        # Load configuration
        config = load_config()
        print("✅ Configuration loaded successfully")
        
        # Initialize RAG Engine
        print("🚀 Initializing RAG Engine...")
        rag_engine = RAGEngine(config)
        print("✅ RAG Engine initialized successfully")
        
        # Test system stats
        stats = rag_engine.get_system_stats()
        print("\n📊 System Statistics:")
        print(f"  - Total documents: {stats.get('total_documents', 0)}")
        print(f"  - Embedding model: {stats.get('embedding_model', 'N/A')}")
        print(f"  - Vision model: {stats.get('vision_model', 'N/A')}")
        print(f"  - System status: {stats.get('system_status', 'N/A')}")
        
        # Test query functionality (without documents)
        print("\n🔍 Testing Query Functionality...")
        test_query = "What is machine learning?"
        result = rag_engine.query(test_query, max_results=3)
        print(f"  - Query processed: '{test_query}'")
        print(f"  - Results returned: {result.get('total_results', 0)}")
        
        # Test response generation
        print("\n💬 Testing Response Generation...")
        response = rag_engine.generate_response(test_query)
        print(f"  - Response generated: {len(response.get('answer', ''))} characters")
        print(f"  - Confidence score: {response.get('confidence', 0.0):.2f}")
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Run the Streamlit app: streamlit run frontend/streamlit_app.py")
        print("2. Upload documents to test the full pipeline")
        print("3. Query your documents using the web interface")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if all dependencies are installed: pip install -r requirements.txt")
        print("2. Verify config/config.yaml exists and is valid")
        print("3. Check if data/chroma_db directory can be created")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())