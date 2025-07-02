"""
MultiModal RAG System

A production-ready Multi-Modal Retrieval-Augmented Generation system
for processing academic documents with PDF, image, and video capabilities.
"""

__version__ = "1.0.0"
__author__ = "Vedant Navadiya"
__email__ = "vedantnavadiya111@gmail.com"

from .document_processor import MultiModalProcessor
from .vector_store import ChromaVectorStore
from .rag_engine import RAGEngine
from .embedding_manager import EmbeddingManager

__all__ = [
    "MultiModalProcessor",
    "ChromaVectorStore", 
    "RAGEngine",
    "EmbeddingManager"
]