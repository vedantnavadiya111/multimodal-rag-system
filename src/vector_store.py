"""
Vector Store implementation using ChromaDB for the MultiModal RAG System.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
import uuid
import json

from .utils import setup_logging, ensure_directory_exists

class ChromaVectorStore:
    """ChromaDB vector store for multi-modal embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging()
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure persist directory exists
            persist_dir = self.config['vector_db']['persist_directory']
            ensure_directory_exists(persist_dir)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=persist_dir)
            
            # Get or create collection
            collection_name = self.config['vector_db']['collection_name']
            
            try:
                self.collection = self.client.get_collection(collection_name)
                self.logger.info(f"Loaded existing collection: {collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "MultiModal RAG documents"}
                )
                self.logger.info(f"Created new collection: {collection_name}")
                
        except Exception as e:
            self.logger.error(f"Error initializing vector store: {e}")
            raise
    
    def add_documents(self, 
                     documents: List[str], 
                     embeddings: List[List[float]], 
                     metadata: List[Dict[str, Any]],
                     ids: Optional[List[str]] = None) -> bool:
        """Add documents with embeddings to the vector store."""
        try:
            if not ids:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            # Ensure all lists have the same length
            if not (len(documents) == len(embeddings) == len(metadata) == len(ids)):
                raise ValueError("All input lists must have the same length")
            
            # Convert metadata to strings for ChromaDB compatibility
            processed_metadata = []
            for meta in metadata:
                processed_meta = {}
                for key, value in meta.items():
                    if isinstance(value, (dict, list)):
                        processed_meta[key] = json.dumps(value)
                    else:
                        processed_meta[key] = str(value)
                processed_metadata.append(processed_meta)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=processed_metadata,
                ids=ids
            )
            
            self.logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    def similarity_search(self, 
                         query_embedding: List[float], 
                         n_results: int = 5,
                         filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform similarity search in the vector store."""
        try:
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    where_clause[key] = {"$eq": str(value)}
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            processed_results = {
                'documents': results['documents'][0] if results['documents'] else [],
                'metadata': [],
                'distances': results['distances'][0] if results['distances'] else [],
                'total_results': len(results['documents'][0]) if results['documents'] else 0
            }
            
            # Process metadata back from strings
            if results['metadatas'] and results['metadatas'][0]:
                for meta in results['metadatas'][0]:
                    processed_meta = {}
                    for key, value in meta.items():
                        try:
                            # Try to parse JSON strings back to objects
                            if value.startswith(('{', '[')):
                                processed_meta[key] = json.loads(value)
                            else:
                                processed_meta[key] = value
                        except:
                            processed_meta[key] = value
                    processed_results['metadata'].append(processed_meta)
            
            self.logger.info(f"Retrieved {processed_results['total_results']} similar documents")
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Error performing similarity search: {e}")
            return {'documents': [], 'metadata': [], 'distances': [], 'total_results': 0}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            return {
                'total_documents': count,
                'collection_name': self.config['vector_db']['collection_name'],
                'persist_directory': self.config['vector_db']['persist_directory']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            return {'total_documents': 0}
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by IDs."""
        try:
            self.collection.delete(ids=ids)
            self.logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting documents: {e}")
            return False
    
    def update_documents(self, 
                        ids: List[str],
                        documents: Optional[List[str]] = None,
                        embeddings: Optional[List[List[float]]] = None,
                        metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Update existing documents."""
        try:
            update_params = {'ids': ids}
            
            if documents:
                update_params['documents'] = documents
            if embeddings:
                update_params['embeddings'] = embeddings
            if metadata:
                # Process metadata
                processed_metadata = []
                for meta in metadata:
                    processed_meta = {}
                    for key, value in meta.items():
                        if isinstance(value, (dict, list)):
                            processed_meta[key] = json.dumps(value)
                        else:
                            processed_meta[key] = str(value)
                    processed_metadata.append(processed_meta)
                update_params['metadatas'] = processed_metadata
            
            self.collection.update(**update_params)
            self.logger.info(f"Updated {len(ids)} documents")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating documents: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Delete the collection and recreate it
            collection_name = self.config['vector_db']['collection_name']
            self.client.delete_collection(collection_name)
            
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "MultiModal RAG documents"}
            )
            
            self.logger.info("Cleared all documents from collection")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing collection: {e}")
            return False