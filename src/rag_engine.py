"""
RAG Engine - Core orchestration for Multi-Modal RAG System.
Coordinates document processing, embedding, storage, and retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import uuid
from pathlib import Path
import json

from .document_processor import MultiModalProcessor
from .embedding_manager import EmbeddingManager
from .vector_store import ChromaVectorStore
from .utils import setup_logging, ProgressTracker

class RAGEngine:
    """Core RAG engine that orchestrates all components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging()
        
        # Initialize components
        self.document_processor = MultiModalProcessor(config)
        self.embedding_manager = EmbeddingManager(config)
        self.vector_store = ChromaVectorStore(config)
        
        self.logger.info("RAG Engine initialized successfully")
    
    def ingest_document(self, file_path: str, file_type: str = None) -> Dict[str, Any]:
        """Ingest a single document into the RAG system."""
        try:
            self.logger.info(f"Ingesting document: {file_path}")
            
            # Step 1: Process document
            processed_doc = self.document_processor.process_document(file_path, file_type)
            
            # Step 2: Generate embeddings
            encoded_items = self.embedding_manager.batch_encode_documents([processed_doc])
            
            if not encoded_items:
                raise ValueError("No content could be encoded from the document")
            
            # Step 3: Store in vector database
            documents = []
            embeddings = []
            metadata = []
            ids = []
            
            for item in encoded_items:
                # Prepare document text
                doc_text = item['content']
                documents.append(doc_text)
                
                # Prepare embedding
                embeddings.append(item['embedding'])
                
                # Prepare metadata
                meta = {
                    'source': item['source'],
                    'type': item['type'],
                    'embedding_model': item['embedding_model'],
                    'ingestion_id': str(uuid.uuid4())
                }
                
                # Add type-specific metadata
                if item['type'] == 'text':
                    meta.update({
                        'page': item.get('page', 1),
                        'chunk_id': item.get('chunk_id', 0)
                    })
                elif item['type'] == 'image':
                    meta.update({
                        'page': item.get('page', 1),
                        'caption': item.get('caption', '')
                    })
                elif item['type'] == 'video_frame':
                    meta.update({
                        'timestamp': item.get('timestamp', 0),
                        'frame_number': item.get('frame_number', 0)
                    })
                
                metadata.append(meta)
                ids.append(str(uuid.uuid4()))
            
            # Store in vector database
            success = self.vector_store.add_documents(
                documents=documents,
                embeddings=embeddings,
                metadata=metadata,
                ids=ids
            )
            
            if not success:
                raise ValueError("Failed to store documents in vector database")
            
            result = {
                'status': 'success',
                'file_path': file_path,
                'items_processed': len(encoded_items),
                'document_type': processed_doc['type'],
                'metadata': processed_doc.get('metadata', {})
            }
            
            self.logger.info(f"Successfully ingested document: {file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error ingesting document {file_path}: {e}")
            return {
                'status': 'error',
                'file_path': file_path,
                'error': str(e)
            }
    
    def ingest_multiple_documents(self, file_paths: List[str], 
                                 progress_callback=None) -> List[Dict[str, Any]]:
        """Ingest multiple documents with progress tracking."""
        try:
            results = []
            total_files = len(file_paths)
            
            for i, file_path in enumerate(file_paths):
                try:
                    result = self.ingest_document(file_path)
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(i + 1, total_files, f"Processed {Path(file_path).name}")
                
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
                    results.append({
                        'status': 'error',
                        'file_path': file_path,
                        'error': str(e)
                    })
            
            successful = sum(1 for r in results if r['status'] == 'success')
            self.logger.info(f"Batch ingestion completed: {successful}/{total_files} successful")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in batch ingestion: {e}")
            return []
    
    def query(self, query_text: str, 
              max_results: int = 5, 
              filters: Optional[Dict[str, Any]] = None,
              include_similarity_scores: bool = True) -> Dict[str, Any]:
        """Query the RAG system and get relevant documents."""
        try:
            self.logger.info(f"Processing query: {query_text[:100]}...")
            
            # Step 1: Generate query embedding
            query_embedding = self.embedding_manager.encode_single_text(query_text)
            
            if not query_embedding:
                raise ValueError("Failed to generate query embedding")
            
            # Step 2: Perform similarity search
            search_results = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                n_results=max_results,
                filters=filters
            )
            
            # Step 3: Format results
            formatted_results = []
            
            for i, (doc, metadata, distance) in enumerate(zip(
                search_results['documents'],
                search_results['metadata'],
                search_results['distances']
            )):
                result_item = {
                    'rank': i + 1,
                    'content': doc,
                    'source': metadata.get('source', 'Unknown'),
                    'type': metadata.get('type', 'text'),
                    'metadata': metadata
                }
                
                if include_similarity_scores:
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    similarity_score = 1 / (1 + distance)
                    result_item['similarity_score'] = similarity_score
                    result_item['distance'] = distance
                
                formatted_results.append(result_item)
            
            # Step 4: Generate response summary
            response = {
                'query': query_text,
                'results': formatted_results,
                'total_results': len(formatted_results),
                'max_results': max_results,
                'filters_applied': filters or {},
                'timestamp': self._get_timestamp()
            }
            
            self.logger.info(f"Query completed: {len(formatted_results)} results returned")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                'query': query_text,
                'results': [],
                'total_results': 0,
                'error': str(e),
                'timestamp': self._get_timestamp()
            }
    
    def generate_response(self, query_text: str, 
                         max_context_results: int = 3,
                         filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a comprehensive response using retrieved context."""
        try:
            # Step 1: Retrieve relevant context
            search_results = self.query(
                query_text=query_text,
                max_results=max_context_results,
                filters=filters
            )
            
            if not search_results['results']:
                return {
                    'query': query_text,
                    'answer': "I couldn't find relevant information to answer your question. Please try rephrasing your query or check if documents have been uploaded to the system.",
                    'sources': [],
                    'confidence': 0.0,
                    'timestamp': self._get_timestamp()
                }
            
            # Step 2: Build context from retrieved documents
            context_pieces = []
            sources = []
            
            for result in search_results['results']:
                context_pieces.append(result['content'])
                
                # Track unique sources
                source_info = {
                    'source': result['source'],
                    'type': result['type'],
                    'similarity_score': result.get('similarity_score', 0.0)
                }
                
                if result['type'] == 'text':
                    source_info['page'] = result['metadata'].get('page', 1)
                elif result['type'] == 'video_frame':
                    source_info['timestamp'] = result['metadata'].get('timestamp', 0)
                
                sources.append(source_info)
            
            # Step 3: Generate response using context
            context = "\n\n".join(context_pieces)
            
            # Simple response generation (can be enhanced with LLM)
            response_text = self._generate_contextual_response(query_text, context, sources)
            
            # Step 4: Calculate confidence score
            confidence = self._calculate_confidence_score(search_results['results'])
            
            return {
                'query': query_text,
                'answer': response_text,
                'sources': sources,
                'confidence': confidence,
                'context_used': len(context_pieces),
                'timestamp': self._get_timestamp()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                'query': query_text,
                'answer': f"An error occurred while processing your query: {str(e)}",
                'sources': [],
                'confidence': 0.0,
                'timestamp': self._get_timestamp()
            }
    
    def _generate_contextual_response(self, query: str, context: str, sources: List[Dict]) -> str:
        """Generate a contextual response (simplified version)."""
        # This is a simplified response generation
        # In a production system, you'd use a language model here
        
        response_parts = [
            f"Based on the available documents, here's what I found regarding your query:\n"
        ]
        
        # Add context summary
        if len(context) > 500:
            response_parts.append(f"**Summary of relevant information:**\n{context[:500]}...\n")
        else:
            response_parts.append(f"**Relevant information:**\n{context}\n")
        
        # Add source information
        if sources:
            response_parts.append(f"\n**Sources referenced:**")
            for i, source in enumerate(sources[:3], 1):
                source_text = f"{i}. {Path(source['source']).name}"
                if source['type'] == 'text' and 'page' in source:
                    source_text += f" (Page {source['page']})"
                elif source['type'] == 'video_frame' and 'timestamp' in source:
                    source_text += f" (Timestamp: {source['timestamp']:.1f}s)"
                response_parts.append(source_text)
        
        return "\n".join(response_parts)
    
    def _calculate_confidence_score(self, results: List[Dict]) -> float:
        """Calculate confidence score based on search results."""
        if not results:
            return 0.0
        
        # Simple confidence calculation based on similarity scores
        scores = [r.get('similarity_score', 0.0) for r in results]
        avg_score = sum(scores) / len(scores)
        
        # Normalize to 0-1 range
        return min(avg_score, 1.0)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            
            return {
                'total_documents': vector_stats.get('total_documents', 0),
                'embedding_model': self.config['models']['text_embedding']['name'],
                'vision_model': self.config['models']['vision_model']['name'],
                'vector_store': {
                    'type': 'ChromaDB',
                    'collection': vector_stats.get('collection_name', ''),
                    'persist_directory': vector_stats.get('persist_directory', '')
                },
                'supported_formats': self.config['document_processing']['supported_formats'],
                'system_status': 'operational'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system stats: {e}")
            return {'system_status': 'error', 'error': str(e)}
    
    def clear_all_documents(self) -> bool:
        """Clear all documents from the system."""
        try:
            success = self.vector_store.clear_collection()
            if success:
                self.logger.info("All documents cleared from the system")
            return success
        except Exception as e:
            self.logger.error(f"Error clearing documents: {e}")
            return False