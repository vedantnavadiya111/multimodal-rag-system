"""
Embedding Manager for MultiModal RAG System.
Handles text and image embeddings using sentence-transformers.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

from .utils import setup_logging

class EmbeddingManager:
    """Manage embeddings for multi-modal content."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging()
        self.text_model = None
        self.device = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding models."""
        try:
            # Set device
            self.device = self.config['models']['text_embedding']['device']
            if self.device == "cuda" and not torch.cuda.is_available():
                self.device = "cpu"
                self.logger.warning("CUDA not available, using CPU")
            
            # Load text embedding model
            model_name = self.config['models']['text_embedding']['name']
            self.logger.info(f"Loading embedding model: {model_name}")
            
            self.text_model = SentenceTransformer(model_name, device=self.device)
            
            self.logger.info(f"Embedding model loaded on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error initializing embedding models: {e}")
            raise
    
    def encode_text(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """Encode text into embeddings."""
        try:
            if not texts:
                return []
            
            # Clean and prepare texts
            cleaned_texts = [text.strip() for text in texts if text.strip()]
            
            if not cleaned_texts:
                return []
            
            # Generate embeddings
            embeddings = self.text_model.encode(
                cleaned_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            # Convert to list format
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            elif torch.is_tensor(embeddings):
                embeddings = embeddings.cpu().numpy().tolist()
            
            self.logger.info(f"Generated embeddings for {len(cleaned_texts)} texts")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error encoding text: {e}")
            return []
    
    def encode_single_text(self, text: str) -> List[float]:
        """Encode a single text into embedding."""
        try:
            if not text.strip():
                return []
            
            embedding = self.text_model.encode(
                [text.strip()],
                convert_to_tensor=False,
                normalize_embeddings=True
            )
            
            if isinstance(embedding, np.ndarray):
                return embedding[0].tolist()
            elif torch.is_tensor(embedding):
                return embedding[0].cpu().numpy().tolist()
            else:
                return embedding[0]
                
        except Exception as e:
            self.logger.error(f"Error encoding single text: {e}")
            return []
    
    def encode_multimodal_content(self, content_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encode multi-modal content items (text, images, video descriptions)."""
        try:
            encoded_items = []
            
            # Group by content type for efficient batch processing
            text_items = []
            text_indices = []
            
            for i, item in enumerate(content_items):
                if item['type'] in ['text', 'image', 'video_frame']:
                    # For images and video frames, we embed their captions/descriptions
                    if item['type'] == 'text':
                        text_content = item['content']
                    elif item['type'] == 'image':
                        text_content = item.get('caption', '')
                    elif item['type'] == 'video_frame':
                        text_content = item.get('caption', '')
                    else:
                        text_content = str(item.get('content', ''))
                    
                    if text_content.strip():
                        text_items.append(text_content)
                        text_indices.append(i)
            
            # Batch encode all text content
            if text_items:
                self.logger.info(f"Encoding {len(text_items)} content items")
                embeddings = self.encode_text(text_items, show_progress=True)
                
                # Map embeddings back to original items
                for embedding, original_index in zip(embeddings, text_indices):
                    item = content_items[original_index].copy()
                    item['embedding'] = embedding
                    item['embedding_model'] = self.config['models']['text_embedding']['name']
                    encoded_items.append(item)
            
            self.logger.info(f"Successfully encoded {len(encoded_items)} content items")
            return encoded_items
            
        except Exception as e:
            self.logger.error(f"Error encoding multimodal content: {e}")
            return []
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model."""
        try:
            return self.text_model.get_sentence_embedding_dimension()
        except Exception as e:
            self.logger.error(f"Error getting embedding dimension: {e}")
            return 384  # Default dimension for all-MiniLM-L6-v2
    
    def batch_encode_documents(self, documents: List[Dict[str, Any]], 
                              progress_callback=None) -> List[Dict[str, Any]]:
        """Encode multiple documents with progress tracking."""
        try:
            all_encoded_items = []
            total_docs = len(documents)
            
            for i, doc in enumerate(documents):
                try:
                    # Prepare content items for encoding
                    content_items = []
                    
                    # Add text chunks
                    if 'text_chunks' in doc:
                        for chunk in doc['text_chunks']:
                            content_items.append({
                                'type': 'text',
                                'content': chunk['content'],
                                'source': doc['source'],
                                'page': chunk.get('page', 1),
                                'chunk_id': chunk.get('chunk_id', 0)
                            })
                    
                    # Add image captions
                    if 'images' in doc:
                        for img in doc['images']:
                            content_items.append({
                                'type': 'image',
                                'content': img['caption'],
                                'caption': img['caption'],
                                'source': doc['source'],
                                'page': img.get('page', 1)
                            })
                    
                    # Add video frame descriptions
                    if 'frames' in doc:
                        for frame in doc['frames']:
                            content_items.append({
                                'type': 'video_frame',
                                'content': frame['caption'],
                                'caption': frame['caption'],
                                'source': doc['source'],
                                'timestamp': frame.get('timestamp', 0),
                                'frame_number': frame.get('frame_number', 0)
                            })
                    
                    # Add video summary
                    if 'summary' in doc and doc['summary']:
                        content_items.append({
                            'type': 'video_summary',
                            'content': doc['summary'],
                            'source': doc['source']
                        })
                    
                    # Encode content items
                    encoded_items = self.encode_multimodal_content(content_items)
                    all_encoded_items.extend(encoded_items)
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(i + 1, total_docs, f"Encoded {len(encoded_items)} items")
                
                except Exception as e:
                    self.logger.error(f"Error encoding document {i}: {e}")
                    continue
            
            self.logger.info(f"Successfully encoded {len(all_encoded_items)} items from {total_docs} documents")
            return all_encoded_items
            
        except Exception as e:
            self.logger.error(f"Error in batch encoding: {e}")
            return []