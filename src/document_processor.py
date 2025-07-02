"""
Multi-Modal Document Processor for RAG System.
Handles PDF text extraction, image processing, and video analysis.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import tempfile

# Document Processing
import PyPDF2
from pdf2image import convert_from_path
import cv2
from PIL import Image
import numpy as np

# AI Models
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

# Utilities
from .utils import setup_logging, validate_file_size, get_file_extension

class MultiModalProcessor:
    """Process documents across multiple modalities: text, images, and videos."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logging()
        self.vision_model = None
        self.vision_processor = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize computer vision models."""
        try:
            model_name = self.config['models']['vision_model']['name']
            device = self.config['models']['vision_model']['device']
            
            self.logger.info(f"Loading vision model: {model_name}")
            self.vision_processor = BlipProcessor.from_pretrained(model_name)
            self.vision_model = BlipForConditionalGeneration.from_pretrained(model_name)
            
            if device == "cuda" and torch.cuda.is_available():
                self.vision_model = self.vision_model.to("cuda")
                self.logger.info("Vision model loaded on GPU")
            else:
                self.logger.info("Vision model loaded on CPU")
                
        except Exception as e:
            self.logger.error(f"Error initializing vision models: {e}")
            raise
    
    def process_document(self, file_path: str, file_type: str = None) -> Dict[str, Any]:
        """Process a document and extract multi-modal content."""
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not validate_file_size(file_path, self.config['document_processing']['max_file_size_mb']):
                raise ValueError("File size exceeds maximum limit")
            
            # Determine file type
            if not file_type:
                file_type = get_file_extension(file_path)
            
            self.logger.info(f"Processing {file_type} file: {file_path}")
            
            # Process based on file type
            if file_type == 'pdf':
                return self._process_pdf(file_path)
            elif file_type in ['png', 'jpg', 'jpeg']:
                return self._process_image(file_path)
            elif file_type in ['mp4', 'avi']:
                return self._process_video(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            raise
    
    def _process_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract text and images from PDF."""
        result = {
            'type': 'pdf',
            'source': pdf_path,
            'text_chunks': [],
            'images': [],
            'metadata': {}
        }
        
        try:
            # Extract text
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text_content += page_text + "\n"
                    
                    # Store page-wise text chunks
                    if page_text.strip():
                        result['text_chunks'].append({
                            'content': page_text.strip(),
                            'page': page_num + 1,
                            'type': 'text'
                        })
            
            # Extract images from PDF
            try:
                images = convert_from_path(pdf_path)
                for i, image in enumerate(images):
                    # Save temporary image
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                        image.save(tmp_file.name)
                        
                        # Generate caption
                        caption = self._generate_image_caption(tmp_file.name)
                        
                        result['images'].append({
                            'page': i + 1,
                            'caption': caption,
                            'path': tmp_file.name,
                            'type': 'image'
                        })
                        
                        # Clean up temp file
                        os.unlink(tmp_file.name)
                        
            except Exception as e:
                self.logger.warning(f"Could not extract images from PDF: {e}")
            
            result['metadata'] = {
                'total_pages': len(pdf_reader.pages),
                'total_text_chunks': len(result['text_chunks']),
                'total_images': len(result['images'])
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing PDF: {e}")
            raise
    
    def _process_image(self, image_path: str) -> Dict[str, Any]:
        """Process standalone image file."""
        try:
            caption = self._generate_image_caption(image_path)
            
            return {
                'type': 'image',
                'source': image_path,
                'caption': caption,
                'metadata': {
                    'file_size': os.path.getsize(image_path)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            raise
    
    def _process_video(self, video_path: str) -> Dict[str, Any]:
        """Extract and analyze frames from video."""
        result = {
            'type': 'video',
            'source': video_path,
            'frames': [],
            'summary': "",
            'metadata': {}
        }
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            # Extract frames at intervals
            frame_interval = max(1, int(fps * 5))  # Every 5 seconds
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % frame_interval == 0:
                    # Save frame as temporary image
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                        cv2.imwrite(tmp_file.name, frame)
                        
                        # Generate caption for frame
                        caption = self._generate_image_caption(tmp_file.name)
                        timestamp = frame_number / fps
                        
                        result['frames'].append({
                            'timestamp': timestamp,
                            'caption': caption,
                            'frame_number': frame_number,
                            'type': 'video_frame'
                        })
                        
                        # Clean up temp file
                        os.unlink(tmp_file.name)
                
                frame_number += 1
            
            cap.release()
            
            # Generate video summary
            if result['frames']:
                captions = [frame['caption'] for frame in result['frames']]
                result['summary'] = f"Video contains {len(captions)} analyzed frames. " + \
                                  " ".join(captions[:3])  # First 3 frame descriptions
            
            result['metadata'] = {
                'duration_seconds': duration,
                'fps': fps,
                'total_frames': frame_count,
                'analyzed_frames': len(result['frames'])
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing video: {e}")
            raise
    
    def _generate_image_caption(self, image_path: str) -> str:
        """Generate caption for an image using BLIP model."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Generate caption
            inputs = self.vision_processor(image, return_tensors="pt")
            
            if self.vision_model.device.type == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                output = self.vision_model.generate(**inputs, max_length=50)
                caption = self.vision_processor.decode(output[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            self.logger.error(f"Error generating image caption: {e}")
            return "Unable to generate caption for this image."
    
    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[Dict[str, Any]]:
        """Split text into chunks for embedding."""
        chunk_size = chunk_size or self.config['document_processing']['chunk_size']
        chunk_overlap = chunk_overlap or self.config['document_processing']['chunk_overlap']
        
        if len(text) <= chunk_size:
            return [{'content': text, 'chunk_id': 0}]
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for punct in ['. ', '! ', '? ', '\n\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct != -1:
                        end = last_punct + len(punct)
                        break
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    'content': chunk_text,
                    'chunk_id': chunk_id,
                    'start_pos': start,
                    'end_pos': end
                })
                chunk_id += 1
            
            start = end - chunk_overlap
        
        return chunks
    
    def batch_process_documents(self, file_paths: List[str], progress_callback=None) -> List[Dict[str, Any]]:
        """Process multiple documents in batch."""
        results = []
        total = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            try:
                result = self.process_document(file_path)
                results.append(result)
                
                if progress_callback:
                    progress_callback(i + 1, total, f"Processed {Path(file_path).name}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                # Continue with other files
                continue
        
        return results