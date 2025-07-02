"""
Utility functions for the MultiModal RAG System.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing configuration file: {e}")
        return {}

def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_directory_exists(directory: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    Path(directory).mkdir(parents=True, exist_ok=True)

def validate_file_size(file_path: str, max_size_mb: int = 50) -> bool:
    """Validate file size is within limits."""
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return file_size_mb <= max_size_mb

def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase."""
    return Path(filename).suffix.lower().lstrip('.')

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

class ProgressTracker:
    """Track progress for long-running operations."""
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
    
    def update(self, step_description: str = ""):
        """Update progress bar and status."""
        self.current_step += 1
        progress = self.current_step / self.total_steps
        self.progress_bar.progress(progress)
        
        status = f"{self.description}: {self.current_step}/{self.total_steps}"
        if step_description:
            status += f" - {step_description}"
        
        self.status_text.text(status)
    
    def complete(self, message: str = "Completed!"):
        """Mark progress as complete."""
        self.progress_bar.progress(1.0)
        self.status_text.success(message)