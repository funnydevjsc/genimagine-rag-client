"""
Application settings module.
This module contains all configuration settings for the application.
"""
import os
from typing import List

# Environment settings
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

# Ollama settings
OLLAMA_NUM_GPU_LAYERS = os.environ.get("OLLAMA_NUM_GPU_LAYERS", "32")
os.environ["OLLAMA_NUM_GPU_LAYERS"] = OLLAMA_NUM_GPU_LAYERS

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
ALLOWED_USER_AGENTS: List[str] = [
    "Gen Imagine Client"
]

# Vector database settings
QDRANT_URL = "http://localhost:6333"

# Model settings
MODEL_NAME = os.environ.get("MODEL_NAME", "vinallama")
MODEL_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDINGS_MODEL_PATH = os.environ.get("EMBEDDINGS_MODEL_PATH", './vietnamese-bi-encoder')

# Chat settings
CHAT_HISTORY_FOLDER = "chat_history/"
