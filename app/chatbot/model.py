"""
Chatbot model module.
This module handles model initialization for the chatbot.
"""
import httpx
from langchain_ollama import ChatOllama

from app.config.settings import MODEL_NAME, MODEL_BASE_URL


def initialize_model():
    """
    Initialize the ChatOllama model with optimized parameters.

    Returns:
        ChatOllama: The initialized model or None if initialization fails

    Raises:
        ConnectionError: If Ollama service is not running or not accessible
    """
    try:
        # Configure ChatOllama to use GPU if available with highly optimized parameters for speed
        model = ChatOllama(
            model=MODEL_NAME,
            base_url=MODEL_BASE_URL,
            temperature=0.1,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            num_predict=1024,
            num_ctx=2048,
            seed=42,
            stop=["</end>"],  # Ensure the model is required to generate </end> in the prompt
            format=None,
            num_gpu=32,
            num_thread=16,  # Adjust based on your physical system
            keep_alive="10m",  # Keep model in RAM
        )

        # Test the connection to Ollama
        # This will raise an exception if Ollama is not running
        try:
            # Make a simple request to check if Ollama is running
            response = httpx.get(f"{MODEL_BASE_URL}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(f"Ollama service returned status code {response.status_code}")
        except httpx.ConnectError as e:
            raise ConnectionError(f"Could not connect to Ollama service at {MODEL_BASE_URL}. Is Ollama running? You can configure the Ollama URL using the OLLAMA_BASE_URL environment variable.") from e

        return model
    except Exception as e:
        # Re-raise as a ConnectionError with a helpful message
        if isinstance(e, ConnectionError):
            raise
        raise ConnectionError(f"Failed to initialize Ollama model: {str(e)}")
