"""
Main application module.
This is the entry point for the application.
"""
import uvicorn
from fastapi import FastAPI
from langchain.globals import set_verbose

from app.config.gpu_config import configure_gpu
from app.database.vector_db import initialize_embeddings
from app.routes import user_routes, chatbot_routes

# Disable verbose output
set_verbose(False)

# Initialize FastAPI app
app = FastAPI(docs_url=None, redoc_url=None)

# Configure GPU
gpu_info = configure_gpu()

# Initialize embeddings
embeddings = initialize_embeddings(gpu_info)

# Include routers
app.include_router(user_routes.router)
app.include_router(chatbot_routes.router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
