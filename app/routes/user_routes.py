"""
User routes module.
This module contains API routes for user-related operations.
"""
from fastapi import APIRouter, Depends

from app.database.vector_db import create_collection, add_documents
from app.database.vector_db import delete_collection
from app.models.user_models import UserRegister, TextData
from app.routes.auth import validate_user_agent
from app.utils.document_processing import load_text

router = APIRouter(tags=["User Management"])

@router.post("/register", dependencies=[Depends(validate_user_agent)])
def register(user: UserRegister):
    """
    Register a new user.
    """
    create_collection(user.username)
    return {"message": "Data created successfully"}

@router.post("/update_business", dependencies=[Depends(validate_user_agent)])
def update_text_data(text_data: TextData):
    """
    Update business text data.
    """
    metadata, text = text_data.title, text_data.text

    documents = load_text(metadata, text)
    create_collection(str(text_data.username))
    add_documents(documents, collection_name=str(text_data.username), embeddings=None, subject=None)  # embeddings will be filled in by the caller
    return {"message": "Data updated successfully"}

@router.delete("/delete", dependencies=[Depends(validate_user_agent)])
def delete_data(user_name: str):
    """
    Delete user data.
    """
    delete_collection(user_name)
    return {"message": "Data deleted successfully"}
