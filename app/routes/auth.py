"""
Authentication module.
This module handles authentication for the API.
"""
from fastapi import Request, HTTPException

from app.config.settings import ALLOWED_USER_AGENTS


def validate_user_agent(request: Request):
    """
    Validate the User-Agent header.
    
    Args:
        request (Request): The request object
        
    Raises:
        HTTPException: If the User-Agent is invalid
    """
    user_agent = request.headers.get("User-Agent", "")
    if user_agent not in ALLOWED_USER_AGENTS:
        raise HTTPException(status_code=403, detail="Forbidden")