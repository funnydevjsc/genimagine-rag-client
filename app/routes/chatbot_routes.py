"""
Chatbot routes module.
This module contains API routes for chatbot-related operations.
"""
import json
import os
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException

from app.chatbot.rag import answer_business, answer_user
from app.config.settings import MODEL_BASE_URL
from app.database.vector_db import add_documents
from app.database.vector_db import create_collection
from app.models.chatbot_models import AskData, AskBusiness
from app.models.user_models import AddQABusiness, AddQA
from app.routes.auth import validate_user_agent
from app.utils.document_processing import load_qa
from app.utils.text_processing import format_response

router = APIRouter(tags=["Chatbot"])

@router.get("/health/ollama")
def check_ollama_health():
    """
    Check if the Ollama service is running and accessible.

    Returns:
        dict: A dictionary with the status of the Ollama service
    """
    try:
        # Try to connect to Ollama
        response = httpx.get(f"{MODEL_BASE_URL}/api/tags", timeout=5.0)
        if response.status_code == 200:
            return {
                "status": "ok",
                "message": f"Ollama service is running at {MODEL_BASE_URL}",
                "models": response.json().get("models", [])
            }
        else:
            return {
                "status": "error",
                "message": f"Ollama service returned status code {response.status_code}",
                "url": MODEL_BASE_URL
            }
    except httpx.ConnectError:
        return {
            "status": "error",
            "message": f"Could not connect to Ollama service at {MODEL_BASE_URL}. Is Ollama running?",
            "url": MODEL_BASE_URL,
            "help": "You can configure the Ollama URL using the OLLAMA_BASE_URL environment variable."
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred while checking Ollama health: {str(e)}",
            "url": MODEL_BASE_URL
        }

@router.post("/ask_bot", dependencies=[Depends(validate_user_agent)])
def ask_question(data: AskData):
    """
    Ask a question to the bot.
    """
    start_time = time.time()

    answer = answer_business(data.subject, data.question, data.username)

    # Calculate response time
    response_time = time.time() - start_time

    return format_response(answer, response_time)

@router.post("/ask_business", dependencies=[Depends(validate_user_agent)])
def ask_business_question(data: AskBusiness):
    """
    Ask a business-related question.
    """
    answer = answer_user(data.question, data.username)

    return format_response(answer)

@router.post("/add_qa_for_business", dependencies=[Depends(validate_user_agent)])
def add_qa_business(data: AddQABusiness):
    """
    Add a QA pair for business.
    """
    try:
        text = [f"{data.question}\n{data.answer}"]
        documents = load_qa(data.username, text)
        create_collection(str(f"{data.username}"))
        add_documents(documents, collection_name=str(f"{data.username}"), embeddings=None, subject=None)

        qa_dict = {
            "question": data.question,
            "answer": data.answer
        }

        file_path = os.path.abspath(f"users/{data.username}.json")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                qa_list = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            qa_list = []

        qa_list.append(qa_dict)

        # Write the entire list to the JSON file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=4)
        del qa_list

        return {"message": "add QA successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/add_qa_bot", dependencies=[Depends(validate_user_agent)])
def add_qa_bot(data: AddQA):
    """
    Add a QA pair for the bot.
    """
    try:
        text = [f"{data.question}\n{data.answer}"]
        documents = load_qa(data.subject, text)
        create_collection(str(f"{data.subject}"))
        add_documents(documents, collection_name=str(f"{data.subject}"), embeddings=None, subject=data.subject)

        qa_dict = {
            "subject": data.subject,
            "question": data.question,
            "answer": data.answer
        }

        file_path = "qa_data_fixed.json"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                qa_list = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            qa_list = []

        qa_list.append(qa_dict)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=4)

        del qa_list

        return {"message": "add QA successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Value error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
