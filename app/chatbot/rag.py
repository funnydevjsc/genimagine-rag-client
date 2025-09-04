"""
RAG (Retrieval Augmented Generation) module.
This module handles RAG functionality for the chatbot.
"""
import time
import uuid

import httpx
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.chatbot.model import initialize_model
from app.chatbot.prompts import get_contextualize_q_prompt, get_qa_prompt, get_user_qa_prompt
from app.database.vector_db import get_vector_store
from app.utils.chat_history import load_previous_conversation, initialize_session_from_history, update_conversation


class TimedRetriever:
    """
    A retriever that tracks retrieval time.
    """
    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(self, query):
        retrieval_start = time.time()
        # Check if query is a string or a dict
        if isinstance(query, str):
            # If it's a string, wrap it in a dict as expected by history_aware_retriever
            docs = self.retriever.invoke({"input": query})
        else:
            # If it's already a dict, pass it through
            docs = self.retriever.invoke(query)
        retrieval_time = time.time() - retrieval_start
        return {"documents": docs, "retrieval_time": retrieval_time}

def answer_business(subject: str, question: str, user_id: str) -> str:
    """
    Answer a business-related question.

    Args:
        subject (str): The subject of the question
        question (str): The question to answer
        user_id (str): The user ID

    Returns:
        str: The answer or an error message if something goes wrong
    """
    # Start timing for performance monitoring
    start_time = time.time()

    try:
        # Initialize model
        model = initialize_model()
    except ConnectionError as e:
        # Return a user-friendly error message
        return f"Error: {str(e)}"
    except Exception as e:
        # Return a generic error message for other exceptions
        return f"An unexpected error occurred: {str(e)}"

    # Log the request for analytics
    store = {}

    # Load only the last few messages instead of the entire history
    first_message, recent_messages = load_previous_conversation(user_id, subject, f"{user_id}.txt")

    if user_id not in store:
        store[user_id] = InMemoryChatMessageHistory()
    if first_message is not None and recent_messages is not None:
        initialize_session_from_history(store[user_id], first_message, recent_messages)

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # Get contextualize question prompt
    contextualize_q_prompt = get_contextualize_q_prompt()

    # Get retrievers with optimized parameters - adjust based on subject
    if subject == "political":
        # Highly optimized political subject for maximum speed
        retriever_1 = get_vector_store(
            collection_name=f"{subject}",
            embeddings=None,  # This will be filled in by the caller
            search_limit=5,  # Drastically reduced from 10 to improve speed
            score_threshold=0.8,  # Increased from 0.75 to focus only on highest quality matches
            subject=subject  # Pass subject for dynamic embedding selection
        )
        retriever_2 = get_vector_store(
            collection_name="base_knowledge",
            embeddings=None,  # This will be filled in by the caller
            search_limit=3,  # Reduced from 5 to improve speed
            score_threshold=0.85,  # Increased from 0.8 to focus only on highest quality matches
            subject=subject  # Pass subject for dynamic embedding selection
        )
        weights = [0.95, 0.05]  # Almost exclusively use subject-specific knowledge
    elif subject in ["legal", "history"]:
        # These subjects need higher precision
        retriever_1 = get_vector_store(
            collection_name=f"{subject}",
            embeddings=None,  # This will be filled in by the caller
            search_limit=20,  # More results for complex topics
            score_threshold=0.65,  # Lower threshold to get more context
            subject=subject  # Pass subject for dynamic embedding selection
        )
        retriever_2 = get_vector_store(
            collection_name="base_knowledge",
            embeddings=None,  # This will be filled in by the caller
            search_limit=10,
            score_threshold=0.75,
            subject=subject  # Pass subject for dynamic embedding selection
        )
        weights = [0.8, 0.2]  # Heavily prioritize subject-specific knowledge
    else:
        # Standard retrieval for other subjects
        retriever_1 = get_vector_store(
            collection_name=f"{subject}",
            embeddings=None,  # This will be filled in by the caller
            search_limit=15,
            score_threshold=0.7,
            subject=subject  # Pass subject for dynamic embedding selection
        )
        retriever_2 = get_vector_store(
            collection_name="base_knowledge",
            embeddings=None,  # This will be filled in by the caller
            search_limit=10,
            score_threshold=0.8,
            subject=subject  # Pass subject for dynamic embedding selection
        )
        weights = [0.7, 0.3]

    # Create ensemble retriever with optimized weights
    retriever = EnsembleRetriever(
        retrievers=[retriever_1, retriever_2],
        weights=weights
    )

    # Create history-aware retriever with timing
    base_history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    # Wrap with timed retriever
    timed_retriever = TimedRetriever(base_history_aware_retriever)

    # Get QA prompt
    qa_prompt = get_qa_prompt(subject)

    # Create question-answer chain
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    # Custom RAG chain with timing and optimizations
    def timed_rag_chain(inputs):
        # Get contextualized question and retrieve documents with timing
        # Make sure we're passing the entire inputs dict to the retriever
        # This ensures chat_history is available for the history-aware retriever
        retriever_output = timed_retriever.invoke(inputs)
        documents = retriever_output["documents"]
        retrieval_time = retriever_output.get("retrieval_time", 0)

        # Highly optimize for political subject - drastically limit documents to reduce context size
        if subject == "political":
            # For political questions, use minimal context to maximize speed
            if hasattr(documents, "sort"):
                try:
                    # Try to sort by score if available and take only top 3 (reduced from 5)
                    documents = sorted(documents, key=lambda doc: doc.metadata.get("score", 0), reverse=True)[:3]
                except:
                    # If sorting fails, just take the first 3 (reduced from 5)
                    documents = documents[:3]
            else:
                # If documents can't be sorted, limit to first 3 (reduced from 5)
                documents = documents[:3]

            # For political questions, also truncate document content to reduce context size
            for i, doc in enumerate(documents):
                if hasattr(doc, "page_content") and len(doc.page_content) > 500:
                    # Truncate long documents to 500 characters
                    documents[i].page_content = doc.page_content[:500] + "..."

        # Generate answer
        generation_start = time.time()
        answer = question_answer_chain.invoke({
            "context": documents,
            "chat_history": inputs.get("chat_history", []),
            "input": inputs["input"]
        })
        generation_time = time.time() - generation_start

        # Return answer with timing information
        return {
            "answer": answer,
            "retrieval_time": retrieval_time,
            "generation_time": generation_time,
            "documents": documents
        }

    # Create conversational chain with message history
    # Wrap the timed_rag_chain function with RunnableLambda to make it a proper Runnable object
    runnable_chain = RunnableLambda(timed_rag_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        runnable_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Set timeout for LLM to prevent hanging
    # Initialize answer_result to a default value
    answer_result = {}
    try:
        answer_result = conversational_rag_chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": f"{user_id}"}
            },
        )

        # Process the answer
        if isinstance(answer_result, dict):
            if "answer" in answer_result:
                if isinstance(answer_result["answer"], dict) and "content" in answer_result["answer"]:
                    answer = answer_result["answer"]["content"].strip()
                else:
                    answer = str(answer_result["answer"]).strip()
            else:
                # Fallback if answer not found in expected format
                answer = str(answer_result).strip()
        else:
            answer = str(answer_result).strip()

        answer = answer.replace("<start>\n", "").replace("<end>\n", "")
        update_conversation(store, subject, file_path=f"{user_id}.txt")
    except httpx.ConnectError as e:
        # Return a user-friendly error message for connection errors
        answer = f"Error: Could not connect to Ollama service. Is Ollama running? Details: {str(e)}"
    except Exception as e:
        # Return a generic error message for other exceptions
        answer = f"An unexpected error occurred while processing your question: {str(e)}"

    return answer

def answer_user(question: str, user_id: str) -> str:
    """
    Answer a user-specific question.

    Args:
        question (str): The question to answer
        user_id (str): The user ID

    Returns:
        str: The answer or an error message if something goes wrong
    """
    try:
        # Initialize model
        model = initialize_model()
    except ConnectionError as e:
        # Return a user-friendly error message
        return f"Error: {str(e)}"
    except Exception as e:
        # Return a generic error message for other exceptions
        return f"An unexpected error occurred: {str(e)}"

    # Create temp uuid 
    session_id = str(uuid.uuid4())

    # Store chat history 
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    # Get contextualize question prompt
    contextualize_q_prompt = get_contextualize_q_prompt()

    # Get retriever with optimized parameters for user-specific knowledge
    retriever = get_vector_store(
        collection_name=user_id,
        embeddings=None,  # This will be filled in by the caller
        search_limit=25,  # Higher limit for user-specific knowledge
        score_threshold=0.65,  # Lower threshold for user-specific knowledge to ensure more results
        subject=None  # Use default embeddings for user-specific knowledge
    )

    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    # Get user QA prompt
    qa_prompt = get_user_qa_prompt()

    # Create question-answer chain
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

    # Create retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Create conversational chain with message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # Invoke the chain
    try:
        answer_result = conversational_rag_chain.invoke(
            {"input": question},
            config={
                "configurable": {"session_id": f"{session_id}"}
            },
        )

        # Process the answer
        answer = answer_result["answer"].strip()
    except httpx.ConnectError as e:
        # Return a user-friendly error message for connection errors
        answer = f"Error: Could not connect to Ollama service. Is Ollama running? Details: {str(e)}"
    except Exception as e:
        # Return a generic error message for other exceptions
        answer = f"An unexpected error occurred while processing your question: {str(e)}"

    return answer
