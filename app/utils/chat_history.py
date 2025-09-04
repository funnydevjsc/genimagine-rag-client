"""
Chat history utility module.
This module handles chat history operations.
"""
import json
import os

from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage

from app.config.settings import CHAT_HISTORY_FOLDER


def message_to_dict(message):
    """
    Convert a message to a dictionary.
    
    Args:
        message: The message to convert
        
    Returns:
        dict: The message as a dictionary
    """
    if isinstance(message, HumanMessage):
        return {'type': 'human', 'content': message.content, 'additional_kwargs': message.additional_kwargs}
    elif isinstance(message, AIMessage):
        return {'type': 'ai', 'content': message.content, 'additional_kwargs': message.additional_kwargs}
    return {}

def dict_to_message(message_dict):
    """
    Convert a dictionary to a message.
    
    Args:
        message_dict (dict): The dictionary to convert
        
    Returns:
        HumanMessage or AIMessage: The converted message
    """
    if message_dict['type'] == 'human':
        return HumanMessage(content=message_dict['content'], additional_kwargs=message_dict.get('additional_kwargs', {}))
    elif message_dict['type'] == 'ai':
        return AIMessage(content=message_dict['content'], additional_kwargs=message_dict.get('additional_kwargs', {}))
    return None

def update_conversation(store, category, file_path="conversation_data.json"):
    """
    Update conversation history in a JSON file by category.
    
    Args:
        store: The store containing chat history
        category (str): The category of the conversation
        file_path (str): The path to the JSON file
    """
    conversation = {}

    os.makedirs(CHAT_HISTORY_FOLDER, exist_ok=True)
    file_path = os.path.join(CHAT_HISTORY_FOLDER, file_path)

    # Convert `store` to a dictionary to save
    for user_id, chat_history in store.items():
        conversation[user_id] = {
            category: [message_to_dict(msg) for msg in chat_history.messages]
        }

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)

        # Merge new data with existing data
        for user_id, new_data in conversation.items():
            if user_id not in existing_data:
                existing_data[user_id] = new_data
            else:
                if category not in existing_data[user_id]:
                    existing_data[user_id][category] = new_data[category]
                else:
                    # Add new messages without duplicates
                    for message in new_data[category]:
                        if message not in existing_data[user_id][category]:
                            existing_data[user_id][category].append(message)
    else:
        existing_data = conversation

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

def load_previous_conversation(user_id, category, file_path="conversation_data.json"):
    """
    Load conversation history from a JSON file for a specific user and category.
    
    Args:
        user_id (str): The user ID
        category (str): The category of the conversation
        file_path (str): The path to the JSON file
        
    Returns:
        tuple: A tuple containing the first message and recent messages
    """
    os.makedirs(CHAT_HISTORY_FOLDER, exist_ok=True)
    file_path = os.path.join(CHAT_HISTORY_FOLDER, file_path)

    if not os.path.exists(file_path):
        return None, None

    with open(file_path, 'r', encoding='utf-8') as file:
        conversation_data = json.load(file)

    if user_id not in conversation_data or category not in conversation_data[user_id]:
        return None, None

    user_messages = conversation_data[user_id][category]
    if not user_messages:
        return None, None

    first_message = dict_to_message(user_messages[0])
    recent_messages = [dict_to_message(msg) for msg in user_messages[-5:]]
    return first_message, recent_messages

def initialize_session_from_history(chat_history, first_message, recent_messages):
    """
    Initialize a chat session from loaded history.
    
    Args:
        chat_history: The chat history to initialize
        first_message: The first message
        recent_messages: Recent messages
    """
    if first_message and not any(msg.content == first_message.content for msg in chat_history.messages):
        chat_history.add_message(first_message)

    for msg in recent_messages:
        if msg and not any(existing_msg.content == msg.content for existing_msg in chat_history.messages):
            chat_history.add_message(msg)