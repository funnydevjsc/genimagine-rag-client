"""
Text processing utility module.
This module handles text processing operations.
"""
import json
import re
from typing import Any, Dict, Union

def process_json_response(answer: str) -> str:
    """
    Process a response that might be in JSON format.
    
    Args:
        answer (str): The response to process
        
    Returns:
        str: The processed response
    """
    # Check if the answer looks like a JSON string
    if isinstance(answer, str) and ((answer.startswith('{') and answer.endswith('}')) or (answer.startswith('[') and answer.endswith(']'))):
        try:
            # Try to parse as JSON
            json_obj = json.loads(answer)

            # If it's a simple key-value structure, extract the value
            if isinstance(json_obj, dict):
                # Look for common response fields
                for key in ['answer', 'response', 'text', 'content', 'message', 'result', 'người trả lời']:
                    if key in json_obj:
                        if isinstance(json_obj[key], dict):
                            # If the value is another dictionary, try to extract a meaningful value
                            for subkey in ['name', 'value', 'text', 'content', 'message', 'tên', 'id']:
                                if subkey in json_obj[key]:
                                    if subkey == 'tên':
                                        return json_obj[key][subkey]
                            # If no meaningful subkey found, convert the dict to a string
                            return str(json_obj[key])
                        return json_obj[key]

                # If no common fields found, just take the first value
                if json_obj:
                    first_value = next(iter(json_obj.values()))
                    if isinstance(first_value, dict):
                        # If the first value is a dict, try to extract a meaningful value
                        for subkey in ['name', 'value', 'text', 'content', 'message', 'tên', 'id']:
                            if subkey in first_value:
                                if subkey == 'tên':
                                    return first_value[subkey]
                        # If no meaningful subkey found, convert the dict to a string
                        return str(first_value)
                    return first_value

            # If it's a list, join the elements
            elif isinstance(json_obj, list) and all(isinstance(item, str) for item in json_obj):
                return ' '.join(json_obj)

            return answer
        except json.JSONDecodeError:
            # Not valid JSON, continue with original answer
            pass

    # Check for JSON-like patterns that might not be valid JSON
    if isinstance(answer, str):
        json_pattern = r'[\{\[].*?[\}\]]'
        if re.search(json_pattern, answer):
            # Try to extract text content from patterns like {"text": "actual answer"}
            text_pattern = r'"(?:answer|response|text|content|message|result|người trả lời|tên)"\s*:\s*"([^"]+)"'
            match = re.search(text_pattern, answer)
            if match:
                return match.group(1)

    # If we get here, just return the original answer
    if isinstance(answer, str) and "người trả lời" in answer and "tên" in answer:
        # Hardcoded fallback for the specific pattern we're seeing
        try:
            # Try to extract the name directly with regex
            name_pattern = r'"tên"\s*:\s*"([^"]+)"'
            match = re.search(name_pattern, answer)
            if match:
                return match.group(1)
        except:
            pass

    return answer

def format_response(answer: Union[str, Dict[str, Any]], response_time: float = None) -> Dict[str, Any]:
    """
    Format a response for API output.
    
    Args:
        answer (Union[str, Dict[str, Any]]): The answer to format
        response_time (float, optional): The response time in seconds
        
    Returns:
        Dict[str, Any]: The formatted response
    """
    # Extract answer from result, handling different result formats
    if isinstance(answer, dict):
        if "answer" in answer:
            if isinstance(answer["answer"], dict) and "content" in answer["answer"]:
                processed_answer = answer["answer"]["content"].strip()
            else:
                processed_answer = str(answer["answer"]).strip()
        else:
            # Fallback if answer not found in expected format
            processed_answer = str(answer).strip()
    else:
        processed_answer = str(answer).strip()

    processed_answer = processed_answer.replace("<start>\n", "").replace("<end>\n", "")
    
    # Process JSON responses
    processed_answer = process_json_response(processed_answer)
    
    # Create response object
    response = {"message": processed_answer}
    
    # Add response time if provided
    if response_time is not None:
        response["response_time_seconds"] = response_time
        
    return response