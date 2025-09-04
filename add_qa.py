import json
import time

import portalocker
from qdrant_client import QdrantClient

from data_insert import load_qa, chunked_metadata

# Collection name for Q&A data
COLLECTION_NAME = "based_knowledge"

# Initialize Qdrant client with retry mechanism or in-memory mode
def initialize_qdrant_client(max_retries=5, retry_delay=2, use_in_memory=False):
    """
    Initialize Qdrant client with retry mechanism to handle locking issues.
    If retries fail or use_in_memory is True, falls back to in-memory mode.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        use_in_memory: If True, uses in-memory mode directly without trying file-based storage

    Returns:
        QdrantClient instance or None if all attempts fail
    """
    # If use_in_memory is True, skip the retry mechanism and use in-memory mode directly
    if use_in_memory:
        print("Using in-memory mode for Qdrant client...")
        try:
            client = QdrantClient(":memory:")
            print("Successfully initialized in-memory Qdrant client")
            return client
        except Exception as e:
            print(f"Error initializing in-memory Qdrant client: {e}")
            return None

    # Try file-based storage with retry mechanism
    for attempt in range(max_retries):
        try:
            print(f"Attempting to connect to Qdrant (attempt {attempt + 1}/{max_retries})...")
            # Try to initialize the client
            client = QdrantClient(url="http://localhost:6333")
            print("Successfully connected to Qdrant Docker container")
            return client
        except portalocker.exceptions.AlreadyLocked:
            print(f"Qdrant storage is locked. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        except RuntimeError as e:
            if "already accessed by another instance" in str(e):
                print(f"Qdrant storage is in use. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Unexpected error: {e}")
                raise

    print("Failed to connect to Qdrant after multiple attempts.")
    print("Falling back to in-memory mode...")

    # Fall back to in-memory mode
    try:
        client = QdrantClient(":memory:")
        print("Successfully initialized in-memory Qdrant client")
        return client
    except Exception as e:
        print(f"Error initializing in-memory Qdrant client: {e}")
        return None

# Initialize Qdrant client with Docker container mode
# Using Docker container for better performance with large datasets
# This connects to the Qdrant instance running in Docker at localhost:6333
client = initialize_qdrant_client(use_in_memory=False)

# Check if client was successfully initialized
if client is None:
    print("Error: Could not initialize Qdrant client. Exiting.")
    exit(1)

def load_qa_from_json(json_file_path, collection_name=COLLECTION_NAME, custom_client=None, handle_duplicates='keep_last'):
    """
    Load Q&A pairs from a JSON file and add them to the specified collection.

    Args:
        json_file_path: Path to the JSON file containing Q&A pairs
        collection_name: Name of the collection to add the data to
        custom_client: Optional QdrantClient instance to use
        handle_duplicates: Strategy for handling duplicate questions:
            - 'keep_first': Keep only the first occurrence of a question (default)
            - 'keep_last': Keep only the last occurrence of a question
            - 'keep_all': Keep all occurrences of a question (not recommended for conflicting answers)
    """
    # Use the provided client or fall back to the global client
    c = custom_client if custom_client is not None else client
    print(f"Loading Q&A data from {json_file_path}...")

    # Read the JSON file
    with open(json_file_path, 'r', encoding='utf-8') as file:
        qa_data = json.load(file)

    print(f"Found {len(qa_data)} Q&A pairs")

    # Dictionary to track questions and detect duplicates
    # This dictionary will store questions as keys and (subject, answer) tuples as values
    # It allows us to detect duplicate questions and handle them according to the strategy
    question_dict = {}
    duplicate_count = 0

    # First pass: Process each Q&A pair and handle duplicates
    # We do this in two passes to ensure we handle all duplicates before creating documents
    for i, qa_pair in enumerate(qa_data):
        subject = qa_pair.get("subject", "other")
        question = qa_pair.get("question", "")
        answer = qa_pair.get("answer", "")

        # Skip empty entries
        if not question or not answer:
            continue

        # Check for duplicates
        if question in question_dict:
            duplicate_count += 1
            existing_subject, existing_answer = question_dict[question]

            if handle_duplicates == 'keep_first':
                # Skip this duplicate (keep the first occurrence)
                print(f"Duplicate question found: '{question[:50]}...' - Keeping first occurrence")
                continue
            elif handle_duplicates == 'keep_last':
                # Replace with the latest occurrence
                print(f"Duplicate question found: '{question[:50]}...' - Keeping last occurrence")
                question_dict[question] = (subject, answer)
            # For 'keep_all', we would process all occurrences, but this is not recommended
            # for conflicting answers as it would create confusion in the model
        else:
            # New question, add to dictionary
            question_dict[question] = (subject, answer)

    print(f"Found {duplicate_count} duplicate questions")

    # Second pass: Create documents from the filtered question dictionary
    all_documents = []
    processed_count = 0

    for question, (subject, answer) in question_dict.items():
        # Format the content as question and answer
        content = f"Câu hỏi: {question}\nCâu trả lời: {answer}"

        # Create metadata
        metadata = f"{subject}: {question[:30]}..."

        # Create document using load_qa function
        document = load_qa(metadata, [content])
        all_documents.extend(document)

        processed_count += 1
        # Log progress
        if processed_count % 100 == 0:
            print(f"Processed {processed_count} unique Q&A pairs")

    # Add all documents to the collection
    if all_documents:
        print(f"Adding {len(all_documents)} documents to collection {collection_name}")
        chunked_metadata(all_documents, collection_name, custom_client=c)
        print("Q&A data successfully added to the model")
    else:
        print("No valid Q&A pairs found")

# Execute the function if this script is run directly
if __name__ == "__main__":
    if client is not None:
        # Default to 'keep_first' strategy for handling duplicates
        # Change to 'keep_last' if you want to keep the most recent answer for duplicate questions
        duplicate_strategy = 'keep_last'

        # Load the Q&A data with duplicate handling
        load_qa_from_json("qa_data_fixed.json", 
                         custom_client=client, 
                         handle_duplicates=duplicate_strategy)

        print("\nIMPORTANT NOTES:")
        print("1. The JSON file is only needed during the loading process. Once the data")
        print("   is successfully loaded into the model, you can archive the JSON file.")
        print("2. If you need to update the model with new Q&A pairs, you'll need the JSON file again.")
        print("3. When using in-memory mode, the data will be lost when the script exits.")
        print("   For persistent storage, ensure no other process is using the Qdrant storage.")
        print(f"4. Duplicate questions are handled using the '{duplicate_strategy}' strategy.")
        print("   - 'keep_first': Keeps only the first occurrence of a question")
        print("   - 'keep_last': Keeps only the last occurrence of a question")
        print("   To change this behavior, modify the 'duplicate_strategy' variable in this script.")
    else:
        print("Error: Could not initialize Qdrant client. Exiting.")
