"""
Document processing utility module.
This module handles document processing operations.
"""
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader

from app.utils.merge_meaning import SemanticChunker


def load_text(metadata: str, text: str) -> list[Document]:
    """
    Load and chunk text data.
    
    Args:
        metadata (str): The metadata for the text
        text (str): The text to load and chunk
        
    Returns:
        list[Document]: A list of Document objects
    """
    chunker = SemanticChunker(
        min_sentences=2,
        max_sentences=20,
        similarity_threshold=0.3
    )
    chunks = chunker.create_semantic_chunks(text)
    metadata_dict = {"source": f"{metadata}"}
    documents = [Document(metadata=metadata_dict, page_content=chunk) for chunk in chunks]
    
    return documents

def load_qa(metadata: str, chunks: list[str]) -> list[Document]:
    """
    Load question-answer pairs.
    
    Args:
        metadata (str): The metadata for the QA pairs
        chunks (list[str]): The QA pairs to load
        
    Returns:
        list[Document]: A list of Document objects
    """
    metadata_dict = {"source": f"{metadata}"}
    documents = [Document(metadata=metadata_dict, page_content=chunk) for chunk in chunks]
    return documents

def load_data(file_name: str) -> list[Document]:
    """
    Load data from a file.
    
    Args:
        file_name (str): The name of the file to load
        
    Returns:
        list[Document]: A list of Document objects
    """
    loader = TextLoader(file_name)
    text_documents = loader.load()
    with open(file_name, 'r') as file:
        content = file.readline().strip()

    chunker = SemanticChunker(
        min_sentences=5,
        max_sentences=20,
        similarity_threshold=0.3
    )

    chunks = chunker.create_semantic_chunks(text_documents[0].page_content)
    metadata = {"source": f"{content}"}
    documents = [Document(metadata=metadata, page_content=chunk) for chunk in chunks]

    return documents