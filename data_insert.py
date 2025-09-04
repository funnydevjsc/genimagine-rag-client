import gc
import hashlib
import time

from langchain.schema import Document
from langchain_community.document_loaders import TextLoader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from app.utils.merge_meaning import SemanticChunker

client = QdrantClient(url="http://localhost:6333")  # (":memory:")

from langchain_huggingface import HuggingFaceEmbeddings
import torch

# Initialize torch and CUDA
print("\n=== GPU Configuration ===")
if torch.cuda.is_available():
    # Get GPU details
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    print(f"✓ CUDA is available")
    print(f"✓ Using GPU: {gpu_name} (Tesla P40)")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ PyTorch Version: {torch.__version__}")
    print(f"✓ Device Count: {torch.cuda.device_count()}")
    print(f"✓ Current Device: {torch.cuda.current_device()}")
    print(f"✓ GPU Memory: {total_memory:.2f}GB")

    # Tesla P40 specific optimizations
    print("✓ Applying Tesla P40 optimizations...")

    # Enable TF32 precision for better performance on Tesla P40
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # Clear GPU memory at startup
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
        torch.cuda.reset_peak_memory_stats()

    # Set memory allocation strategy for Tesla P40
    if hasattr(torch.cuda, 'memory_stats'):
        # Tesla P40 has 24GB, we can use up to 80% safely
        torch.cuda.set_per_process_memory_fraction(0.8)

    # Set optimal thread count for Tesla P40
    if hasattr(torch, 'set_num_threads'):
        # Tesla P40 works well with this thread configuration
        torch.set_num_threads(4)

    # Enable CUDA graph capture for repeated operations if available
    if hasattr(torch.cuda, 'is_available') and torch.__version__ >= '1.10.0':
        torch.jit.enable_onednn_fusion(True)

    print("✓ Tesla P40 optimizations applied")
    print("========================\n")

    # Force model to use CUDA with Tesla P40 optimizations
    device = "cuda:0"

    # Initialize embeddings with Tesla P40 optimizations
    print("Initializing embeddings with Tesla P40 optimizations...")
    # Optimize for Tesla P40: Pre-allocate GPU memory if needed
    allocated_memory = torch.cuda.memory_allocated(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = total_memory - allocated_memory
    total_free = total_memory - allocated_memory

    # If memory is fragmented (reserved but not used), clear cache
    if total_free - free_memory > 1 * 1024 * 1024 * 1024:  # 1GB difference
        torch.cuda.empty_cache()

    # Use optimized model kwargs for Tesla P40
    model_kwargs = {
        'device': device
    }

    # Initialize embeddings with context manager for better memory handling
    with torch.no_grad():  # Disable gradient calculation for embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name='./vietnamese-bi-encoder',
            model_kwargs=model_kwargs
        )
    print("Embeddings initialized with Tesla P40 optimizations")
else:
    print("✗ CUDA is NOT available - application will use CPU")
    print("✗ This will significantly reduce performance")
    print("✗ Check your PyTorch installation and GPU drivers")
    print("========================\n")

    # CPU fallback
    device = "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name='./vietnamese-bi-encoder',
        model_kwargs={'device': device}
    )

# Force garbage collection at startup
gc.collect()

# from langchain_ollama import OllamaEmbeddings
# embeddings = OllamaEmbeddings(model="llama3.2:1b")

def create_collections(uuid, custom_client=None):
    # Use the provided client or fall back to the global client
    c = custom_client if custom_client is not None else client

    if not c.collection_exists(uuid):
            c.create_collection(
                collection_name=uuid,
                vectors_config={
                    "content": VectorParams(
                        size=768, 
                        distance=Distance.COSINE,
                        # Optimized HNSW index parameters
                        hnsw_config={
                            "m": 16,  # Number of bidirectional links created for each new element (higher = better recall, more memory)
                            "ef_construct": 200,  # Size of the dynamic candidate list during index building (higher = better recall, slower build)
                            "full_scan_threshold": 10000,  # Threshold for full scan vs HNSW search (higher = more accurate, slower)
                        }
                    )
                },
                # Add optimized options for collection
                optimizers_config={
                    "default_segment_number": 2,  # Optimal number of segments for this collection size
                    "indexing_threshold": 20000,  # Threshold for creating index (smaller = faster updates, more memory)
                    "memmap_threshold": 50000,  # Threshold for using memmap (larger = more RAM usage, faster)
                    "vacuum_min_vector_number": 1000,  # Minimum number of vectors to vacuum (smaller = more frequent vacuuming)
                }
            )

def load_text(metadata, text):
    chunker = SemanticChunker(
    min_sentences= 2,
    max_sentences=20,
    similarity_threshold=0.3
    )
    chunks = chunker.create_semantic_chunks(text)
    metadata = {"source": f"{metadata}"}
    document = [Document(metadata=metadata, page_content=chunk) for chunk in chunks]

    return document

def load_qa(metadata, chunks):
    metadata = {"source": f"{metadata}"}
    document = [Document(metadata=metadata, page_content=chunk) for chunk in chunks]
    return document

def load_data(file_name):
    loader = TextLoader(file_name)
    text_documents=loader.load()
    with open(file_name, 'r') as file:
        content = file.readline().strip()

    chunker = SemanticChunker(
    min_sentences= 5,
    max_sentences=20,
    similarity_threshold=0.3
    )

    chunks = chunker.create_semantic_chunks(text_documents[0].page_content)
    metadata = {"source": f"{content}"}
    document = [Document(metadata=metadata, page_content=chunk) for chunk in chunks]

    return document

def chunked_metadata(data, collection_name = "", custom_client=None, batch_size=100): #collection_name = uuid
    """
    Add documents to a Qdrant collection with content-based IDs to prevent conflicts.

    This function uses a hash of the content as the ID for each point, which ensures:
    1. Identical content will have the same ID (deduplication)
    2. New data won't overwrite existing data with different content
    3. Updates to existing data will replace the old version

    Args:
        data: List of Document objects to add to the collection
        collection_name: Name of the collection to add the data to
        custom_client: Optional QdrantClient instance to use
        batch_size: Number of points to insert in a single batch (default: 100)
    """
    # Use the provided client or fall back to the global client
    c = custom_client if custom_client is not None else client

    # Ensure collection exists and has payload index for source field
    if not c.collection_exists(collection_name):
        create_collections(collection_name, custom_client=c)
    else:
        # Add payload index for source field if it doesn't exist
        try:
            c.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.source",
                field_schema="keyword"
            )
            print(f"Created payload index for 'metadata.source' in collection {collection_name}")
        except Exception as e:
            # Index might already exist, which is fine
            pass

    chunked_metadata = []
    total_points = len(data)
    points_processed = 0

    for item in data:
        content = item.page_content
        source = item.metadata["source"]

        # Generate a deterministic ID based on the content
        # This ensures that identical content will have the same ID
        # and prevents conflicts when adding new data
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
        # Convert the hash to an integer (Qdrant requires integer IDs)
        # We use the first 16 characters of the hash (64 bits)
        point_id = int(content_hash[:16], 16)

        # Tesla P40 optimized embedding generation
        try:
            # Use torch.no_grad() for better memory efficiency on Tesla P40
            with torch.no_grad():
                # Optimize embedding by using embed_query which is faster for single documents
                content_vector = embeddings.embed_query(content)
        except Exception as e:
            print(f"Error during embedding generation: {str(e)}")
            # Tesla P40 optimized recovery procedure
            # Try to clean up memory and retry with more aggressive cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Reset peak memory stats for Tesla P40
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()

            # Wait a bit for Tesla P40 to stabilize
            time.sleep(1)

            # Retry with explicit CUDA synchronization for Tesla P40
            with torch.no_grad():
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                content_vector = embeddings.embed_query(content)

        vector_dict = {"content": content_vector}

        payload = {
            "page_content": content,
            "metadata": {
                        "id": point_id,
                        "source": source,
                        }
            }

        metadata = PointStruct(id=point_id, vector=vector_dict, payload=payload)
        chunked_metadata.append(metadata)

        # Process in batches for better performance
        if len(chunked_metadata) >= batch_size or points_processed == total_points - 1:
            # Use upsert to add or update points
            # If a point with the same ID already exists, it will be updated
            c.upsert(
                collection_name=collection_name,
                points=chunked_metadata,
                wait=False,  # Don't wait for immediate indexing (faster)
            )
            print(f"Inserted batch of {len(chunked_metadata)} points ({points_processed + 1}/{total_points})")
            chunked_metadata = []  # Reset for next batch

        points_processed += 1

    # Final wait to ensure all data is indexed
    if points_processed > 0:
        c.update_collection(
            collection_name=collection_name,
            optimizers_config={
                "indexing_threshold": 0  # Force immediate indexing of all data
            }
        )
        print(f"Completed insertion of {points_processed} points into collection {collection_name}")
        # Reset optimizer config to normal values
        c.update_collection(
            collection_name=collection_name,
            optimizers_config={
                "indexing_threshold": 20000  # Reset to normal value
            }
        )

def delete_collection(uuid, custom_client=None):
    # Use the provided client or fall back to the global client
    c = custom_client if custom_client is not None else client
    c.delete_collection(collection_name=uuid)
