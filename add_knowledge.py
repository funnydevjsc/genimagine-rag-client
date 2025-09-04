import gc
import hashlib
import json
import os
import threading
import time

import psutil
from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, \
    UnstructuredWordDocumentLoader, UnstructuredPDFLoader, UnstructuredExcelLoader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from tqdm import tqdm

from app.utils.merge_meaning import SemanticChunker

# Constants for resource management
MAX_MEMORY_PERCENT = 85  # Maximum memory usage percentage
CRITICAL_MEMORY_PERCENT = 90  # Critical memory threshold
FORCE_EXIT_MEMORY_PERCENT = 95  # Force exit if memory exceeds this percentage
MIN_AVAILABLE_MEMORY_GB = 2.0  # Minimum available memory in GB
MEMORY_CHECK_INTERVAL = 5  # Memory check interval in seconds
CHECKPOINT_FILE = "knowledge_checkpoint.json"  # Checkpoint file

# from langchain_ollama import OllamaEmbeddings
# embeddings = OllamaEmbeddings(model="llama3.2:1b")

from langchain_huggingface import HuggingFaceEmbeddings
import torch

# Initialize global variables
embeddings = None
client = None

def initialize_embeddings():
    """Initialize the embedding model with proper error handling and optimization."""
    global embeddings, device

    # Check if embeddings are already initialized
    if embeddings is not None:
        return embeddings

    print("🔄 Initializing embedding model...")

    try:
        # Check if CUDA is available and force GPU usage
        if not torch.cuda.is_available():
            print("⚠️ CUDA is not available for embeddings. Using CPU instead.")
            device = "cpu"
        else:
            # Get available GPU memory before loading model
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            free_memory = total_memory - allocated_memory
            print(f"🖥️ Using CUDA device for embeddings: {gpu_name} (Tesla P40)")
            print(f"   GPU memory: {total_memory / (1024**3):.2f}GB total, {free_memory / (1024**3):.2f}GB free")

            # Force garbage collection before loading model
            gc.collect()
            torch.cuda.empty_cache()

            device = "cuda:0"

        # Load the model with optimized settings
        model_kwargs = {
            'device': device
        }

        # Add optimization for CUDA if available
        if device == "cuda:0":
            # Tesla P40 specific optimizations
            # Set lower precision for CUDA operations where possible
            # This doesn't use torch_dtype which is unsupported by SentenceTransformer
            torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn
            torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner for Tesla P40

            # Tesla P40 has 24GB of memory, optimize memory usage
            # Enable memory optimization
            torch.cuda.empty_cache()

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

        # Initialize the embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name='./vietnamese-bi-encoder',
            model_kwargs=model_kwargs
        )

        print("✅ Embedding model initialized successfully")
        return embeddings

    except Exception as e:
        print(f"❌ Error initializing embedding model: {str(e)}")
        # Fallback to CPU if GPU initialization fails
        if 'cuda' in str(e).lower() and device == "cuda:0":
            print("⚠️ Falling back to CPU for embeddings")
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name='./vietnamese-bi-encoder',
                    model_kwargs={'device': 'cpu'}
                )
                print("✅ Embedding model initialized on CPU")
                return embeddings
            except Exception as cpu_e:
                print(f"❌ Failed to initialize embedding model on CPU: {str(cpu_e)}")
                raise
        else:
            raise

def initialize_qdrant_client():
    """Initialize the Qdrant client with proper error handling."""
    global client

    # Check if client is already initialized
    if client is not None:
        return client

    print("🔄 Connecting to Qdrant server...")

    try:
        # Initialize with retry logic
        max_retries = 3
        retry_delay = 2

        for retry in range(max_retries):
            try:
                client = QdrantClient(url="http://localhost:6333")
                # Test connection
                client.get_collections()
                print("✅ Connected to Qdrant server successfully")
                return client
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"⚠️ Error connecting to Qdrant (retry {retry+1}/{max_retries}): {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"❌ Failed to connect to Qdrant after {max_retries} retries: {str(e)}")
                    raise
    except Exception as e:
        print(f"❌ Error initializing Qdrant client: {str(e)}")
        raise

def create_collection(collection_name=''):
    """Create a collection if it doesn't exist with proper error handling."""
    # Initialize client if not already initialized
    client = initialize_qdrant_client()

    try:
        # Check if collection exists
        if not client.collection_exists(collection_name):
            print(f"   Creating collection '{collection_name}'...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "content": VectorParams(size=768, distance=Distance.COSINE)
                }
            )
            print(f"   Collection '{collection_name}' created successfully")
        else:
            print(f"   Collection '{collection_name}' already exists")
    except Exception as e:
        print(f"   ❌ Error creating collection '{collection_name}': {str(e)}")
        raise

def chunked_metadata(data, client=None, collection_name="base_knowledge"):
    """
    Add documents to a Qdrant collection with content-based IDs to prevent conflicts.

    This function uses a hash of the content as the ID for each point, which ensures:
    1. Identical content will have the same ID (deduplication)
    2. New data won't overwrite existing data with different content
    3. Updates to existing data will replace the old version

    Args:
        data: List of Document objects to add to the collection
        client: QdrantClient instance to use (if None, will be initialized)
        collection_name: Name of the collection to add the data to
    """
    # Initialize client and embeddings if not provided
    if client is None:
        client = initialize_qdrant_client()

    # Initialize embeddings
    global embeddings
    if embeddings is None:
        embeddings = initialize_embeddings()
    # Start memory monitoring thread
    memory_monitor_stop = threading.Event()
    memory_monitor = threading.Thread(
        target=memory_monitor_thread_func,
        args=(memory_monitor_stop, collection_name),
        daemon=True
    )
    memory_monitor.start()
    print(f"   Started memory monitoring thread for collection {collection_name}")

    # Calculate batch size based on data size and system resources
    batch_size = calculate_batch_size(len(data))
    print(f"   Processing {len(data)} documents with batch size {batch_size}")

    # Create progress bar
    pbar = tqdm(total=len(data), desc=f"Processing {collection_name}", 
                unit="docs", ncols=100, position=0, leave=True)

    try:
        # Process documents in batches
        total_processed = 0
        for i in range(0, len(data), batch_size):
            # Monitor system resources
            monitor_system_resources()

            # Get current batch
            batch = data[i:i+batch_size]
            chunked_metadata = []

            # Process each document in the batch
            for item in batch:
                content = item.page_content
                source = item.metadata["source"]

                # Generate a deterministic ID based on the content
                content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                point_id = int(content_hash[:16], 16)

                try:
                    # Embed the content with Tesla P40 optimized error handling
                    try:
                        # Optimize for Tesla P40: Pre-allocate GPU memory if needed
                        if torch.cuda.is_available():
                            # Check if GPU memory is fragmented
                            allocated_memory = torch.cuda.memory_allocated(0)
                            total_memory = torch.cuda.get_device_properties(0).total_memory
                            free_memory = total_memory - allocated_memory
                            total_free = total_memory - allocated_memory

                            # If memory is fragmented (reserved but not used), clear cache
                            if total_free - free_memory > 1 * 1024 * 1024 * 1024:  # 1GB difference
                                torch.cuda.empty_cache()

                        # Use a context manager for better memory handling on Tesla P40
                        with torch.no_grad():  # Disable gradient calculation for embeddings
                            content_vector = embeddings.embed_query(content)
                    except Exception as e:
                        print(f"   ⚠️ Error embedding content: {str(e)}")
                        # Tesla P40 optimized recovery procedure
                        # Try to clean up memory and retry with more aggressive cleanup
                        gc.collect(generation=2)  # Full collection
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            # Reset peak memory stats for Tesla P40
                            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                                torch.cuda.reset_peak_memory_stats()

                        # Wait a bit longer for Tesla P40 to stabilize
                        time.sleep(2)

                        try:
                            # Retry with explicit CUDA synchronization for Tesla P40
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            with torch.no_grad():
                                content_vector = embeddings.embed_query(content)
                        except Exception as retry_e:
                            print(f"   ❌ Failed to embed content after retry: {str(retry_e)}")
                            continue

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
                except Exception as e:
                    print(f"   ❌ Error processing document: {str(e)}")

            # Upsert batch with retry logic
            max_retries = 3
            retry_delay = 2
            for retry in range(max_retries):
                try:
                    if chunked_metadata:  # Only upsert if we have data
                        client.upsert(
                            collection_name=collection_name,
                            wait=True,
                            points=chunked_metadata
                        )
                    break  # Success, exit retry loop
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"   ⚠️ Error upserting batch (retry {retry+1}/{max_retries}): {str(e)}")
                        # Clean up memory before retry
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"   ❌ Failed to upsert batch after {max_retries} retries: {str(e)}")

            # Update progress
            total_processed += len(batch)
            pbar.update(len(batch))

            # Save checkpoint every few batches
            if i % (batch_size * 3) == 0 and i > 0:
                save_checkpoint(collection_name, "batch_processing", total_processed, len(data))

            # Clean up after each batch
            chunked_metadata = []
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    finally:
        # Stop memory monitoring thread
        memory_monitor_stop.set()
        memory_monitor.join(timeout=1.0)
        pbar.close()

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_processed

def get_cpu_usage():
    """Get current CPU usage information."""
    cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
    return {
        "overall": sum(cpu_percent) / len(cpu_percent),
        "per_core": cpu_percent,
        "max_core": max(cpu_percent)
    }

def monitor_system_resources():
    """Monitor system resources and take action if necessary."""
    memory = psutil.virtual_memory()
    cpu_info = get_cpu_usage()
    process = psutil.Process(os.getpid())
    process_memory_gb = process.memory_info().rss / (1024 ** 3)
    total_memory_gb = memory.total / (1024 ** 3)

    # Calculate process memory as percentage of total system memory
    process_memory_percent = (process_memory_gb / total_memory_gb) * 100

    # Check if available memory is critically low
    available_memory_gb = memory.available / (1024 ** 3)
    critical_memory_condition = (
        (memory.percent > FORCE_EXIT_MEMORY_PERCENT and process_memory_percent > 50) or
        (available_memory_gb < MIN_AVAILABLE_MEMORY_GB and process_memory_percent > 40) or
        (memory.percent > 99 and process_memory_percent > 30)  # Extreme case
    )

    if critical_memory_condition:
        print(f"\n🚨 EXTREME MEMORY PRESSURE: {memory.percent:.1f}% (Process: {process_memory_gb:.2f}GB, {process_memory_percent:.1f}% of total)")
        print(f"   Available memory: {available_memory_gb:.2f}GB of {total_memory_gb:.2f}GB total")
        print(f"   Forcing graceful exit before OOM killer terminates the process...")

        # Save checkpoint data before exiting
        try:
            checkpoint_data = load_checkpoint() or {}
            checkpoint_data["forced_exit"] = {
                "timestamp": time.time(),
                "memory_percent": memory.percent,
                "process_memory_gb": process_memory_gb,
                "process_memory_percent": process_memory_percent,
                "total_memory_gb": total_memory_gb,
                "available_memory_gb": available_memory_gb
            }
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint_data, f)
            print(f"   Checkpoint saved before exit")
        except Exception as e:
            print(f"   Failed to save checkpoint: {str(e)}")

        # Log the event
        try:
            with open("oom_prevention.log", "a") as f:
                f.write(f"\n--- OOM Prevention Exit at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                f.write(f"System memory: {memory.percent:.1f}% used\n")
                f.write(f"Process memory: {process_memory_gb:.2f}GB ({process_memory_percent:.1f}% of total)\n")
                f.write(f"Total memory: {total_memory_gb:.2f}GB\n")
                f.write(f"Available memory: {available_memory_gb:.2f}GB\n")
                f.write(f"Forcing exit to prevent OOM killer\n")
        except:
            pass

        # Force exit with a non-zero status code
        print("💥 Exiting to prevent OOM killer termination. Restart the script to continue from checkpoint.")
        os._exit(1)  # Force immediate exit

def calculate_batch_size(items_count: int) -> int:
    """Calculate appropriate batch size based on collection size and system resources.
    Optimized for Tesla P40 GPU with 24GB memory."""
    # Start with a higher default for Tesla P40
    batch_size = 16

    # Adjust based on collection size
    if items_count > 10000:
        batch_size = 12
    elif items_count > 1000:
        batch_size = 14

    # Check system memory conditions
    memory = psutil.virtual_memory()

    # Check GPU memory if available
    gpu_memory_available = False
    if torch.cuda.is_available():
        try:
            # Get GPU memory info for Tesla P40
            allocated_memory = torch.cuda.memory_allocated(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_percent = (allocated_memory / total_memory) * 100
            gpu_memory_available = True

            # Tesla P40 specific batch size adjustments
            if gpu_memory_percent > 90:
                batch_size = max(4, batch_size // 2)
                print(f"   ⚠️ GPU memory usage high ({gpu_memory_percent:.1f}%), reducing batch size to {batch_size}")
            elif gpu_memory_percent < 30 and items_count > 100:
                # If GPU has plenty of memory, we can increase batch size
                batch_size = min(24, batch_size + 4)
                print(f"   ℹ️ GPU memory usage low ({gpu_memory_percent:.1f}%), increasing batch size to {batch_size}")
        except Exception as e:
            print(f"   ⚠️ Error checking GPU memory: {e}")

    # If system memory usage is very high, reduce batch size
    if memory.percent > 60:
        batch_size = max(4, batch_size // 2)
        print(f"   ⚠️ System memory usage high ({memory.percent:.1f}%), reducing batch size to {batch_size}")

    # Consider process memory usage as well
    process = psutil.Process(os.getpid())
    process_memory_gb = process.memory_info().rss / (1024 ** 3)
    total_memory_gb = memory.total / (1024 ** 3)

    # Calculate process memory as percentage of total system memory
    process_memory_percent = (process_memory_gb / total_memory_gb) * 100

    # If process is using more than 2GB or more than 15% of total memory, reduce batch size
    # Higher thresholds for Tesla P40 system
    if process_memory_gb > 2.0 or process_memory_percent > 15:
        batch_size = max(4, batch_size // 2)

    # Check available memory in GB
    available_memory_gb = memory.available / (1024 ** 3)

    # Log memory status with GPU info if available
    if gpu_memory_available:
        print(f"   Memory status: System {memory.percent:.1f}%, GPU {gpu_memory_percent:.1f}%, Available RAM {available_memory_gb:.2f}GB, Process {process_memory_gb:.2f}GB ({process_memory_percent:.1f}% of total)")
    else:
        print(f"   Memory status: System {memory.percent:.1f}%, Available RAM {available_memory_gb:.2f}GB, Process {process_memory_gb:.2f}GB ({process_memory_percent:.1f}% of total)")

    # If less than 4GB available, reduce batch size further
    if available_memory_gb < 4.0:
        batch_size = max(2, batch_size // 2)
        print(f"   ⚠️ Low available memory ({available_memory_gb:.2f}GB), reducing batch size to {batch_size}")

    return batch_size

def load_checkpoint():
    """Load checkpoint data from file if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    return None

def save_checkpoint(collection_name: str, file_path: str, processed_count: int, total_count: int):
    """Save checkpoint data to file."""
    try:
        checkpoint_data = load_checkpoint() or {}
        checkpoint_data[collection_name] = {
            "file_path": file_path,
            "processed_count": processed_count,
            "total_count": total_count,
            "timestamp": time.time()
        }
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def memory_monitor_thread_func(stop_event, collection_name):
    """Thread function to monitor memory usage during processing.
    Optimized for Tesla P40 GPU monitoring and management."""
    # Track GPU memory stats over time for Tesla P40
    gpu_memory_history = []
    history_size = 5  # Keep track of last 5 measurements

    while not stop_event.is_set():
        # Monitor system RAM
        memory = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        process_memory_gb = process.memory_info().rss / (1024 ** 3)
        total_memory_gb = memory.total / (1024 ** 3)

        # Calculate process memory as percentage of total system memory
        process_memory_percent = (process_memory_gb / total_memory_gb) * 100

        # Monitor Tesla P40 GPU memory if available
        gpu_memory_percent = 0
        gpu_memory_critical = False
        if torch.cuda.is_available():
            try:
                # Get Tesla P40 memory stats
                allocated_memory = torch.cuda.memory_allocated(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory
                free_memory = total_memory - allocated_memory

                # Calculate usage percentages
                gpu_memory_percent = (allocated_memory / total_memory) * 100
                gpu_memory_history.append(gpu_memory_percent)

                # Keep history at fixed size
                if len(gpu_memory_history) > history_size:
                    gpu_memory_history.pop(0)

                # Check for memory leaks or rapid growth in Tesla P40
                if len(gpu_memory_history) >= 3:
                    # If memory usage is consistently increasing rapidly
                    if all(gpu_memory_history[i] < gpu_memory_history[i+1] for i in range(len(gpu_memory_history)-1)):
                        growth_rate = gpu_memory_history[-1] - gpu_memory_history[0]
                        if growth_rate > 10:  # More than 10% growth over last measurements
                            gpu_memory_critical = True
                            print(f"\n⚠️ Tesla P40 GPU memory growing rapidly: {growth_rate:.1f}% increase detected")

                # Critical if over 85% for Tesla P40
                if gpu_memory_percent > 85:
                    gpu_memory_critical = True
            except Exception as e:
                print(f"   ⚠️ Error monitoring Tesla P40 GPU memory: {e}")

        # If memory usage is high, perform cleanup
        if memory.percent > CRITICAL_MEMORY_PERCENT or process_memory_percent > 50 or gpu_memory_critical:
            print(f"\n⚠️ High resource usage detected: System RAM {memory.percent:.1f}%, Process {process_memory_gb:.2f}GB ({process_memory_percent:.1f}%)")
            if torch.cuda.is_available():
                print(f"   Tesla P40 GPU memory: {gpu_memory_percent:.1f}%")
            print(f"   Performing emergency cleanup for collection {collection_name}...")

            # Tesla P40 optimized cleanup sequence
            # Force garbage collection first
            gc.collect(generation=2)

            # Tesla P40 specific GPU cleanup
            if torch.cuda.is_available():
                try:
                    # Synchronize CUDA operations before emptying cache
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    # Reset peak memory stats for Tesla P40
                    if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                        torch.cuda.reset_peak_memory_stats()

                    # For severe GPU memory issues, try more aggressive approach
                    if gpu_memory_critical:
                        print("   Performing aggressive Tesla P40 memory cleanup...")
                        # Release any cached tensors
                        torch.cuda.empty_cache()
                        # Force synchronization again
                        torch.cuda.synchronize()
                except Exception as e:
                    print(f"   ⚠️ Error during Tesla P40 GPU cleanup: {e}")

            # Try to compact system memory if possible
            try:
                import ctypes
                libc = ctypes.CDLL('libc.so.6')
                libc.malloc_trim(0)
            except:
                pass

        # Sleep before next check - adaptive sleep based on memory pressure
        # Tesla P40 specific monitoring intervals
        if gpu_memory_critical:
            # Check very frequently if GPU memory is critical
            time.sleep(MEMORY_CHECK_INTERVAL / 4)
        elif memory.percent > CRITICAL_MEMORY_PERCENT or gpu_memory_percent > 80:
            # Check more frequently under high pressure
            time.sleep(MEMORY_CHECK_INTERVAL / 2)
        elif gpu_memory_percent > 60 or memory.percent > 70:
            # Moderate pressure
            time.sleep(MEMORY_CHECK_INTERVAL * 0.75)
        else:
            # Normal monitoring interval
            time.sleep(MEMORY_CHECK_INTERVAL)

def remove_lock():
    # Delete the lock file after all processing is complete
    try:
        lock_file_path = os.path.abspath("docker_qdrant/.lock")
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)
            print(f"Lock file {lock_file_path} has been deleted.")
        else:
            print(f"Lock file {lock_file_path} does not exist.")
    except Exception as e:
        print(f"Error deleting lock file: {e}")

def process_directory(directory_path, collection_name):
    """
    Recursively process all files in a directory and its subdirectories.

    Args:
        directory_path: Path to the directory to process
        collection_name: Name of the collection to add the data to
    """
    subdir_files = sorted(os.listdir(directory_path))
    for subfile in subdir_files:
        subfile_path = os.path.join(directory_path, subfile)
        if os.path.isfile(subfile_path):
            process_file(subfile_path, collection_name)
        elif os.path.isdir(subfile_path):
            # Recursively process subdirectories
            process_directory(subfile_path, collection_name)

def process_file(file_path, collection_name):
    """
    Process a single file and add its content to the collection.

    Args:
        file_path: Path to the file to process
        collection_name: Name of the collection to add the data to
    """
    # Check if we have a checkpoint for this file
    checkpoint_data = load_checkpoint()
    if checkpoint_data and collection_name in checkpoint_data:
        checkpoint_info = checkpoint_data[collection_name]
        if checkpoint_info.get("file_path") == file_path:
            print(f"   📋 Resuming processing of {os.path.basename(file_path)} from checkpoint")
            # If the file was already fully processed, we can skip it
            if checkpoint_info.get("processed_count") == checkpoint_info.get("total_count"):
                print(f"   ✅ File {os.path.basename(file_path)} was already fully processed")
                try:
                    os.remove(file_path)
                    return
                except Exception as e:
                    print(f"   ⚠️ Error deleting file '{os.path.basename(file_path)}': {e}")
                    return

    filename = os.path.basename(file_path)
    print(f"Processing file: {filename}")

    # Monitor system resources before processing
    monitor_system_resources()

    # Initialize variables
    text_documents = []
    content = filename  # Default source is the filename
    loader = None

    try:
        # Process file based on extension
        if filename.lower().endswith(".txt"):
            # Explicitly use UTF-8 encoding for Vietnamese content
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                text_documents = loader.load()

                # Get the first line as source for txt files
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.readline().strip()
            except Exception as e:
                print(f"   ❌ Error loading text file: {str(e)}")
                return

        elif filename.lower().endswith(".pdf"):
            # First try with PyPDFLoader
            try:
                loader = PyPDFLoader(file_path)
                text_documents = loader.load()
            except Exception as pdf_error:
                error_msg = str(pdf_error).lower()
                # If PyPDFLoader fails, try with UnstructuredPDFLoader as a fallback
                if "cryptography" in error_msg:
                    print(f"   ⚠️ Encrypted PDF detected: {filename}")
                    return
                elif "invalid pdf header" in error_msg or "eof marker not found" in error_msg:
                    print(f"   ⚠️ Invalid PDF format, trying alternative loader: {filename}")
                    try:
                        loader = UnstructuredPDFLoader(file_path)
                        text_documents = loader.load()
                    except Exception as unstructured_error:
                        print(f"   ❌ Failed to load PDF with alternative loader: {str(unstructured_error)}")
                        return
                else:
                    print(f"   ❌ Error loading PDF: {error_msg}")
                    return

        elif filename.lower().endswith(".docx"):
            try:
                # Use Docx2txtLoader for .docx files with UTF-8 encoding
                loader = Docx2txtLoader(file_path)
                text_documents = loader.load()
            except Exception as e:
                print(f"   ❌ Error loading DOCX file: {str(e)}")
                return

        elif filename.lower().endswith(".doc"):
            try:
                # Use UnstructuredWordDocumentLoader for .doc files with UTF-8 encoding
                loader = UnstructuredWordDocumentLoader(file_path)
                text_documents = loader.load()
            except Exception as e:
                print(f"   ❌ Error loading DOC file: {str(e)}")
                return

        elif filename.lower().endswith(".xlsx") or filename.lower().endswith(".xls"):
            try:
                # Use UnstructuredExcelLoader for Excel files
                loader = UnstructuredExcelLoader(file_path)
                text_documents = loader.load()
            except Exception as e:
                print(f"   ❌ Error loading Excel file: {str(e)}")
                return
        else:
            # Skip files that are not txt, pdf, doc, docx, xls, or xlsx
            print(f"   ⚠️ Unsupported file format: {filename}")
            return

        # Check if we got any documents
        if not text_documents:
            print(f"   ⚠️ No content extracted from {filename}")
            return

        print(f"   📄 Extracted {len(text_documents)} pages/sections from {filename}")

        # Create semantic chunker with optimized parameters
        chunker = SemanticChunker(
            min_sentences=2,
            max_sentences=20,
            similarity_threshold=0.3
        )

        # Process documents in batches to avoid memory issues
        all_documents = []

        # Create progress bar for chunking
        chunk_pbar = tqdm(total=len(text_documents), desc=f"Chunking {filename}", 
                          unit="pages", ncols=100, position=0, leave=True)

        # Process each document/page
        for doc in text_documents:
            # Monitor system resources
            monitor_system_resources()

            try:
                # Create semantic chunks
                chunks = chunker.create_semantic_chunks(doc.page_content)

                # Create Document objects
                metadata = {"source": content}
                documents = [Document(metadata=metadata, page_content=chunk) for chunk in chunks]

                all_documents.extend(documents)
                chunk_pbar.update(1)

                # Perform garbage collection after processing each document
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"   ⚠️ Error chunking document: {str(e)}")
                # Continue with next document
                chunk_pbar.update(1)
                continue

        chunk_pbar.close()

        # Save checkpoint before embedding and uploading
        save_checkpoint(collection_name, file_path, 0, len(all_documents))

        # Add documents to the collection
        if all_documents:
            print(f"   🔢 Created {len(all_documents)} chunks from {filename}")
            processed_count = chunked_metadata(all_documents, collection_name=collection_name)
            print(f"   ✅ Successfully processed {processed_count} chunks from {filename}")

            # Update checkpoint to mark file as fully processed
            save_checkpoint(collection_name, file_path, len(all_documents), len(all_documents))
        else:
            print(f"   ⚠️ No chunks created from {filename}")

        # Delete the file after successful processing
        try:
            os.remove(file_path)
            print(f"   🗑️ Deleted file: {filename}")
        except Exception as e:
            print(f"   ⚠️ Error deleting file '{filename}': {e}")

    except Exception as e:
        print(f"   ❌ Unexpected error processing file '{filename}': {str(e)}")
        # Save checkpoint with current progress
        if 'all_documents' in locals() and all_documents:
            save_checkpoint(collection_name, file_path, 0, len(all_documents))

    finally:
        # Clean up resources
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main function to process all folders in database as collections.
    Optimized for Tesla P40 GPU."""
    print("🚀 Starting knowledge base processing")

    # Print system information
    memory = psutil.virtual_memory()
    print(f"System memory: {memory.total / (1024**3):.2f}GB total, {memory.available / (1024**3):.2f}GB available ({memory.percent}% used)")

    # Tesla P40 specific initialization and logging
    if torch.cuda.is_available():
        # Verify we're using the Tesla P40 (GPU 1)
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print(f"GPU: {gpu_name} (Tesla P40), Memory: {total_memory:.2f}GB")
        print(f"CUDA Version: {torch.version.cuda}")

        # Optimize CUDA settings for Tesla P40
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Clear GPU memory at startup
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'reset_peak_memory_stats'):
            torch.cuda.reset_peak_memory_stats()

        print(f"Tesla P40 GPU optimizations enabled")
    else:
        print("GPU: Not available")

    # Remove lock file at start
    remove_lock()

    # Initialize resources
    try:
        # Initialize Qdrant client
        initialize_qdrant_client()

        # Initialize embeddings model
        initialize_embeddings()
    except Exception as e:
        print(f"❌ Error initializing resources: {str(e)}")
        return

    try:
        # Get base path
        base_path = os.path.abspath("database")

        # Check if base path exists
        if not os.path.exists(base_path):
            print(f"❌ Base path {base_path} does not exist")
            return

        # Get all folders (collections)
        folders = sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))])

        if not folders:
            print("⚠️ No collections found in database directory")
            return

        print(f"📚 Found {len(folders)} collections to process")

        # Process each collection
        for folder_idx, folder_name in enumerate(folders):
            folder_path = os.path.join(base_path, folder_name)

            # Get all files in this folder (including subdirectories)
            all_files = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    all_files.append(os.path.join(root, file))

            # Skip empty collections
            if not all_files:
                print(f"⚠️ Collection '{folder_name}' is empty, skipping")
                continue

            print(f"\n📁 Processing collection {folder_idx+1}/{len(folders)}: {folder_name} ({len(all_files)} files)")

            # Create collection if it doesn't exist
            try:
                create_collection(collection_name=folder_name)
                print(f"   ✅ Collection '{folder_name}' ready")
            except Exception as e:
                print(f"   ❌ Error creating collection '{folder_name}': {str(e)}")
                continue

            # Sort files alphabetically
            all_files.sort()

            # Process files with progress tracking
            with tqdm(total=len(all_files), desc=f"Files in {folder_name}", 
                      unit="file", ncols=100, position=0, leave=True) as pbar:

                for file_idx, file_path in enumerate(all_files):
                    # Update progress bar description with current file
                    pbar.set_description(f"File {file_idx+1}/{len(all_files)}: {os.path.basename(file_path)}")

                    # Process file
                    try:
                        process_file(file_path, folder_name)
                    except Exception as e:
                        print(f"   ❌ Unexpected error processing file: {str(e)}")

                    # Update progress
                    pbar.update(1)

                    # Perform cleanup after each file
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            print(f"✅ Completed processing collection: {folder_name}")

        print("\n🎉 All collections processed successfully")

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")

    finally:
        # Always remove lock file at the end
        remove_lock()

        # Final cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("👋 Knowledge base processing completed")

# Run the main function
if __name__ == "__main__":
    main()
