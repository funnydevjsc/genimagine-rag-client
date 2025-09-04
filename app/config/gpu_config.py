"""
GPU configuration module.
This module handles GPU configuration and optimization.
"""
import gc

import torch


def configure_gpu():
    """
    Configure GPU settings and optimizations.
    Returns a dictionary with GPU information and availability status.
    """
    gpu_info = {
        "is_available": False,
        "device": "cpu",
        "name": None,
        "memory": None,
        "cuda_version": None,
        "pytorch_version": torch.__version__,
        "device_count": 0,
        "current_device": None
    }
    
    if torch.cuda.is_available():
        # Get GPU details
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Update GPU info
        gpu_info.update({
            "is_available": True,
            "device": "cuda:0",
            "name": gpu_name,
            "memory": f"{total_memory:.2f}GB",
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device()
        })

        # Tesla P40 specific optimizations

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

    else:
        # CPU fallback - no CUDA available
        pass

    # Force garbage collection at startup
    gc.collect()
    
    return gpu_info

def optimize_for_embeddings(gpu_info):
    """
    Apply optimizations specifically for embedding models.
    """
    if gpu_info["is_available"]:
        # Optimize for Tesla P40: Pre-allocate GPU memory if needed
        allocated_memory = torch.cuda.memory_allocated(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory = total_memory - allocated_memory
        total_free = total_memory - allocated_memory

        # If memory is fragmented (reserved but not used), clear cache
        if total_free - free_memory > 1 * 1024 * 1024 * 1024:  # 1GB difference
            torch.cuda.empty_cache()
            
        # Return optimized model kwargs
        return {
            'device': gpu_info["device"]
        }
    else:
        # CPU fallback
        return {
            'device': 'cpu'
        }