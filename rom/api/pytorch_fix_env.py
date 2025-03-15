# Add this to the beginning of your main.py file
import os
import sys

# Set environment variables to control PyTorch and ONNX behavior
os.environ["PYTORCH_JIT"] = "0"  # Disable JIT compilation
os.environ["TORCH_USE_RTLD_GLOBAL"] = "YES"  # Make PyTorch symbols globally available
os.environ["OMP_NUM_THREADS"] = "1"  # Limit OpenMP threads
os.environ["ORT_DISABLE_COREML"] = "1"  # Disable CoreML backend in ONNX Runtime

# Force CPU mode for ONNX
os.environ["ONNXRUNTIME_BACKEND"] = "CPU"

# Try to preload required PyTorch modules
try:
    import torch
    import torch.nn as nn
    # Force eager mode
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    print("PyTorch initialized successfully in eager mode")
except Exception as e:
    print(f"Warning: Could not initialize PyTorch: {e}")