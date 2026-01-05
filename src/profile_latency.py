import time  # For measuring high-precision execution time
import torch  # The core deep learning library
import numpy as np  # For statistical calculations (mean latency)
import faiss  # The high-speed search library for the memory bank
from src.models import get_feature_extractor  # Imports your custom MobileNetV3 wrapper

# --- Configuration ---
# 128x128 resolution is chosen to minimize computational load (FLOPs)
RESOLUTION = 128
# Automatically detect GPU (CUDA) for acceleration; fallback to CPU if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CATEGORY = "bottle"  # Product category to profile
MODEL_NAME = "tf_mobilenetv3_large_100"  # Specific lightweight model architecture
RAW_FEATURE_DIM = 176  # The fixed size of the feature vector (Layers 1+2+3)

def load_components():
    """
    Loads the Model and Memory Bank into RAM/VRAM.
    This happens once at startup to avoid re-loading overhead during profiling.
    """
    print(f"Loading {MODEL_NAME} and raw memory bank...")
    # Initialize model and move to GPU
    model = get_feature_extractor().to(DEVICE)
    model.eval() # Set to evaluation mode (optimization)
    
    # Load the pre-built FAISS index from disk
    bank_path = f"models/{CATEGORY}_patch_memory_bank.index"
    index = faiss.read_index(bank_path)
    
    return model, index

def profile_inference():
    # 1. Setup
    model, index = load_components()
    
    # Create a 'Dummy Input' (Random Noise) of the correct shape [1, 3, 128, 128]
    # This simulates a camera image without the I/O overhead of loading a file from disk
    dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION).to(DEVICE)
    
    print(f"\n Profiling Latency for MobileNetV3 (128x128) (Target: < 100ms)...")
    
    # 2. Warmup Phase (CRITICAL)
    # The first few passes on a GPU are always slow due to kernel initialization and memory allocation.
    # We run 10 dummy passes to "warm up" the GPU so our measurements reflect stable performance.
    print("Warming up GPU...")
    for _ in range(10):
        with torch.no_grad(): _ = model(dummy_input)
            
    latencies = []
    iterations = 100 # Run 100 times to get a statistically significant average
    
    print(f"Running {iterations} cycles...")
    for _ in range(iterations):
        # 3. Start Timer
        # torch.cuda.synchronize() ensures all previous GPU tasks are done before we start the clock.
        # This prevents measuring background noise.
        torch.cuda.synchronize() 
        start_time = time.perf_counter() # Use perf_counter for microsecond precision
        
        with torch.no_grad(): # Disable gradient calculation to simulate real inference
            # A. Model Inference: Forward pass through MobileNetV3
            patch_features = model(dummy_input)
            
            # B. Reshape: Flatten the spatial feature maps into vectors for FAISS
            # [1, 176, 8, 8] -> [64, 176]
            H, W = patch_features.shape[2:]
            patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(-1, RAW_FEATURE_DIM).cpu().numpy()
            
            # C. Search: Nearest Neighbor search in the Memory Bank
            D, _ = index.search(patch_vectors, 1)
            score = np.max(D) # The max distance is the anomaly score
            
        # 4. Stop Timer
        # Synchronize again to make sure the GPU actually finished the work before stopping the clock.
        torch.cuda.synchronize() 
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # Convert to milliseconds

    # 5. Report Statistics
    avg_latency = np.mean(latencies)
    print(f"\nResults (MobileNetV3 128x128):")
    print(f"   Average Latency: {avg_latency:.2f} ms")
    print(f"   Throughput:      {1000/avg_latency:.2f} FPS")
    
    # 6. Pass/Fail Check
    if avg_latency < 100:
        print("\nSUCCESS: Real-time requirement met (< 100ms)!")
    else:
        print("\n WARNING: Still too slow.")

if __name__ == "__main__":
    profile_inference()
