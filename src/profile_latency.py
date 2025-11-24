import time
import torch
import numpy as np
import faiss
from src.models import get_feature_extractor

# --- Configuration ---
RESOLUTION = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CATEGORY = "bottle"

def load_components():
    print("Loading model and memory bank...")
    model = get_feature_extractor().to(DEVICE)
    model.eval()
    
    bank_path = f"models/{CATEGORY}_patch_memory_bank.index"
    projector_path = f"models/{CATEGORY}_patch_projector.pth"
    
    index = faiss.read_index(bank_path)
    projector = torch.load(projector_path, weights_only=False)
    
    return model, index, projector

def profile_inference():
    model, index, projector = load_components()
    
    # Create a dummy input (representing a live camera frame)
    dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION).to(DEVICE)
    raw_feature_dim = 1792
    
    print(f"\n🚀 Starting Latency Profiling (Target: < 100ms)...")
    print("Warming up GPU...")
    # Warmup runs to wake up the GPU
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
            
    latencies = []
    iterations = 100
    
    print(f"Running {iterations} inference cycles...")
    
    for _ in range(iterations):
        # ⏱️ START TIMER
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # 1. Feature Extraction (ResNet)
            patch_features = model(dummy_input) # (1, 1792, 16, 16)
            
            # 2. Reshape
            H, W = patch_features.shape[2:]
            patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(-1, raw_feature_dim).cpu().numpy()
            
            # 3. Projection (Sparse)
            projected_vectors = projector.transform(patch_vectors).astype(np.float32)
            
            # 4. Search (FAISS)
            D, _ = index.search(projected_vectors, 1)
            
            # 5. Scoring (Max Distance)
            score = np.max(D)
            
        # ⏱️ END TIMER
        end_time = time.perf_counter()
        
        # Record time in milliseconds
        latencies.append((end_time - start_time) * 1000)

    # --- Statistics ---
    avg_latency = np.mean(latencies)
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    fps = 1000 / avg_latency
    
    print("\n📊 Latency Results:")
    print(f"   Average Latency: {avg_latency:.2f} ms")
    print(f"   Min Latency:     {min_latency:.2f} ms")
    print(f"   Max Latency:     {max_latency:.2f} ms")
    print(f"   Throughput:      {fps:.2f} FPS")
    
    if avg_latency < 100:
        print("\n✅ SUCCESS: System meets real-time requirement (< 100ms)")
    else:
        print("\n⚠️ WARNING: System is too slow (> 100ms). Optimization needed.")

if __name__ == "__main__":
    profile_inference()
