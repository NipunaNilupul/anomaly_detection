import time
import torch
import numpy as np
import faiss
from src.models import get_feature_extractor

# --- Configuration ---
RESOLUTION = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CATEGORY = "bottle"
# ⚡ OPTIMIZATION: ResNet-18 Dimensions
RAW_FEATURE_DIM = 448 

def load_components():
    print(f"Loading ResNet-18 and memory bank...")
    model = get_feature_extractor().to(DEVICE)
    model.eval()
    
    bank_path = f"models/{CATEGORY}_patch_memory_bank.index"
    projector_path = f"models/{CATEGORY}_patch_projector.pth"
    
    index = faiss.read_index(bank_path)
    projector = torch.load(projector_path, weights_only=False)
    
    return model, index, projector

def profile_inference():
    model, index, projector = load_components()
    
    dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION).to(DEVICE)
    
    print(f"\n🚀 Profiling Latency for ResNet-18 (Target: < 100ms)...")
    print("Warming up GPU...")
    for _ in range(10):
        with torch.no_grad(): _ = model(dummy_input)
            
    latencies = []
    iterations = 100
    
    print(f"Running {iterations} cycles...")
    for _ in range(iterations):
        torch.cuda.synchronize() 
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # 1. Model Inference
            patch_features = model(dummy_input)
            
            # 2. Reshape
            H, W = patch_features.shape[2:]
            patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(-1, RAW_FEATURE_DIM).cpu().numpy()
            
            # 3. Projection
            projected_vectors = projector.transform(patch_vectors).astype(np.float32)
            
            # 4. Search
            D, _ = index.search(projected_vectors, 1)
            score = np.max(D)
            
        torch.cuda.synchronize() 
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)

    avg_latency = np.mean(latencies)
    print(f"\n📊 Results (ResNet-18):")
    print(f"   Average Latency: {avg_latency:.2f} ms")
    print(f"   Throughput:      {1000/avg_latency:.2f} FPS")
    
    if avg_latency < 100:
        print("\n✅ SUCCESS: Real-time requirement met (< 100ms)!")
    else:
        print("\n⚠️ WARNING: Still too slow.")

if __name__ == "__main__":
    profile_inference()
