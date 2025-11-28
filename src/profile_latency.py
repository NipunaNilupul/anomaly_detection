import time
import torch
import numpy as np
import faiss
from src.models import get_feature_extractor

# --- Configuration ---
# 128x128
RESOLUTION = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CATEGORY = "bottle"
MODEL_NAME = "tf_mobilenetv3_large_100"
RAW_FEATURE_DIM = 176 

def load_components():
    print(f"Loading {MODEL_NAME} and raw memory bank...")
    model = get_feature_extractor().to(DEVICE)
    model.eval()
    
    bank_path = f"models/{CATEGORY}_patch_memory_bank.index"
    index = faiss.read_index(bank_path)
    
    return model, index

def profile_inference():
    model, index = load_components()
    
    dummy_input = torch.randn(1, 3, RESOLUTION, RESOLUTION).to(DEVICE)
    
    print(f"\n🚀 Profiling Latency for MobileNetV3 (128x128) (Target: < 100ms)...")
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
            
            # 3. Search (Raw Vectors)
            D, _ = index.search(patch_vectors, 1)
            score = np.max(D)
            
        torch.cuda.synchronize() 
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000)

    avg_latency = np.mean(latencies)
    print(f"\n📊 Results (MobileNetV3 128x128):")
    print(f"   Average Latency: {avg_latency:.2f} ms")
    print(f"   Throughput:      {1000/avg_latency:.2f} FPS")
    
    if avg_latency < 100:
        print("\n✅ SUCCESS: Real-time requirement met (< 100ms)!")
    else:
        print("\n⚠️ WARNING: Still too slow.")

if __name__ == "__main__":
    profile_inference()
