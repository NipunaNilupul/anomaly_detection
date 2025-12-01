import time
import torch
import numpy as np
import faiss
from PIL import Image
import torchvision.transforms as T
from src.models import get_feature_extractor

# --- System Configuration ---
RESOLUTION = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAW_FEATURE_DIM = 176 # MobileNetV3
CATEGORY = "bottle"

class AnomalyDetector:
    def __init__(self):
        print("⚡ System Initializing...")
        
        # 1. Load Model
        self.model = get_feature_extractor().to(DEVICE)
        self.model.eval()
        
        # 2. Load Memory Bank
        bank_path = f"models/{CATEGORY}_patch_memory_bank.index"
        print(f"   Loading Memory Bank: {bank_path}")
        self.index = faiss.read_index(bank_path)
        
        # 3. Load Optimized Threshold
        thresh_path = f"models/{CATEGORY}_threshold.txt"
        with open(thresh_path, "r") as f:
            self.threshold = float(f.read().strip())
        print(f"   Loaded Threshold (tau): {self.threshold:.4f}")
        
        # 4. Setup Transform
        self.transform = T.Compose([
            T.Resize((RESOLUTION, RESOLUTION)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 5. Warmup
        dummy = torch.randn(1, 3, RESOLUTION, RESOLUTION).to(DEVICE)
        with torch.no_grad(): self.model(dummy)
        print("✅ System Ready.")

    def predict(self, image_path):
        """
        End-to-End Inference for Industrial Control.
        Returns: (is_anomaly: bool, score: float, latency: float)
        """
        # Load Image
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(DEVICE)
        
        # Start Timer
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            # Extract
            patch_features = self.model(x)
            
            # Reshape
            B, C, H, W = patch_features.shape
            patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(-1, RAW_FEATURE_DIM).cpu().numpy()
            
            # Search
            D, _ = self.index.search(patch_vectors, 1)
            
            # Score
            score = np.max(D)
        
        # Stop Timer
        torch.cuda.synchronize()
        latency = (time.perf_counter() - t0) * 1000
        
        # Decision Logic (The "Bridge" to PLC)
        is_anomaly = score > self.threshold
        
        return is_anomaly, score, latency

def simulate_production_line():
    """
    Simulates a conveyor belt passing images to the system.
    """
    detector = AnomalyDetector()
    
    # Test images (One Good, One Defective)
    test_images = [
        "data/mvtec_ad/bottle/test/good/000.png",
        "data/mvtec_ad/bottle/test/broken_large/000.png"
    ]
    
    print("\n🏭 STARTING PRODUCTION LINE SIMULATION\n")
    
    for img_path in test_images:
        print(f"📸 Camera Input: {img_path}")
        
        is_defective, score, latency = detector.predict(img_path)
        
        # PLC Signal Logic
        status = "❌ FAIL (Defect)" if is_defective else "✅ PASS (Good)"
        signal = "SIGNAL_HIGH (Reject)" if is_defective else "SIGNAL_LOW (Accept)"
        
        print(f"   ⏱️  Inference Time: {latency:.2f} ms")
        print(f"   📊 Anomaly Score:  {score:.4f} (Threshold: {detector.threshold:.4f})")
        print(f"   🤖 PLC Output:     {signal}")
        print(f"   📝 Status:         {status}\n")
        
        time.sleep(1) # Simulate belt movement

if __name__ == "__main__":
    simulate_production_line()
