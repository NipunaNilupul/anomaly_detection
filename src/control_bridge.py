import os  # Used to check if model files exist on the disk
import time  # Used for measuring the inference latency (speed) of the system
import torch  # The core Deep Learning framework
import numpy as np  # Used for numerical operations on the score arrays
import faiss  # The library used for the high-speed Nearest Neighbor search (Memory Bank)
from PIL import Image  # Used to load images from disk
import torchvision.transforms as T  # Used to preprocess images (resize/normalize) before the model
from src.models import get_feature_extractor  # Imports your custom MobileNetV3 wrapper

# --- System Configuration ---
RESOLUTION = 128  # Fixed resolution (128x128) to ensure <40ms speed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto-selects GPU if available
RAW_FEATURE_DIM = 176  # The size of the feature vector output by MobileNetV3 (Layers 1-3)
CATEGORY = "bottle"  # The specific product line currently being inspected

class AnomalyDetector:
    def __init__(self):
        print(" System Initializing...")  # User feedback log
        
        # 1. Load Model
        # Initializes the lightweight feature extractor and moves it to GPU for speed
        self.model = get_feature_extractor().to(DEVICE)
        self.model.eval()  # Sets model to evaluation mode (freezes BatchNorm/Dropout layers)
        
        # 2. Load Memory Bank (FAISS Index)
        bank_path = f"models/{CATEGORY}_patch_memory_bank.index"  # Path to the trained memory bank
        
        # Safety check: ensures the memory bank exists before trying to load it
        if not os.path.exists(bank_path):
             raise FileNotFoundError(f"Memory bank missing: {bank_path}")
             
        print(f"   Loading Memory Bank: {bank_path}")
        self.index = faiss.read_index(bank_path)  # Loads the FAISS index from disk into RAM
        
        # 3. Load Optimized Threshold (tau)
        thresh_path = f"models/{CATEGORY}_threshold.txt"  # Path to the threshold calculated during training
        with open(thresh_path, "r") as f:
            self.threshold = float(f.read().strip())  # Reads the threshold value
        print(f"   Loaded Threshold (tau): {self.threshold:.4f}")
        
        # 4. Setup Transform
        # Defines the exact preprocessing pipeline used during training to ensure consistency
        self.transform = T.Compose([
            T.Resize((RESOLUTION, RESOLUTION)),  # Resizes to 128x128
            T.ToTensor(),  # Converts image to PyTorch Tensor (0-1 float)
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizes using ImageNet stats
        ])
        
        # 5. Warmup GPU (Critical for accurate first-inference timing)
        # Creates a fake image to run through the model once.
        # This forces CUDA to initialize its kernels so the first real prediction isn't slow.
        dummy = torch.randn(1, 3, RESOLUTION, RESOLUTION).to(DEVICE)
        with torch.no_grad(): self.model(dummy)
        print(" System Ready.")  # Confirms the system is fully loaded and ready

    def predict(self, image_path):
        """
        End-to-End Inference for Industrial Control.
        Returns: (is_anomaly: bool, score: float, latency: float)
        """
        # Load Image from disk and convert to RGB to handle grayscale/alpha inputs correctly
        image = Image.open(image_path).convert("RGB")
        # Transform image and add a batch dimension [1, 3, 128, 128] for the model
        x = self.transform(image).unsqueeze(0).to(DEVICE)
        
        # Start Timer
        torch.cuda.synchronize()  # Waits for all previous GPU tasks to finish for accurate timing
        t0 = time.perf_counter()  # Records the start time with high precision
        
        with torch.no_grad():  # Disables gradient calculation to save memory and speed up inference
            # Extract Features: Passes image through MobileNetV3
            patch_features = self.model(x)
            
            # Reshape [B, C, H, W] -> [N_Patches, C]
            # Flattens the spatial map so each patch becomes a separate 176-d vector
            B, C, H, W = patch_features.shape
            patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(-1, RAW_FEATURE_DIM).cpu().numpy()
            
            # Search Memory Bank: Finds the nearest neighbor for every patch
            D, _ = self.index.search(patch_vectors, 1)
            
            # Score = Max distance of any patch
            # If even one patch is far from the memory bank (high distance), the whole image is anomalous
            score = np.max(D)
        
        # Stop Timer
        torch.cuda.synchronize()  # Waits for the GPU to finish the search
        latency = (time.perf_counter() - t0) * 1000  # Calculates elapsed time in milliseconds
        
        # Decision Logic (The "Bridge" to PLC)
        # Compares the score against the pre-calculated threshold (tau)
        is_anomaly = score > self.threshold
        
        return is_anomaly, score, latency

    def send_signal_to_plc(self, signal_type):
        """
        Mock hardware interface.
        In a real deployment, this would use a library like 'pymodbus' or 'RPi.GPIO'.
        """
        if signal_type == "REJECT":
            # Simulator for sending a 24V signal to a pneumatic air-jet to reject the bottle
            print("    [HARDWARE] GPIO_18 HIGH -> Triggering Reject Mechanism (Air Jet)")
        else:
            # Simulator for keeping the line running
            print("    [HARDWARE] GPIO_18 LOW  -> Conveyor Continue")

def simulate_production_line():
    """
    Simulates a conveyor belt passing images to the system.
    """
    detector = AnomalyDetector()  # Instantiate the detector (loads model/index once)
    
    # List of images representing items coming down the conveyor belt
    test_images = [
        "data/mvtec_ad/bottle/test/good/000.png",
        "data/mvtec_ad/bottle/test/broken_large/000.png",
        "data/mvtec_ad/bottle/test/good/001.png",
        "data/mvtec_ad/bottle/test/broken_large/019.png"
    ]
    
    print("\n STARTING PRODUCTION LINE SIMULATION\n")
    
    # Loop through each image, simulating the camera trigger
    for img_path in test_images:
        print(f" Camera Input: {img_path}")
        
        # 1. Run Inference
        # Gets the decision, score, and speed from the AI
        is_defective, score, latency = detector.predict(img_path)
        
        # 2. Determine Signal for the Operator
        status = " FAIL (Defect)" if is_defective else " PASS (Good)"
        signal = "REJECT" if is_defective else "ACCEPT"
        
        # 3. Log Results to Console
        print(f"    Inference Time: {latency:.2f} ms")
        print(f"   Anomaly Score:  {score:.4f} (Threshold: {detector.threshold:.4f})")
        print(f"    Decision:        {status}")
        
        # 4. Trigger Hardware
        # Sends the physical signal to the machine
        detector.send_signal_to_plc(signal)
        print("-" * 50)  # Separator for readability
        
        time.sleep(1) # Simulates the time gap between bottles on the belt

if __name__ == "__main__":
    simulate_production_line()  # Runs the simulation when script is executed
