import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
from src.dataloader import MVTecADTrainDataset
from src.models import get_feature_extractor

# --- Configuration ---
RESOLUTION = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEATURE_DIM = 1792 # 256 + 512 + 1024 channels from our model

def build_memory_bank(category="bottle"):
    """
    Extracts features from all 'good' training images and saves them
    in a FAISS index (our 'memory bank').
    """
    print(f"🚀 Starting memory bank build for '{category}' on {DEVICE}")
    
    # 1. Initialize Model and Dataloader
    model = get_feature_extractor().to(DEVICE)
    dataset = MVTecADTrainDataset("data/mvtec_ad", category, img_size=RESOLUTION)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    all_features = []
    print("Extracting features from 'good' images...")
    
    # 2. Extract Features
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(DEVICE)
            features = model(imgs) # Shape (B, 1792)
            all_features.append(features.cpu().numpy())

    # 3. Create FAISS Index (The Memory Bank)
    all_features = np.concatenate(all_features, axis=0).astype(np.float32)
    print(f"Building FAISS index (Memory Bank) with {all_features.shape[0]} vectors...")
    
    index = faiss.IndexFlatL2(FEATURE_DIM) # Use L2 (Euclidean) distance
    index.add(all_features) # Add all feature vectors to the index

    # 4. Save the memory bank
    os.makedirs("models", exist_ok=True)
    bank_path = f"models/{category}_memory_bank.index"
    
    print(f"Saving memory bank to {bank_path} ({index.ntotal} vectors)")
    faiss.write_index(index, bank_path)
    
    print("✅ Memory bank build complete.")

if __name__ == "__main__":
    build_memory_bank(category="bottle")
