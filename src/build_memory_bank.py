import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
from src.dataloader import MVTecADTrainDataset
from src.models import get_feature_extractor

# --- Configuration ---
# ⚡ OPTIMIZATION: Reduce resolution to 128x128 for 4x speedup
RESOLUTION = 128 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAW_FEATURE_DIM = 176 # MobileNetV3 Layers 1+2+3

def build_memory_bank(category="bottle"):
    print(f"🚀 Starting RAW memory bank build for '{category}' (128x128) on {DEVICE}")
    
    model = get_feature_extractor().to(DEVICE)
    dataset = MVTecADTrainDataset("data/mvtec_ad", category, img_size=RESOLUTION)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    all_patch_features = []
    print("Extracting patch features...")
    
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(DEVICE)
            patch_features = model(imgs) 
            
            # Reshape: [B, C, H, W] -> [B*H*W, C]
            B, C, H, W = patch_features.shape
            patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(B * H * W, C)
            all_patch_features.append(patch_vectors.cpu().numpy())

    all_patch_features = np.concatenate(all_patch_features, axis=0).astype(np.float32)
    print(f"Total patches: {all_patch_features.shape[0]}")
    
    print(f"Building FAISS index (Dim: {RAW_FEATURE_DIM})...")
    index = faiss.IndexFlatL2(RAW_FEATURE_DIM)
    
    chunk_size = 50000
    for i in tqdm(range(0, all_patch_features.shape[0], chunk_size), desc="Indexing"):
        chunk = all_patch_features[i:i+chunk_size]
        index.add(chunk) 

    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, f"models/{category}_patch_memory_bank.index")
    
    print("✅ Memory bank build complete.")

if __name__ == "__main__":
    build_memory_bank(category="bottle")
