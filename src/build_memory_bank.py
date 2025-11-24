import os
import torch
import numpy as np
import faiss
from tqdm import tqdm
from sklearn.random_projection import SparseRandomProjection
from src.dataloader import MVTecADTrainDataset
from src.models import get_feature_extractor

# --- Configuration ---
RESOLUTION = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Original feature dim from our model
RAW_FEATURE_DIM = 1792 
# Dimension to reduce to for efficiency and performance
PROJECTED_DIM = 1024 

def build_memory_bank(category="bottle"):
    print(f"🚀 Starting PATCH memory bank build for '{category}' on {DEVICE}")
    
    model = get_feature_extractor().to(DEVICE)
    dataset = MVTecADTrainDataset("data/mvtec_ad", category, img_size=RESOLUTION)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True
    )

    all_patch_features = []
    print("Extracting patch features from 'good' images...")
    
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(DEVICE)
            patch_features = model(imgs) # Shape (B, 1792, 64, 64)
            
            # Reshape for projection: [B, C, H, W] -> [B*H*W, C]
            B, C, H, W = patch_features.shape
            # Each pixel's feature vector is now a row
            patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(B * H * W, C)
            
            all_patch_features.append(patch_vectors.cpu().numpy())

    # Fit Random Projector
    all_patch_features = np.concatenate(all_patch_features, axis=0)
    print(f"Total patches extracted: {all_patch_features.shape[0]}")
    
    print(f"Fitting projector (Dim: {RAW_FEATURE_DIM} -> {PROJECTED_DIM})...")
    projector = SparseRandomProjection(n_components=PROJECTED_DIM, eps=0.9, random_state=42)
    projector.fit(all_patch_features)

    # Create and Populate FAISS Index
    print(f"Building FAISS index (Memory Bank)...")
    index = faiss.IndexFlatL2(PROJECTED_DIM)
    
    # Process in chunks to save RAM during projection/addition
    chunk_size = 50000
    for i in tqdm(range(0, all_patch_features.shape[0], chunk_size), desc="Projecting/Adding"):
        chunk = all_patch_features[i:i+chunk_size]
        projected_chunk = projector.transform(chunk).astype(np.float32)
        index.add(projected_chunk)

    # Save
    os.makedirs("models", exist_ok=True)
    bank_path = f"models/{category}_patch_memory_bank.index"
    projector_path = f"models/{category}_patch_projector.pth"
    
    print(f"Saving patch memory bank to {bank_path}")
    faiss.write_index(index, bank_path)
    
    print(f"Saving projector to {projector_path}")
    torch.save(projector, projector_path)
    
    print("✅ Patch memory bank build complete.")

if __name__ == "__main__":
    build_memory_bank(category="bottle")
