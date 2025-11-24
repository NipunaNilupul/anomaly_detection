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
# ⚡ OPTIMIZATION: ResNet-18 Dimensions
RAW_FEATURE_DIM = 448   # (64 + 128 + 256)
PROJECTED_DIM = 256     # Reduced dimension for extra speed

def build_memory_bank(category="bottle"):
    print(f"🚀 Starting memory bank build for '{category}' (ResNet-18) on {DEVICE}")
    
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
            
            B, C, H, W = patch_features.shape
            patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(B * H * W, C)
            all_patch_features.append(patch_vectors.cpu().numpy())

    all_patch_features = np.concatenate(all_patch_features, axis=0)
    print(f"Total patches: {all_patch_features.shape[0]}")
    
    print(f"Fitting projector (Dim: {RAW_FEATURE_DIM} -> {PROJECTED_DIM})...")
    projector = SparseRandomProjection(n_components=PROJECTED_DIM, eps=0.9, random_state=42)
    projector.fit(all_patch_features)

    print(f"Building FAISS index...")
    index = faiss.IndexFlatL2(PROJECTED_DIM)
    
    chunk_size = 50000
    for i in tqdm(range(0, all_patch_features.shape[0], chunk_size), desc="Indexing"):
        chunk = all_patch_features[i:i+chunk_size]
        projected_chunk = projector.transform(chunk).astype(np.float32)
        index.add(projected_chunk)

    os.makedirs("models", exist_ok=True)
    faiss.write_index(index, f"models/{category}_patch_memory_bank.index")
    torch.save(projector, f"models/{category}_patch_projector.pth")
    
    print("✅ Fast memory bank build complete.")

if __name__ == "__main__":
    build_memory_bank(category="bottle")
