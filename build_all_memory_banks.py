import os  # Standard library for filesystem operations (creating folders, checking paths)
import torch  # PyTorch deep learning framework for tensor operations and GPU support
import faiss  # Facebook AI Similarity Search: The core library for the nearest-neighbor 'memory bank'
import numpy as np  # Library for handling large arrays of feature vectors
from PIL import Image  # Library for loading images from disk
from pathlib import Path  # Object-oriented path handling for cleaner file management
from tqdm import tqdm  # Progress bar library to visualize training status
import torchvision.transforms as T  # PyTorch tools for resizing and normalizing images
from src.models import get_feature_extractor  # Imports your custom MobileNetV3 backbone wrapper

# --- Configuration (Must match evaluation!) ---
# Resolution 128x128 matches the evaluation and inference steps. 
# Consistency is critical: features extracted at 256px will NOT match features at 128px.
RESOLUTION = 128
# Automatically detect GPU for acceleration. Feature extraction is much faster on CUDA.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Standard ImageNet normalization. MobileNetV3 expects this specific distribution (mean/std).
IMAGENET_NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# List of all 15 MVTec AD Categories to process
MVTEC_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 
    'leather', 'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 
    'transistor', 'wood', 'zipper'
]

def build_memory_bank(category, model):
    print(f"\nBuilding Memory Bank for '{category}'...")
    
    # Setup paths
    # Defines the location of the 'good' training images for the current category
    train_dir = Path("data/mvtec_ad") / category / "train" / "good"
    # Defines where the resulting FAISS index file will be saved
    save_path = f"models/{category}_patch_memory_bank.index"
    
    # Ensure directory exists
    os.makedirs("models", exist_ok=True)  # Creates the 'models' folder if it's missing
    
    # Transform (Critical: Must be 128x128 like evaluation)
    # Defines the preprocessing pipeline applied to every training image
    transform = T.Compose([
        T.Resize((RESOLUTION, RESOLUTION)),  # Resizes to 128x128
        T.ToTensor(),  # Converts to PyTorch Tensor (0-1 float)
        IMAGENET_NORM  # Applies ImageNet normalization stats
    ])
    
    features_list = []  # Initialize an empty list to collect feature vectors
    
    # 1. Extract Features from Training Data
    # Get a sorted list of all PNG images in the training folder
    image_files = sorted(train_dir.glob("*.png"))
    
    # Error handling: If the folder is empty or path is wrong, skip this category
    if not image_files:
        print(f" Error: No training images found in {train_dir}")
        return

    # Loop through each training image with a progress bar
    for img_path in tqdm(image_files, desc="Extracting features"):
        image = Image.open(img_path).convert("RGB")  # Load image and ensure 3 channels (RGB)
        x = transform(image).unsqueeze(0).to(DEVICE)  # Preprocess & add batch dimension [1, 3, 128, 128]
        
        with torch.no_grad():  # Disable gradient calculation (we are only extracting, not training)
            patch_features = model(x) # Pass image through MobileNetV3 to get feature maps
            
            # Reshape: (B, C, H, W) -> (N_Patches, C)
            # MobileNetV3 layer 1-3 concat dimension is 176
            B, C, H, W = patch_features.shape
            # Permute moves channels to the end, reshape flattens the spatial grid into a list of vectors
            patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(-1, C)
            
            # Move vectors to CPU and convert to NumPy for FAISS storage
            features_list.append(patch_vectors.cpu().numpy())
    
    # 2. Stack and Index
    # Safety check: If no features were extracted, exit
    if not features_list:
        return

    # Combine the list of arrays into one massive NumPy array (Total_Patches x 176)
    all_features = np.concatenate(features_list, axis=0)
    
    # Initialize FAISS Index (L2 Distance)
    # 'IndexFlatL2' is a brute-force search index using Euclidean distance
    d = all_features.shape[1] # Dimension (176)
    index = faiss.IndexFlatL2(d)
    
    # Add features to index
    # This stores the vectors in the FAISS structure, ready for searching
    index.add(all_features)
    
    # 3. Save
    # Write the built index to disk so we can load it later during inference
    faiss.write_index(index, save_path)
    print(f" Saved index to {save_path} ({index.ntotal} patches)")

if __name__ == "__main__":
    # Load model once
    # We load the MobileNetV3 backbone only once and reuse it for all categories to save time
    print("Loading MobileNetV3 backbone...")
    model = get_feature_extractor().to(DEVICE)
    model.eval() # Set to evaluation mode
    
    # Loop through all categories
    # Automates the process for the entire dataset
    for cat in MVTEC_CATEGORIES:
        build_memory_bank(cat, model)
        
    print("\n DONE. All memory banks built!")
