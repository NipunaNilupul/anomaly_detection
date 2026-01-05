import os  # Standard library for interacting with the operating system (e.g., creating directories)
import torch  # The main PyTorch library used for deep learning operations and tensor management
import numpy as np  # Library for efficient numerical array manipulation
import faiss  # Facebook AI Similarity Search library, used here to create the 'Memory Bank'
from tqdm import tqdm  # Library to display progress bars during long loops
from src.dataloader import MVTecADTrainDataset  # Imports your custom dataset loader for MVTec AD
from src.models import get_feature_extractor  # Imports your custom function to load the MobileNetV3 model

# --- Configuration ---
# resolution 128x128 for 4x speedup
RESOLUTION = 128  # Sets the input image height/width to 128px to significantly reduce inference time
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Automatically selects GPU if available for faster processing, otherwise falls back to CPU
RAW_FEATURE_DIM = 176  # Defines the vector size (176) which matches the output of MobileNetV3 layers 1, 2, and 3 concatenated

def build_memory_bank(category="bottle"):
    # Prints a status message confirming the category and device being used
    print(f" Starting RAW memory bank build for '{category}' (128x128) on {DEVICE}")
    
    # Initializes the MobileNetV3 feature extractor and moves it to the active device (GPU/CPU)
    model = get_feature_extractor().to(DEVICE)
    
    # Loads the training dataset (containing only 'good' images) for the specific category
    dataset = MVTecADTrainDataset("data/mvtec_ad", category, img_size=RESOLUTION)
    
    # Creates a DataLoader to handle batching (16 images at a time) and parallel loading (4 workers)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    all_patch_features = []  # Initializes an empty list to collect feature vectors from all batches
    print("Extracting patch features...")  # user feedback
    
    # Disables gradient calculation since we are only extracting features (saves memory and computation)
    with torch.no_grad():
        # Iterates through the dataloader batch by batch with a progress bar
        for imgs, _ in tqdm(dataloader, desc="Extracting"):
            imgs = imgs.to(DEVICE)  # Moves the current batch of images to the GPU
            patch_features = model(imgs)  # Passes images through the model to get the feature maps
            
            # Reshape: [B, C, H, W] -> [B*H*W, C]
            B, C, H, W = patch_features.shape  # Unpacks the dimensions: Batch, Channels, Height, Width
            
            # 1. Permute: Rearranges dimensions to [Batch, Height, Width, Channels] so channels are last
            # 2. Reshape: Flattens the spatial dimensions so every pixel location becomes a separate feature vector
            patch_vectors = patch_features.permute(0, 2, 3, 1).reshape(B * H * W, C)
            
            # Moves the processed vectors back to CPU, converts to NumPy, and adds to our list
            all_patch_features.append(patch_vectors.cpu().numpy())

    # Combines all batch lists into one massive NumPy array and ensures float32 precision (required by FAISS)
    all_patch_features = np.concatenate(all_patch_features, axis=0).astype(np.float32)
    print(f"Total patches: {all_patch_features.shape[0]}")  # Prints total count of features extracted
    
    print(f"Building FAISS index (Dim: {RAW_FEATURE_DIM})...")  # User feedback
    
    # Initializes a brute-force L2 (Euclidean) distance index for similarity search
    index = faiss.IndexFlatL2(RAW_FEATURE_DIM)
    
    chunk_size = 50000  # Sets a limit on how many vectors to add at once to prevent memory crashes
    
    # Iterates through the massive feature array in chunks of 50,000
    for i in tqdm(range(0, all_patch_features.shape[0], chunk_size), desc="Indexing"):
        chunk = all_patch_features[i:i+chunk_size]  # Slices the current chunk of data
        index.add(chunk)  # Adds the chunk to the FAISS index structure

    os.makedirs("models", exist_ok=True)  # Creates a 'models' directory if it doesn't already exist
    
    # Saves the fully built FAISS index to a file on the disk
    faiss.write_index(index, f"models/{category}_patch_memory_bank.index")
    
    print(" Memory bank build complete.")  # Final success message

if __name__ == "__main__":
    # Entry point: calls the function to build the memory bank for the 'bottle' category
    build_memory_bank(category="bottle")
