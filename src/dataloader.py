from pathlib import Path  # Modern library for handling file paths (works on Windows/Linux/Mac automatically)
from torch.utils.data import Dataset, DataLoader  # Base classes for creating efficient data pipelines in PyTorch
from PIL import Image  # Library to load images from disk
import torchvision.transforms as T  # Library for image preprocessing (resizing, normalizing)

class MVTecADTrainDataset(Dataset):
    """
    Custom Dataset class that Loads ONLY defect-free images (MVTec AD train/good).
    This is critical for Unsupervised Learning: the model must only see 'normal' data during training.
    """
    def __init__(self, root_dir: str, category: str, img_size: int = 256):
        self.root_dir = Path(root_dir)  # Converts string path to a Path object for easier manipulation
        self.category = category  # Stores the category name (e.g., "bottle")
        self.img_size = img_size  # Stores target resolution (e.g., 128 or 256)
        
        # Constructs the specific path to the 'good' training data
        # MVTec AD structure is: root/category/train/good
        self.good_dir = self.root_dir / category / "train" / "good"
        
        # Fail-Fast Check: Verifies the directory actually exists before trying to load anything
        if not self.good_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {self.good_dir}")
        
        # 1. Recursive Search (rglob): Finds files even in subfolders
        # 2. Filtering: Only keeps actual images (.png, .jpg), ignoring system files like .DS_Store
        # 3. Sorted: Sorts files alphabetically to ensure the order is deterministic (reproducible)
        self.image_paths = sorted(p for p in self.good_dir.rglob("*") 
                                 if p.suffix.lower() in [".png", ".jpg", ".jpeg"])
        
        # Check if directory was empty
        if not self.image_paths:
            raise ValueError(f"No images found in {self.good_dir}")
        
        # Define the Preprocessing Pipeline
        # This matches the training configuration of MobileNetV3 (ImageNet stats)
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),  # Resizes image to square resolution
            T.ToTensor(),  # Converts pixel values [0-255] to Tensor [0.0-1.0]
            # Normalizes using standard ImageNet mean/std. Critical for pre-trained models.
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # Returns the total number of images in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Open Image: Loads image from disk based on index
        # .convert("RGB") ensures 3 channels even if image is Grayscale (prevents crash on 'grid' or 'screw')
        img = Image.open(self.image_paths[idx]).convert("RGB")
        
        # 2. Apply Transforms: Resizes and Normalizes the image
        img_tensor = self.transform(img)
        
        # Returns tuple: (The Image Tensor, The Path String)
        # Returning path is useful for visualization/debugging later
        return img_tensor, str(self.image_paths[idx])

def get_train_dataloader(root_dir: str, category: str, batch_size: int = 16, img_size: int = 256):
    """
    Factory function to easily create the DataLoader.
    """
    # Instantiate the custom dataset
    dataset = MVTecADTrainDataset(root_dir, category, img_size)
    
    # Create DataLoader
    # shuffle=True: randomization is good for training stability (though less relevant for Feature Extraction)
    # num_workers=4: Uses 4 CPU cores to load images in parallel (speeds up training)
    # pin_memory=True: Speeds up transfer of data from CPU RAM to GPU VRAM
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
