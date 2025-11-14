from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class MVTecADTrainDataset(Dataset):
    """Loads ONLY defect-free images (MVTec AD train/good)[cite: 45]."""
    def __init__(self, root_dir: str, category: str, img_size: int = 256):
        self.root_dir = Path(root_dir)
        self.category = category
        self.img_size = img_size
        self.good_dir = self.root_dir / category / "train" / "good" [cite: 184]
        
        if not self.good_dir.exists(): [cite: 46]
            raise FileNotFoundError(f"Training directory not found: {self.good_dir}")
        
        self.image_paths = sorted(p for p in self.good_dir.rglob("*") 
                                 if p.suffix.lower() in [".png", ".jpg", ".jpeg"]) [cite: 185]
        if not self.image_paths:
            raise ValueError(f"No images found in {self.good_dir}") [cite: 47]
        
        # Transform ensures 256x256 and normalization [0, 1]
        self.transform = T.Compose([
            T.Resize((img_size, img_size)), 
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet Norm
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB") [cite: 186]
        return self.transform(img), str(self.image_paths[idx])

def get_train_dataloader(root_dir: str, category: str, batch_size: int = 16, img_size: int = 256):
    dataset = MVTecADTrainDataset(root_dir, category, img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
