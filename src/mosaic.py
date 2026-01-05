import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

# Categories to feature in the main report (Diverse mix)
FEATURED_CATEGORIES = ['bottle', 'screw', 'carpet', 'leather', 'cable', 'pill']

def create_mosaic():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    for i, cat in enumerate(FEATURED_CATEGORIES):
        # Find a heatmap for this category
        heatmap_dir = Path(f"results/heatmaps/{cat}")
        
        # Try to find a specific defect type (visuals look better with defects)
        # We just grab the first image from the first defect folder we find
        found_img = None
        if heatmap_dir.exists():
            for defect_folder in heatmap_dir.iterdir():
                if defect_folder.is_dir() and defect_folder.name != 'good':
                    images = list(defect_folder.glob("*.png"))
                    if images:
                        found_img = str(images[0])
                        break
        
        if found_img:
            img = cv2.imread(found_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img)
            axes[i].set_title(f"{cat.capitalize()}", fontsize=12)
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, "Image Not Found", ha='center')
            axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("results/Figure_4_4_Generalization_Mosaic.png", dpi=300)
    print("✅ Mosaic saved to results/Figure_4_4_Generalization_Mosaic.png")

if __name__ == "__main__":
    create_mosaic()
