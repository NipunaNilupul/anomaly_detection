import torch
import torch.nn as nn
from timm import create_model

def get_feature_extractor(model_name="wide_resnet50_2", pretrained=True):
    """
    Loads a pre-trained Wide-ResNet-50 and extracts intermediate feature maps.
    This model is our new 'encoder'.
    """
    model = create_model(
        model_name,
        pretrained=pretrained,
        features_only=True,
        out_indices=[1, 2, 3] # Extract features from 3 different scales
    )
    model.eval() # Set to evaluation mode (no gradients needed)
    
    class FeatureExtractorWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.channels = self.model.feature_info.channels()
            self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in self.channels])
        
        def forward(self, x):
            features = self.model(x)
            pooled_features = [pool(f) for pool, f in zip(self.pools, features)]
            
            # h shape is (B, 1792, 1, 1)
            h = torch.cat(pooled_features, dim=1)
            
            # 💡 FIX: Only squeeze the last two (H, W) dimensions
            # This ensures output is always 2D: (B, 1792)
            return h.squeeze(-1).squeeze(-1)

    return FeatureExtractorWrapper(model)

if __name__ == "__main__":
    # Test the feature extractor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_feature_extractor().to(device)
    
    dummy_input = torch.randn(16, 3, 256, 256).to(device)
    
    with torch.no_grad():
        features = model(dummy_input)
    
    print("--- Feature Extractor Test ---")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output feature vector shape: {features.shape}")
    # Expected output: torch.Size([16, 1792])
