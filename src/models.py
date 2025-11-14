import torch
import torch.nn as nn
from timm import create_model

def get_feature_extractor(model_name="wide_resnet50_2", pretrained=True):
    """
    Loads a pre-trained Wide-ResNet-50 and extracts intermediate feature maps.
    This model is our new 'encoder'.
    """
    # Use 'features_only=True' to get an intermediate feature extractor
    # 'out_indices' specifies which layers to return (layers 1, 2, and 3)
    model = create_model(
        model_name,
        pretrained=pretrained,
        features_only=True,
        out_indices=[1, 2, 3] # Extract features from 3 different scales
    )
    model.eval() # Set to evaluation mode (no gradients needed)
    
    # We patch the model to add AdaptiveAvgPool2d to each output layer
    # This gives us a fixed-size feature vector per layer regardless of input size
    # and reduces the feature map HxW to 1x1.
    
    # Example: Layer 1 output (B, 256, 64, 64) -> (B, 256, 1, 1)
    # Example: Layer 2 output (B, 512, 32, 32) -> (B, 512, 1, 1)
    # Example: Layer 3 output (B, 1024, 16, 16) -> (B, 1024, 1, 1)
    
    class FeatureExtractorWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            # Get the number of output channels for each layer
            self.channels = self.model.feature_info.channels()
            # Create a pooling layer for each feature map
            self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in self.channels])
        
        def forward(self, x):
            # Get the list of feature maps
            features = self.model(x)
            # Apply pooling to each feature map
            pooled_features = [pool(f) for pool, f in zip(self.pools, features)]
            # Concatenate all pooled features along the channel dimension
            # (B, 256, 1, 1) + (B, 512, 1, 1) + (B, 1024, 1, 1) -> (B, 1792, 1, 1)
            h = torch.cat(pooled_features, dim=1)
            # Squeeze the HxW dimensions -> (B, 1792)
            return h.squeeze()

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
