import torch  # The core PyTorch library for tensor computations and automatic gradients
import torch.nn as nn  # Submodule containing neural network layers (Conv2d, Linear, etc.)
import torch.nn.functional as F  # Functional interface for activations (ReLU, Sigmoid) and loss functions
from timm import create_model  # "PyTorch Image Models" library: Used to load the pre-trained MobileNetV3

# ==========================================
# 1. SOTA Feature Extractor (MobileNetV3)
# ==========================================
def get_feature_extractor(model_name="tf_mobilenetv3_large_100", pretrained=True):
    """
    Loads a pre-trained MobileNetV3-Large.
    OPTIMIZED for extreme speed (<100ms latency).
    """
    # Create the MobileNetV3 model using timm
    model = create_model(
        model_name,
        pretrained=pretrained,  # Load ImageNet weights (critical for transfer learning)
        features_only=True,     # Remove the classification head (we only want feature maps)
        # Extract features from layers 1, 2, and 3. 
        # Layer 0 is too shallow (edges only), Layer 4 is too deep/slow. 1-3 is the "Goldilocks" zone.
        out_indices=[1, 2, 3]   
    )
    model.eval()  # Set to evaluation mode (freezes BatchNorm stats and disables Dropout)

    # Define a custom wrapper class to handle resizing and concatenation automatically
    class PatchFeatureExtractor(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model  # Store the timm MobileNet backbone
        
        def forward(self, x):
            # Pass input image through MobileNet to get a list of feature maps [f1, f2, f3]
            features = self.model(x)
            
            # Identify the spatial size of the smallest feature map (deepest layer)
            # We align everything to this size so we can stack them.
            target_size = features[-1].shape[2:] 
            
            # Iterate through all feature maps and resize them to 'target_size'
            resized_features = [
                F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                for f in features
            ]
            
            # Concatenate features along the channel dimension (dim=1)
            # MobileNetV3 Large layers 1+2+3 have 24 + 40 + 112 = 176 channels total.
            # This rich feature vector describes the texture at that patch location.
            patch_features = torch.cat(resized_features, dim=1)
            
            return patch_features  # Returns tensor of shape [Batch, 176, H, W]

    return PatchFeatureExtractor(model)  # Return the instantiated wrapper

# ==========================================
# 2. Pixel-Based Models (Legacy)
# ==========================================

# Standard Convolutional Encoder: Compresses 256x256 image into a small vector
class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # Layer 1: 3 input channels (RGB) -> 32 filters. Stride 2 halves resolution (256->128).
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        # Layer 2: 32 -> 64 filters. Resolution becomes 64x64.
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        # Layer 3: 64 -> 128 filters. Resolution becomes 32x32.
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        # Layer 4: 128 -> 256 filters. Resolution becomes 16x16.
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1) 
        # Fully Connected Layer: Flattens the 16x16x256 tensor into the latent vector 'z'
        self.fc = nn.Linear(256 * 16 * 16, latent_dim) 

    def forward(self, x):
        # Apply Convolutions + ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Save batch size for reshaping
        batch_size = x.size(0)
        # Flatten and pass through linear layer to get the Latent Vector
        return self.fc(x.view(batch_size, -1))

# Standard Convolutional Decoder: Reconstructs image from latent vector
class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        # Expand latent vector back to spatial dimensions (256 * 16 * 16)
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        # Transposed Convolutions (Deconvolutions) to upscale resolution
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1) # 16x16 -> 32x32
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)  # 32x32 -> 64x64
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)   # 64x64 -> 128x128
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)    # 128x128 -> 256x256 (3 channels)

    def forward(self, z):
        # Project latent vector and reshape into a 3D tensor
        x = self.fc(z).view(-1, 256, 16, 16)
        # Apply Upscaling layers
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        # Sigmoid activation ensures output pixels are between 0.0 and 1.0
        return torch.sigmoid(self.deconv4(x))

# Convolutional Autoencoder (CAE) Wrapper
class CAE(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(latent_dim) # Instantiate Encoder
        self.decoder = Decoder(latent_dim) # Instantiate Decoder
        
    def forward(self, x):
        # Standard pass: Encode Input -> Latent Code -> Decode Reconstruction
        return self.decoder(self.encoder(x))

# Variational Autoencoder (VAE)
class VAE(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        # Convolutional layers identical to Encoder above
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        
        # The Key Difference: VAE learns a probability distribution (Mean + Variance)
        # Instead of one FC layer, we have two:
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)      # Predicts Mean
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)  # Predicts Log-Variance
        
        self.decoder = Decoder(latent_dim) # Reuse the same decoder

    def encode(self, x):
        # Run convolutions
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(h.size(0), -1) # Flatten
        # Output the parameters of the Gaussian distribution
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        The 'Reparameterization Trick': Allows backprop through random sampling.
        z = mu + sigma * epsilon
        """
        std = torch.exp(0.5 * logvar) # Convert log-variance to standard deviation
        return mu + torch.randn_like(std) * std # Add random noise

    def forward(self, x):
        mu, logvar = self.encode(x) # Get distribution params
        z = self.reparameterize(mu, logvar) # Sample latent vector
        return self.decoder(z), mu, logvar # Return recon AND params (for loss calculation)

# CAE Loss Function
def cae_loss(recon, x):
    # Uses L1 Loss (Mean Absolute Error) for sharper edges than MSE
    return F.l1_loss(recon, x, reduction='mean')

# VAE Loss Function
def vae_loss(recon, x, mu, logvar):
    # 1. Reconstruction Loss (L1) - How well does it look like the input?
    recon_loss = F.l1_loss(recon, x, reduction='mean') 
    
    # 2. KL Divergence - Forces the latent space to be a standard Normal distribution
    # This acts as a regularizer but causes the "blurriness" observed in your thesis.
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss is sum of both
    return recon_loss + kl_loss, recon_loss, kl_loss
