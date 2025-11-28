import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

# ==========================================
# 1. SOTA Feature Extractor (MobileNetV3)
# ==========================================
def get_feature_extractor(model_name="tf_mobilenetv3_large_100", pretrained=True):
    """
    Loads a pre-trained MobileNetV3-Large.
    OPTIMIZED for extreme speed (<100ms latency).
    """
    model = create_model(
        model_name,
        pretrained=pretrained,
        features_only=True,
        # 💡 FIX: Use layers 1, 2, 3 (Total 176 channels) for max speed and safety
        out_indices=[1, 2, 3] 
    )
    model.eval()

    class PatchFeatureExtractor(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, x):
            features = self.model(x)
            
            # Align features to the smallest feature map (Layer 3)
            target_size = features[-1].shape[2:] 
            
            resized_features = [
                F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
                for f in features
            ]
            
            # MobileNetV3 layers 1+2+3: 24 + 40 + 112 = 176 channels
            patch_features = torch.cat(resized_features, dim=1)
            
            return patch_features

    return PatchFeatureExtractor(model)

# ==========================================
# 2. Pixel-Based Models (Legacy)
# ==========================================
class Encoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1) 
        self.fc = nn.Linear(256 * 16 * 16, latent_dim) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        batch_size = x.size(0)
        return self.fc(x.view(batch_size, -1))

class Decoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 16 * 16)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1) 

    def forward(self, z):
        x = self.fc(z).view(-1, 256, 16, 16)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        return torch.sigmoid(self.deconv4(x))

class CAE(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    def forward(self, x):
        return self.decoder(self.encoder(x))

class VAE(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        self.decoder = Decoder(latent_dim)

    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def cae_loss(recon, x):
    return F.l1_loss(recon, x, reduction='mean')

def vae_loss(recon, x, mu, logvar):
    recon_loss = F.l1_loss(recon, x, reduction='mean') 
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss
