import os
import torch
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.dataloader import MVTecADTrainDataset
from src.models import CAE, VAE, cae_loss, vae_loss

def train_pixel_model(model_type="cae", category="bottle", latent_dim=512, batch_size=8, num_epochs=50, lr=1e-4, img_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("models", exist_ok=True)
    
    dataset = MVTecADTrainDataset("data/mvtec_ad", category, img_size=img_size)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
    
    train_loader = torch.utils.data.DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    if model_type == "cae":
        model = CAE(latent_dim).to(device)
        loss_fn = cae_loss
    else: # VAE
        model = VAE(latent_dim).to(device)
        loss_fn = lambda recon, x, mu, logvar: vae_loss(recon, x, mu, logvar)[0]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    
    print(f"\n🚀 Starting PIXEL-BASED {model_type.upper()} training on '{category}'")
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            data = data.to(device)
            optimizer.zero_grad()
            
            output = model(data)
            if model_type == "cae":
                loss = loss_fn(output, data)
            else:
                recon, mu, logvar = output
                loss = loss_fn(recon, data, mu, logvar)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                output = model(data)
                if model_type == "cae":
                    loss = loss_fn(output, data)
                else:
                    recon, mu, logvar = output
                    loss = loss_fn(recon, data, mu, logvar)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            path = f"models/{model_type}_{category}_best.pth"
            torch.save(model.state_dict(), path)
            print(f"  ✅ Saved best weights: {path}")

if __name__ == "__main__":
    # You can change this to "vae" to train the VAE
    train_pixel_model("cae", "bottle")
