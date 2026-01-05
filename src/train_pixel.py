import os  # Standard library for filesystem operations (creating the 'models' folder)
import torch  # PyTorch: The core library for deep learning tensors and autodiff
from torch.utils.data import Subset  # Utility to create a smaller dataset from a list of indices (for splitting)
from sklearn.model_selection import train_test_split  # Scikit-learn tool to randomly split data into Train/Val sets
from tqdm import tqdm  # Library for the progress bar to track training speed
from src.dataloader import MVTecADTrainDataset  # Import your custom dataset class
from src.models import CAE, VAE, cae_loss, vae_loss  # Import your neural architectures and loss functions

def train_pixel_model(model_type="cae", category="bottle", latent_dim=512, batch_size=8, num_epochs=50, lr=1e-4, img_size=256):
    # Automatically select the GPU if available to speed up training; otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the directory to save trained model weights. 'exist_ok' prevents crashing if it already exists.
    os.makedirs("models", exist_ok=True)
    
    # 1. Prepare Data
    # Initialize the custom dataset for the specific category (e.g., 'bottle')
    dataset = MVTecADTrainDataset("data/mvtec_ad", category, img_size=img_size)
    
    # Create a 90/10 Train/Validation split.
    # Validation is critical to detect overfitting (where the model memorizes inputs instead of learning features).
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.1, random_state=42)
    
    # Create PyTorch DataLoaders
    # Subset: Wraps the dataset with specific indices
    # shuffle=True (Train): Essential for Stochastic Gradient Descent to work correctly
    # num_workers=4: Uses parallel CPU cores to load images, preventing GPU starvation
    train_loader = torch.utils.data.DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # shuffle=False (Val): Order doesn't matter for validation, and it's slightly faster
    val_loader = torch.utils.data.DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Initialize Model & Loss
    if model_type == "cae":
        model = CAE(latent_dim).to(device) # Instantiate Convolutional Autoencoder and move to GPU
        loss_fn = cae_loss # Use L1 Reconstruction Loss
    else: # VAE
        model = VAE(latent_dim).to(device) # Instantiate Variational Autoencoder
        # VAE loss returns a tuple (total, recon, kl). We only need the first element [0] for backprop.
        loss_fn = lambda recon, x, mu, logvar: vae_loss(recon, x, mu, logvar)[0]
    
    # Initialize Adam Optimizer. It adapts learning rates per parameter, converging faster than standard SGD.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    
    print(f"\n Starting PIXEL-BASED {model_type.upper()} training on '{category}'")
    
    # Track the best validation loss to save the optimal checkpoint
    best_val_loss = float('inf')
    
    # 3. Training Loop
    for epoch in range(num_epochs):
        model.train() # Set model to Train mode (enables Dropout and BatchNorm updates)
        train_loss = 0.0
        
        # Iterate through batches with a progress bar
        for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
            data = data.to(device) # Move image batch to GPU
            
            # Reset gradients from previous step (accumulating gradients is a common bug)
            optimizer.zero_grad()
            
            # Forward Pass
            output = model(data)
            
            # Calculate Loss based on model type
            if model_type == "cae":
                loss = loss_fn(output, data) # Compare reconstruction vs original
            else:
                recon, mu, logvar = output # Unpack VAE outputs
                loss = loss_fn(recon, data, mu, logvar) # Calculate ELBO loss

            # Backward Pass (Backpropagation)
            loss.backward() # Compute gradients
            optimizer.step() # Update model weights
            
            train_loss += loss.item() # Accumulate batch loss
        
        # 4. Validation Loop
        model.eval() # Set model to Evaluation mode (freezes BatchNorm, disables Dropout)
        val_loss = 0.0
        
        # Disable gradient calculation context manager (saves massive amounts of VRAM)
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device) # Move to GPU
                output = model(data) # Forward pass
                
                # Calculate validation loss (same logic as training)
                if model_type == "cae":
                    loss = loss_fn(output, data)
                else:
                    recon, mu, logvar = output
                    loss = loss_fn(recon, data, mu, logvar)
                val_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")
        
        # 5. Checkpointing
        # Save model only if validation loss improves (prevents saving an overfitted model)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            path = f"models/{model_type}_{category}_best.pth"
            torch.save(model.state_dict(), path) # Serialize weights to disk
            print(f"   Saved best weights: {path}")

if __name__ == "__main__":
    # Entry point: Train a CAE model on the 'bottle' category
    # Change "cae" to "vae" to train the Variational Autoencoder
    train_pixel_model("cae", "bottle")
