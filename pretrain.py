"""
Pre-training script for Cosmic-Diffusion
Trains the LDM on synthetic galaxy data and saves the checkpoint
"""
import torch
import torch.nn.functional as F
import numpy as np
from model import GalaxyLDM
from tqdm import tqdm

def create_synthetic_galaxy_batch(batch_size=4, image_size=64):
    """Generate synthetic galaxy images"""
    images = []
    for _ in range(batch_size):
        img = np.zeros((image_size, image_size, 3))
        
        # Random center
        cx, cy = np.random.randint(20, image_size-20, 2)
        
        # Create galaxy with spiral-like structure
        for x in range(image_size):
            for y in range(image_size):
                dx, dy = x - cx, y - cy
                r = np.sqrt(dx**2 + dy**2)
                theta = np.arctan2(dy, dx)
                
                # Spiral arms
                spiral = np.sin(2 * theta + r/5) * np.exp(-r/15)
                bulge = np.exp(-r/8)
                
                intensity = 0.7 * bulge + 0.3 * spiral + np.random.randn() * 0.1
                img[x, y] = np.clip(intensity, 0, 1)
                
        images.append(img)
    
    # Convert to tensor [B, 3, H, W]
    batch = torch.tensor(np.array(images)).permute(0, 3, 1, 2).float()
    batch = batch * 2 - 1  # Normalize to [-1, 1]
    return batch

def train(epochs=5, steps_per_epoch=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")
    
    model = GalaxyLDM(image_size=64).to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=1e-4)
    
    print("Starting pre-training...")
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        
        for step in pbar:
            # Generate batch
            images = create_synthetic_galaxy_batch(batch_size=2).to(device)
            
            # Encode to latents
            with torch.no_grad():
                latents = model.vae.encode(images).latent_dist.sample() * 0.18215
            
            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, 
                model.noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=device
            ).long()
            
            # Add noise
            noisy_latents = model.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Dummy conditioning
            encoder_hidden_states = torch.zeros(latents.shape[0], 77, 768).to(device)
            
            # Forward pass
            noise_pred = model(noisy_latents, timesteps, encoder_hidden_states)
            loss = F.mse_loss(noise_pred, noise)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint
    print("Saving checkpoint...")
    torch.save({
        'unet_state_dict': model.unet.state_dict(),
        'epochs': epochs,
        'final_loss': avg_loss
    }, 'pretrained_model.pth')
    
    print("Pre-training complete!")

if __name__ == "__main__":
    train()
