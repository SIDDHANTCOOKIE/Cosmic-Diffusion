import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import structural_similarity as ssim
import numpy as np

from model import GalaxyLDM

# Mock Dataset
class GalaxyDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Return random tensor mocking a 3-channel image (RGB)
        return torch.randn(3, 64, 64), torch.randn(1, 77, 768) # Image + Mock Text Embedding

def compute_ssim(real_img, gen_img):
    """
    Compute Structural Similarity Index (SSIM) between real and generated images.
    Images should be numpy arrays in range [0, 1] or [0, 255].
    """
    # Simply taking the first channel for grayscale comparison or loop over channels
    real = real_img.permute(1, 2, 0).cpu().detach().numpy()
    gen = gen_img.permute(1, 2, 0).cpu().detach().numpy()
    
    # Normalize to [0, 1] for ssim if not already
    real = (real - real.min()) / (real.max() - real.min())
    gen = (gen - gen.min()) / (gen.max() - gen.min())

    return ssim(real, gen, channel_axis=2, data_range=1.0)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = GalaxyLDM().to(device)
    optimizer = AdamW(model.unet.parameters(), lr=1e-4)
    dataset = GalaxyDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    epochs = 1
    
    model.train()
    for epoch in range(epochs):
        for step, (images, text_embeddings) in enumerate(dataloader):
            images = images.to(device)
            text_embeddings = text_embeddings.squeeze(1).to(device)
            
            # Encode images to latent space
            # In a real scenario, we would scale these latents
            with torch.no_grad():
                latents = model.vae.encode(images).latent_dist.sample() * 0.18215
                
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, model.noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            
            # Add noise to latents
            noisy_latents = model.noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            noise_pred = model(noisy_latents, timesteps, text_embeddings)
            
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 5 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item()}")
                
        # --- Validation SSIM ---
        print("Evaluating SSIM on a generated sample...")
        # For demo purposes, we compare a 'generated' latent (just noise denoised for 1 step) to a real one
        # In production this would be full inference pipeline
        model.eval()
        with torch.no_grad():
             # Start from noise
             latents_gen = torch.randn_like(latents[0]).unsqueeze(0)
             # Basic single step denoising for demo (fake generation)
             pred_noise = model(latents_gen, torch.tensor([0], device=device), text_embeddings[0].unsqueeze(0))
             latents_out = latents_gen - pred_noise # Very rough approximation
             
             # Decode
             decoded_gen = model.vae.decode(latents_out / 0.18215).sample
             
             score = compute_ssim(images[0], decoded_gen[0])
             print(f"SSIM Score (Mock): {score:.4f}")
        model.train()

if __name__ == "__main__":
    train()
