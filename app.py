import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from model import GalaxyLDM

st.set_page_config(page_title="Cosmic-Diffusion", layout="wide")

st.title("🌌 Cosmic-Diffusion: Galaxy Morphology Generator")
st.markdown("Generative Modeling for Galaxy Morphology using **Pre-Trained** Latent Diffusion Models.")

# Load pre-trained model
@st.cache_resource
def load_model():
    model = GalaxyLDM(image_size=64)
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'pretrained_model.pth')
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.unet.load_state_dict(checkpoint['unet_state_dict'])
        st.sidebar.success(f"✅ Loaded pre-trained model (Epochs: {checkpoint['epochs']}, Loss: {checkpoint['final_loss']:.4f})")
    else:
        st.sidebar.warning("⚠️ No checkpoint found, using untrained model")
    
    return model

model = load_model()

col1, col2 = st.columns(2)

with col1:
    st.header("1. Generation Settings")
    image_size = st.slider("Image Size", 32, 128, 64, step=32, disabled=True)
    timesteps = st.slider("Inference Timesteps", 10, 50, 20)
    
    if st.button("Generate Galaxy Image", type="primary"):
        with st.spinner("Running diffusion process..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load model within button context
            curr_model = load_model()
            curr_model.to(device)
            curr_model.eval()
            
            latent_size = image_size // 8
            latents = torch.randn(1, 4, latent_size, latent_size).to(device)
            
            scheduler = curr_model.noise_scheduler
            scheduler.set_timesteps(timesteps)
            
            progress_bar = st.progress(0)
            for idx, t in enumerate(scheduler.timesteps):
                latent_model_input = latents
                encoder_hidden_states = torch.zeros(1, 77, 768).to(device)
                
                with torch.no_grad():
                    noise_pred = curr_model(latent_model_input, t, encoder_hidden_states)
                    
                latents = scheduler.step(noise_pred, t, latents).prev_sample
                progress_bar.progress((idx + 1) / len(scheduler.timesteps))
                
            with torch.no_grad():
                latents = 1 / 0.18215 * latents
                image = curr_model.vae.decode(latents).sample
                
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            
            st.session_state['generated_image'] = image

with col2:
    st.header("2. Generated Output")
    if 'generated_image' in st.session_state:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(st.session_state['generated_image'])
        ax.axis('off')
        ax.set_title("Generated Galaxy (Pre-Trained Model)")
        st.pyplot(fig)
    else:
        st.info("Click 'Generate Galaxy Image' to see results")

st.divider()
st.header("3. Model Information")
st.markdown("""
**Architecture**: Latent Diffusion Model (LDM) with UNet2D backbone  
**Training Data**: Synthetic galaxy images with spiral arms and central bulges  
**Metric**: MSE Loss for noise prediction
""")
