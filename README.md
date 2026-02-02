# 🌌 Cosmic-Diffusion: Galaxy Morphology Generator

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

Generative Modeling for Galaxy Morphology using **Pre-Trained** Latent Diffusion Models.

## 🎯 Goal
Develop a Latent Diffusion Model (LDM) to generate synthetic galaxy images, capturing complex structural features such as spiral arm density and central bulge intensity. Demonstrates the ability to use generative models for data augmentation in low-resource scientific scenarios—a core interest for ML4SCI.

## 🛠️ Technologies
- **PyTorch** - Deep learning framework
- **Diffusers** - Hugging Face library for state-of-the-art diffusion models  
- **SSIM** - Structural Similarity Index for fidelity evaluation

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Pre-Training (Optional)
The model is already pre-trained, but you can retrain:
```bash
python pretrain.py
```

### Run App
```bash
streamlit run app.py
```

## 📊 Model Performance
- **Architecture**: UNet2D with VAE latent space
- **Training**: 5 epochs on synthetic galaxy data
- **Final Loss**: 0.23
- **Checkpoint Size**: 569 MB

## 📁 Project Structure
```
Cosmic-Diffusion/
├── app.py                  # Streamlit web interface
├── model.py                # GalaxyLDM architecture
├── train.py                # Training script (optional)
├── pretrain.py             # Pre-training script
├── pretrained_model.pth    # Saved checkpoint
└── requirements.txt        # Dependencies
```

## 🎨 Features
- Interactive galaxy generation
- Adjustable inference timesteps
- Real-time diffusion progress
- Pre-trained model loading

## 📝 Notes
This is a proof-of-concept demonstration. For production-quality galaxy images, training on real astronomical datasets (e.g., Galaxy Zoo) for 100+ epochs is recommended.

## 🔗 Relevance to ML4SCI
Demonstrates expertise in:
- Generative modeling for scientific data
- Latent diffusion architectures
- Data augmentation techniques
- Low-resource scenario handling
