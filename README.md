# GenVision 🛰️🌤️

**GenVision** is an end-to-end deep learning platform for **cloud removal in satellite imagery**, paired with downstream agricultural intelligence (NDVI, crop health, spectral indices), a FastAPI inference backend, and a modern Next.js interactive web studio.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Prerequisites & Git LFS](#prerequisites--git-lfs)
  - [1. Python ML Environment](#1-python-ml-environment)
  - [2. Backend API Service](#2-backend-api-service)
  - [3. Frontend Web Studio](#3-frontend-web-studio)
- [Model Training & Evaluation](#model-training--evaluation)
- [Downstream Agriculture Analytics](#downstream-agriculture-analytics)
- [License](#license)

---

## 🔬 Project Overview
Satellite optical sensors frequently suffer from cloud occlusion, which severely disrupts time-series earth observation, environmental monitoring, and precision agriculture. 

GenVision addresses this challenge by reconstructing clear surface imagery from cloudy optical inputs using:
- **U-Net**: Baseline encoder-decoder with skip connections.
- **Attention U-Net**: Attention gates highlighting cloud-free ground structures and suppressing irrelevant features.
- **Agricultural Post-Processing**: NDVI, NDWI, SAVI, and cloud shadow masking for vegetation analysis.

---

## 🏗️ System Architecture

```text
[ Cloudy Satellite Image (RGB / Multispectral) ]
                     │
                     ▼
       ┌───────────────────────────┐
       │     Preprocessing &       │
       │   Resolution Resizing     │
       └─────────────┬─────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
 ┌───────────────┐       ┌───────────────────┐
 │ Standard UNet │       │ Attention UNet    │
 └───────┬───────┘       └─────────┬─────────┘
         │                         │
         └───────────┬─────────────┘
                     │
                     ▼
      [ Cloud-Free Reconstructed Image ]
                     │
                     ▼
  ┌──────────────────────────────────────┐
  │ Downstream Agricultural Analytics    │
  │ • NDVI / NDWI / SAVI Calculations    │
  │ • Vegetative Health & Crop Zoning    │
  └──────────────────────────────────────┘
```

---

## 📁 Repository Structure

```text
cloud-removal-poc/
├── dataset/                    # 500 paired cloudy and clean RGB scenes (512x512)
│   ├── clean/                  # Ground truth cloud-free scenes
│   └── cloudy/                 # Corresponding cloudy scenes
├── models/                     # Trained PyTorch model checkpoints (Git LFS)
│   ├── attention_controlled_best.pth
│   ├── attention_controlled_final.pth
│   ├── attention_unet.pth
│   ├── best_attention_unet.pth
│   ├── unet.pth
│   ├── unet_controlled_best.pth
│   └── unet_controlled_final.pth
├── agriculture/                # Spectral indices, cloud masking, and validation
│   ├── cloud_mask.py           # Automated cloud masking
│   ├── spectral_indices.py     # NDVI, NDRE, SAVI, NDWI computations
│   ├── preprocessing.py        # Normalization and band alignment
│   └── visualization.py        # Colormap and false-color generation
├── backend/                    # FastAPI Inference Service
│   ├── app/
│   │   ├── main.py             # FastAPI entry point & routes
│   │   ├── model_service.py    # PyTorch inference pipeline
│   │   └── schemas.py          # Request & response data models
│   └── requirements.txt
├── frontend/                   # Next.js 16 Web Application
│   ├── src/
│   │   ├── app/                # App Router (Studio, Analytics, Comparison)
│   │   └── components/         # Interactive UI components, split sliders
│   └── package.json
├── outputs/                    # Training curves and prediction sample plots
├── results/                    # Dataset split manifests and evaluation metrics
│   ├── data_split.json         # 350 train / 75 val / 75 test partition
│   └── evaluation/             # Test benchmark metrics (PSNR, SSIM, MAE)
├── attention_unet.py           # Attention U-Net PyTorch module
├── unet.py                     # Standard U-Net PyTorch module
├── dataset.py                  # PyTorch Dataset & DataLoader implementation
├── train_attention_controlled.py # Controlled Attention U-Net training pipeline
├── train_unet_controlled.py    # Controlled U-Net training pipeline
├── evaluate_test_set.py        # Held-out test evaluation runner
├── requirements.txt            # Root Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites & Git LFS
Because model weights (`models/*.pth`) are large binary files (~120 MB each), this repository uses **Git LFS** (Large File Storage).

Ensure Git LFS is installed before cloning or pulling weights:
```bash
# Install Git LFS
git lfs install

# Clone the repository
git clone https://github.com/Chandrasekhar246800/Genvision.git
cd Genvision

# Fetch large model weights
git lfs pull
```

### 1. Python ML Environment
Create and activate a virtual environment:
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
```

### 2. Backend API Service
To start the inference backend:
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
API docs will be accessible at: `http://localhost:8000/docs`

### 3. Frontend Web Studio
Run the Next.js development server:
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser to interact with the Cloud Removal Studio.

---

## 📊 Model Training & Evaluation

### Training
To train the Attention U-Net using controlled seeds and split manifests:
```bash
python train_attention_controlled.py
```

### Evaluation
To run evaluation across the held-out test split (75 images) and calculate PSNR & SSIM:
```bash
python evaluate_test_set.py
```

Results will be stored in `results/evaluation/metrics_summary.json`.

---

## 🌾 Downstream Agriculture Analytics
Run spectral index validation:
```bash
python agriculture/validate_indices.py
```
Supported indices:
- **NDVI** (Normalized Difference Vegetation Index)
- **SAVI** (Soil Adjusted Vegetation Index)
- **NDWI** (Normalized Difference Water Index)

---

## 📄 License
This project is licensed under the MIT License.
