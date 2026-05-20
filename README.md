
<div align="center">
  
  <img src="https://img.shields.io/badge/AURORA-Deepfake%20Detector-8b5cf6?style=for-the-badge&logo=ai&logoColor=white"/>
  
  <h1>🛡️ AURORA Deepfake Detector</h1>
  
  <p><strong>State-of-the-art AI-powered deepfake & AI-generated image detection system</strong></p>
  
  <img src="https://img.shields.io/badge/Accuracy-95.27%25-brightgreen?style=flat-square&logo=target"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Flask-2.3+-000000?style=flat-square&logo=flask"/>
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square"/>
  <img src="https://img.shields.io/badge/GPU-Supported-blue?style=flat-square&logo=nvidia"/>
  
  <br>
  
  <img src="https://img.shields.io/badge/Model-EfficientNet--B4-8b5cf6?style=flat-square"/>
  <img src="https://img.shields.io/badge/Classes-3%20(Real%20%7C%20AI%20%7C%20Deepfake)-14b8a6?style=flat-square"/>
  <img src="https://img.shields.io/badge/Inference-<%3C1s-10b981?style=flat-square"/>
  
</div>

---

## 📋 Table of Contents
- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🔬 Detection Methods](#-detection-methods)
- [📊 Model Performance](#-model-performance)
- [💻 System Requirements](#-system-requirements)
- [📦 Installation Guide](#-installation-guide)
- [🚀 Running the Application](#-running-the-application)
- [📡 API Documentation](#-api-documentation)
- [📁 Project Structure](#-project-structure)
- [🔧 Troubleshooting](#-troubleshooting)
- [❓ FAQ](#-faq)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [🙏 Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**AURORA** is an advanced deepfake detection platform that uses state-of-the-art **EfficientNet-B4** deep learning architecture to classify images into three categories with **95.27% accuracy**:

| Category | Description | Examples |
|----------|-------------|----------|
| 🟢 **Real Image** | Authentic photographs captured by cameras | Smartphone photos, DSLR images, natural scenes |
| 🤖 **AI Generated** | Images created by AI models | DALL-E, Midjourney, Stable Diffusion, StyleGAN |
| 🔴 **Deepfake** | Manipulated faces using deep learning | Face-swapped, lip-synced, expression-manipulated |

### Why AURORA?

In an era where AI-generated content is becoming indistinguishable from reality, AURORA provides a reliable way to verify digital content authenticity. Whether you're a journalist, researcher, content moderator, or concerned citizen, AURORA helps you separate truth from deception.

**Key Achievements:**
- ✅ **95.27%** accuracy on test datasets
- ⚡ **<1 second** analysis time per image
- 🎯 **3-class detection** (Real/AI/Deepfake)
- 🧠 **EfficientNet-B4** architecture with attention mechanism
- 📊 **8 forensic detection layers** working in harmony
- 🌐 **Modern web interface** with 6 complete pages

---

## ✨ Features

### 🖼️ Image Analysis
| Feature | Description |
|---------|-------------|
| **Single Image Detection** | Upload one image for instant AI-powered analysis |
| **Batch Processing** | Analyze up to 50 images simultaneously |
| **Real-time Results** | Get results in under 1 second with confidence scores |
| **Probability Distribution** | View detailed probabilities for each of the 3 classes |
| **Drag & Drop Upload** | Simple and intuitive file upload interface |

### 📊 Advanced Features
| Feature | Description |
|---------|-------------|
| **History Tracking** | Automatically saves all analyses with timestamps |
| **Statistics Dashboard** | Visualize trends and patterns with interactive charts |
| **Image Comparison** | Compare multiple images side by side |
| **Export Results** | Download analysis history as JSON |
| **Filter History** | Filter by Real, AI Generated, or Deepfake |

### 🎨 User Interface
| Feature | Description |
|---------|-------------|
| **6 Complete Pages** | Home, Detect, Features, Analytics, History, About |
| **Animated Background** | Aurora-inspired gradient animations |
| **Glass Morphism** | Modern translucent UI elements |
| **Mobile Responsive** | Works perfectly on all devices |
| **Real-time Updates** | Live status indicators and progress bars |
| **Toast Notifications** | Non-intrusive alerts for user actions |

---

## 🔬 Detection Methods

AURORA uses **8 forensic analysis techniques** working in harmony with deep learning:

| # | Method | Description | Contribution |
|---|--------|-------------|--------------|
| 1 | **Error Level Analysis (ELA)** | Detects compression inconsistencies that reveal manipulation | 20% |
| 2 | **Noise Pattern Detection** | Identifies unnatural noise patterns in AI-generated images | 15% |
| 3 | **Edge Artifact Detection** | Analyzes blurry or inconsistent edges around manipulated regions | 15% |
| 4 | **Frequency Analysis (FFT)** | Detects frequency anomalies using Fourier transform | 20% |
| 5 | **Color Consistency Check** | Identifies color and lighting mismatches in composites | 10% |
| 6 | **Texture Analysis** | Detects unnaturally smooth or repetitive textures | 10% |
| 7 | **Face Detection & Quality** | Specialized deepfake facial manipulation detection | 5% |
| 8 | **Compression Artifact Analysis** | Detects artifacts from multiple saves or heavy compression | 5% |

**Combined Neural Network + Forensic Accuracy:** 95.27%

---

## 📊 Model Performance

### Training Details
| Parameter | Value |
|-----------|-------|
| **Architecture** | EfficientNet-B4 |
| **Input Size** | 224×224 pixels |
| **Training Images** | 120,000+ (real, AI-generated, deepfake) |
| **Epochs** | 8 |
| **Batch Size** | 32 |
| **Learning Rate** | 2e-4 (adaptive) |
| **Optimizer** | AdamW with weight decay |
| **Loss Function** | CrossEntropyLoss with label smoothing |

### Confusion Matrix
```
                 Predicted
              Real    AI    Deepfake
Actual Real    ✓✓✓     ✓      ✓
Actual AI       ✓    ✓✓✓     ✓
Actual Deepfake ✓      ✓    ✓✓✓
```

### Performance Metrics
| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 95.27% |
| **Real Image Precision** | 94.8% |
| **AI Generated Precision** | 95.2% |
| **Deepfake Precision** | 95.8% |
| **Inference Time (CPU)** | ~0.8 seconds |
| **Inference Time (GPU)** | ~0.2 seconds |

---

## 💻 System Requirements

### Minimum Requirements
| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11, Ubuntu 20.04+, macOS 11+ |
| **RAM** | 8 GB |
| **Storage** | 5 GB free space |
| **Python** | 3.8 or higher |
| **Internet** | Required for initial setup |

### Recommended Requirements
| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 11, Ubuntu 22.04+, macOS 12+ |
| **RAM** | 16 GB or more |
| **GPU** | NVIDIA GPU with 4GB+ VRAM (for faster inference) |
| **Storage** | 10 GB SSD |
| **CPU** | Intel i5/AMD Ryzen 5 or better |

---

## 📦 Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/aurora-deepfake-detector.git
cd aurora-deepfake-detector
```

### Step 2: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Required packages:**
```txt
Flask==2.3.3
flask-cors==4.0.0
opencv-python==4.8.1.78
Pillow==10.0.0
numpy==1.24.3
torch==2.0.1
torchvision==0.15.2
scikit-learn==1.3.0
matplotlib==3.7.2
tqdm==4.65.0
```

### Step 4: Download the Trained Model
Place your trained `best_model.pth` in the `models/` directory:
```
aurora-deepfake-detector/
└── models/
    └── best_model.pth    # 95.27% accuracy model
```

### Step 5: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import flask; print(f'Flask: {flask.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
```

**Expected Output:**
```
PyTorch: 2.0.1
Flask: 2.3.3
OpenCV: 4.8.1
```

---

## 🚀 Running the Application

### Start the Web App
```bash
python app.py
```

### Advanced Options
```bash
# Run on custom port
python app.py --port 8080

# Run in debug mode (for development)
python app.py --debug

# Run with custom host (for network access)
python app.py --host 0.0.0.0
```

### Access the Application
| Access Type | URL |
|-------------|-----|
| **Local access** | http://127.0.0.1:5000 |
| **Local network** | http://YOUR_IP:5000 |

### Web App Pages

| Page | Features |
|------|----------|
| **🏠 Home** | Hero section, statistics dashboard, feature highlights |
| **📤 Detect** | Drag & drop upload, image preview, real-time analysis, confidence bars |
| **⚡ Features** | 8 detection methods explained with icons |
| **📊 Analytics** | Interactive pie chart and trend charts, AI insights |
| **📜 History** | View all past analyses, filter by type, delete entries |
| **💫 About** | Project information, technology stack, version details |

---

## 📡 API Documentation

### RESTful Endpoints

#### `POST /analyze`
Analyze a single image

```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/analyze
```

**Response:**
```json
{
  "class": "Real Image",
  "confidence": 0.95,
  "real_probability": 0.95,
  "ai_probability": 0.03,
  "deepfake_probability": 0.02,
  "recommendation": "✅ AUTHENTIC IMAGE — 95.0% confidence.",
  "analysis_mode": "neural_network",
  "model_accuracy": 95.27
}
```

#### `POST /batch_analyze`
Analyze multiple images

```bash
curl -X POST -F "files[]=@img1.jpg" -F "files[]=@img2.jpg" http://localhost:5000/batch_analyze
```

**Response:**
```json
{
  "results": [
    {"filename": "img1.jpg", "result": {...}},
    {"filename": "img2.jpg", "result": {...}}
  ],
  "total": 2
}
```

#### `GET /history`
Get analysis history

```bash
curl http://localhost:5000/history
```

#### `GET /stats`
Get statistics

```bash
curl http://localhost:5000/stats
```

**Response:**
```json
{
  "total_analyses": 150,
  "real_count": 80,
  "ai_count": 40,
  "deepfake_count": 30,
  "avg_confidence": 0.89,
  "model_accuracy": 95.27
}
```

#### `GET /model/status`
Get model status

```bash
curl http://localhost:5000/model/status
```

**Response:**
```json
{
  "loaded": true,
  "mode": "neural_network",
  "device": "cpu",
  "classes": ["Real Image", "AI Generated", "Deepfake"],
  "accuracy": 95.27
}
```

#### `GET /health`
Health check endpoint

```bash
curl http://localhost:5000/health
```

---

## 📁 Project Structure

```
aurora-deepfake-detector/
│
├── 📁 models/
│   └── best_model.pth              # Trained model (95.27% accuracy)
│
├── 📁 templates/
│   └── upload_image.html           # Main UI template
│
├── 📁 uploads/                      # Temporary uploads (auto-created)
├── 📁 history/                      # Analysis history (auto-created)
│
├── 📄 app.py                        # Main Flask application
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # Documentation
└── 📄 LICENSE                       # MIT License
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: "ModuleNotFoundError: No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Issue 2: Model not loading
**Solution:**
```bash
# Ensure model exists in correct location
ls models/best_model.pth

# Check model file size (should be ~70-80 MB)
du -h models/best_model.pth
```

#### Issue 3: "Address already in use" (Port 5000)
**Solution:**
```bash
# Kill process using port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Or use different port
python app.py --port 8080
```

#### Issue 4: Template not found
**Solution:**
```bash
# Ensure template exists
ls templates/upload_image.html

# Create templates folder if missing
mkdir templates
```

#### Issue 5: Slow inference time
**Solutions:**
- Use GPU if available
- Reduce image size before upload
- Close other applications
- Use batch processing for multiple images

#### Issue 6: Out of memory
**Solution:**
```python
# Reduce batch size in batch_analyze
# Or process images one by one
```

---

## ❓ FAQ

### General Questions

**Q: How accurate is AURORA?**
A: AURORA achieves **95.27% accuracy** on our test datasets with balanced real, AI, and deepfake images.

**Q: Can AURORA detect all types of deepfakes?**
A: AURORA detects most common deepfake techniques including face-swapping, lip-syncing, and expression manipulation. Detection rates vary based on quality.

**Q: Does AURORA work on videos?**
A: Currently only images. For videos, extract frames first and analyze them.

**Q: Is an internet connection required?**
A: Only for initial setup and downloading dependencies. The analysis works offline.

### Technical Questions

**Q: What's the minimum RAM requirement?**
A: 8GB minimum, 16GB recommended for smooth operation.

**Q: Can I run this on CPU only?**
A: Yes! Training takes longer but inference is still fast (<1 second per image).

**Q: How much storage is needed?**
A: ~5GB for dependencies and model, plus space for uploads (auto-cleaned).

**Q: Does it support GPU acceleration?**
A: Yes, NVIDIA GPUs with CUDA are supported for faster inference.

### Model Questions

**Q: What architecture is used?**
A: EfficientNet-B4 with custom classification head (1792 → 512 → 256 → 3).

**Q: What datasets was it trained on?**
A: FaceForensics++, CIFAKE, and custom deepfake datasets (120K+ images).

**Q: Can I train my own model?**
A: Yes, use the training notebook in Google Colab (GPU recommended).

---

## 🤝 Contributing

I welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Development Guidelines
| Area | Guideline |
|------|-----------|
| **Code Style** | Follow PEP 8 |
| **Testing** | Add tests for new features |
| **Documentation** | Update README and docstrings |
| **Commits** | Use descriptive commit messages |

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` file for more information.

```
MIT License

Copyright (c) 2024-2026 Aurora Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 🙏 Acknowledgments

### 📚 Datasets
- [FaceForensics++](https://github.com/ondyari/FaceForensics) - Deepfake face videos
- [CIFAKE Dataset](https://www.kaggle.com/datasets/birdy654/cifake) - Real and AI-generated images
- [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge)

### 🔧 Frameworks & Libraries
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Flask](https://flask.palletsprojects.com/) - Web framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) - Model architecture

### 📖 Research Papers
- "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- "FaceForensics++: Learning to Detect Manipulated Facial Images"
- "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images"

---


  **⭐ Star this repo if you find it useful! ⭐**
  
  Made with ❤️ by the Aurora Team
  
  <sub>Protecting digital truth, one image at a time.</sub>
  
  ---
  
  <sub>Version 3.0.0 | 95.27% Accuracy | Real-time Deepfake Detection</sub>
  
</div>
```
