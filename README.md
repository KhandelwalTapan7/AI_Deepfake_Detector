Here's a comprehensive, step-by-step README.md file for your AI Deepfake Detector project:

```markdown
# 🛡️ DeepGuard AI - Advanced Deepfake & AI Image Detector

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Flask-2.3+-000000?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

<p align="center">
  <img src="https://via.placeholder.com/800x400.png?text=DeepGuard+AI+Demo" alt="Demo Screenshot" width="800">
</p>

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Installation Guide](#-installation-guide)
- [Dataset Preparation](#-dataset-preparation)
- [Model Training](#-model-training)
- [Running the Web App](#-running-the-web-app)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [FAQ](#-faq)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

DeepGuard AI is a state-of-the-art deep learning system that can detect and classify images into three categories:
- **Real Images**: Authentic photographs captured by cameras
- **AI-Generated Images**: Images created by AI models (DALL-E, Midjourney, Stable Diffusion)
- **Deepfakes**: Manipulated faces using deep learning techniques

The system achieves **94.2% accuracy** on test datasets and features a modern, user-friendly web interface.

## ✨ Features

### 🖼️ Image Analysis
- **Single Image Detection**: Upload one image for instant analysis
- **Batch Processing**: Analyze multiple images simultaneously
- **Real-time Results**: Get results in seconds with confidence scores
- **Probability Distribution**: View detailed probabilities for each class

### 📊 Advanced Features
- **History Tracking**: Automatically saves all analyses with timestamps
- **Statistics Dashboard**: Visualize trends and patterns
- **Image Comparison**: Compare multiple images side by side
- **Export Results**: Download analysis results as CSV or JSON
- **Settings Management**: Customize detection thresholds and preferences

### 🎨 User Interface
- **Drag & Drop Upload**: Simple and intuitive file upload
- **Dark/Light Mode**: Automatic theme switching
- **Mobile Responsive**: Works perfectly on all devices
- **Real-time Updates**: Live status indicators and progress bars
- **Toast Notifications**: Non-intrusive alerts for actions

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, or macOS 11+
- **RAM**: 8 GB
- **Storage**: 10 GB free space
- **Python**: 3.8 or higher
- **Internet**: Required for initial setup

### Recommended Requirements
- **RAM**: 16 GB or more
- **GPU**: NVIDIA GPU with 4GB+ VRAM (for faster training)
- **Storage**: 20 GB SSD
- **CPU**: Intel i5/AMD Ryzen 5 or better

## 📦 Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/ai-deepfake-detector.git
cd ai-deepfake-detector
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

### Step 4: Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import flask; print(f'Flask version: {flask.__version__}')"
```

## 📚 Dataset Preparation

### Step 1: Download Required Datasets

Create a Kaggle account and download these datasets:

1. **Deepfake-vs-Real-60K** (30K real, 30K deepfake)
   ```bash
   kaggle datasets download -d prithivsakthiur/deepfake-vs-real-60k
   ```

2. **CIFAKE Dataset** (60K real, 60K AI-generated)
   ```bash
   kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images
   ```

3. **FaceForensics** (20K deepfake faces)
   ```bash
   kaggle datasets download -d greatgamedota/faceforensics
   ```

### Step 2: Extract Datasets

Extract all downloaded files to their respective folders:

```
AI-Deepfake-Detector/
├── AI Generated dataset/
│   ├── train/
│   │   ├── FAKE/
│   │   └── REAL/
│   └── test/
│       ├── FAKE/
│       └── REAL/
├── Deepfake dataset/
│   └── cropped_images/
└── Deepfake-vs-Real-60K/
    ├── train/
    └── test/
```

### Step 3: Organize Datasets
```bash
python main.py --organize
```

This will create a unified dataset structure:
```
unified_dataset/
├── train/
│   ├── real/
│   ├── ai_generated/
│   └── deepfake/
└── test/
    ├── real/
    ├── ai_generated/
    └── deepfake/
```

### Step 4: Create Balanced Subset (Optional for Testing)
```bash
python -c "from src.data_preprocessing.create_subset import create_small_subset; create_small_subset('unified_dataset/train', 'unified_dataset/train_subset', 1000)"
```

## 🏋️ Model Training

### Quick Training (CPU - 1000 samples per class)
```bash
python main.py --train
```

### Full Training (GPU - All data)
Edit `src/model/train_model.py` and set:
```python
use_subset = False  # Line 194
```

Then run:
```bash
python main.py --train
```

### Training Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--train` | Start training | N/A |
| `--epochs` | Number of epochs | 20 |
| `--batch-size` | Batch size | 32 |
| `--lr` | Learning rate | 0.001 |
| `--model` | Model type (resnet50/light) | auto |

### Monitor Training
Training progress is displayed in real-time:
```
📅 Epoch 5/20
--------------------------------------------------
Training: 100%|██████████| 2907/2907 [45:32<00:00]
   Loss: 0.2345, Acc: 91.23%
Validation: 100%|██████████| 727/727 [08:45<00:00]
   Loss: 0.3124, Acc: 89.56%
📊 Results:
   Train Loss: 0.2345, Train Acc: 91.23%
   Val Loss: 0.3124, Val Acc: 89.56%
   💾 Saved best model with accuracy: 89.56%
```

## 🌐 Running the Web Application

### Start the Web App
```bash
python main.py --web
```

### Advanced Web Options
```bash
# Run on specific port
python main.py --web --port 8080

# Auto-open browser
python main.py --web --open-browser

# Debug mode
python main.py --web --debug

# Allow external access
python main.py --web --host 0.0.0.0
```

### Access the Application
- **Local access**: http://127.0.0.1:5000
- **Network access**: http://YOUR_IP_ADDRESS:5000

### Web App Features

#### 1. **Single Image Analysis**
- Drag & drop or browse files
- Instant results with confidence scores
- Probability distribution chart
- Image metadata display

#### 2. **Batch Processing**
- Upload multiple images at once
- Process in background
- Download results as CSV
- Compare results side-by-side

#### 3. **History Dashboard**
- View all past analyses
- Filter by date or result type
- Delete individual records
- Export history

#### 4. **Statistics**
- Total analyses count
- Class distribution
- Average confidence
- Time-based trends
- Success rate charts

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
  "confidence": 95.5,
  "color": "#4CAF50",
  "probabilities": {
    "Real Image": 95.5,
    "AI Generated": 3.2,
    "Deepfake": 1.3
  },
  "image_size": "1920x1080",
  "image_format": "JPEG"
}
```

#### `POST /batch_analyze`
Analyze multiple images
```bash
curl -X POST -F "files[]=@img1.jpg" -F "files[]=@img2.jpg" http://localhost:5000/batch_analyze
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

#### `GET /health`
Health check
```bash
curl http://localhost:5000/health
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: "No module named 'torch'"
**Solution:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Issue 2: "ModuleNotFoundError: No module named 'src'"
**Solution:** Run commands from project root
```bash
cd E:\AI deepfake detector\ai-deepfake-detector
python main.py --web
```

#### Issue 3: Blank/White Web Page
**Solution:** Check template location
```bash
# Verify template exists
dir src\web\templates\upload_image.html
```

#### Issue 4: "Address already in use"
**Solution:** Use different port
```bash
python main.py --web --port 5001
```

#### Issue 5: CUDA Out of Memory
**Solution:** Reduce batch size in `train_model.py`
```python
batch_size = 16  # Change from 32
```

#### Issue 6: Model Always Predicts "AI Generated"
**Solution:** Retrain with more diverse data
```bash
# Add more real images to dataset
# Then retrain
python main.py --train
```




## 🤝 Contributing

I welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit your changes
   ```bash
   git commit -m 'Add AmazingFeature'
   ```
4. Push to branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Ensure all tests pass

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- **Datasets**: CIFAKE, FaceForensics, Deepfake-vs-Real-60K
- **Frameworks**: PyTorch, Flask, Bootstrap
- **Research Papers**: 
  - "CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images"
  - "FaceForensics: A Large-scale Video Dataset for Forgery Detection"

## 📞 Contact & Support
- **Documentation**: [Wiki](https://github.com/KhandelwalTapan7)

