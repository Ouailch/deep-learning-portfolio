# 🎓 Deep Learning Portfolio - Ouail Chlih

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue)](https://en.cppreference.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Professional portfolio of deep learning and computer vision projects. Demonstrates expertise in model training, deployment, and optimization across multiple platforms.

## 📚 Projects Overview

### 1. 🪟 YOLOs-CPP: Window & Door Detection
**Production-grade C++ implementation for real-time building façade analysis**

- **Tech**: C++17, OpenCV, ONNX Runtime
- **Features**: Image/Video/Camera inference, GPU support, cross-platform
- **Performance**: 30+ FPS on CPU
- **Use Case**: Building Information Modeling (BIM), Smart Cities

**Quick Links**:
- [📖 Full Documentation](yolos-cpp/README.md)
- [💻 Source Code](yolos-cpp/src/)
- [🛠️ Setup Guide](yolos-cpp/README.md#-quick-start)

---

### 2. 🍎 Fruit Quality Classification
**Deep learning system for automated fruit quality assessment**

- **Tech**: Python, PyTorch, YOLOv11, Transfer Learning
- **Features**: Single image, batch, video, real-time webcam
- **Accuracy**: 92%+ validation accuracy
- **Classes**: Good, Bad (damaged), Rotten

**Quick Links**:
- [📖 Full Documentation](fruit-quality/README.md)
- [🐍 Training Scripts](fruit-quality/scripts/)
- [⚙️ Setup Instructions](fruit-quality/README.md#-installation)

---

## 🚀 Quick Start

### YOLOs-CPP
```bash
cd yolos-cpp
./build.sh
./bin/image_inference data/test_image.jpg
```

### Fruit Quality
```bash
cd fruit-quality
bash setup.sh
source venv/bin/activate
python -c "from scripts.inference import FruitQualityClassifier; FruitQualityClassifier('fruit_quality_model.pt').predict_camera()"
```

## 🎯 Key Competencies

### Computer Vision
- ✅ Object Detection (YOLO family)
- ✅ Image Classification (Transfer Learning)
- ✅ Real-time Inference
- ✅ Data Augmentation & Preprocessing

### Deep Learning
- ✅ Model Architecture Design
- ✅ Transfer Learning
- ✅ Hyperparameter Optimization
- ✅ Training & Validation Strategies

### Software Engineering
- ✅ C++ Production Code
- ✅ Python Data Science Stack
- ✅ Model Deployment & Export (ONNX, TensorFlow)
- ✅ Cross-Platform Development
- ✅ Performance Optimization

### Tools & Frameworks
| Category | Technologies |
|----------|--------------|
| **Deep Learning** | PyTorch, Ultralytics YOLO, TensorFlow |
| **Computer Vision** | OpenCV, scikit-image |
| **Languages** | Python, C++17 |
| **Inference** | ONNX Runtime, TensorRT |
| **Build Systems** | CMake, pip, conda |
| **Version Control** | Git, GitHub |

## 📊 Performance Metrics

### YOLOs-CPP
| Metric | Value |
|--------|-------|
| Inference Speed (CPU) | 25-35 FPS |
| Inference Latency | 28-40 ms |
| Model Size | ~140 MB |
| ONNX Runtime | 1.20.0 |

### Fruit Quality
| Metric | Value |
|--------|-------|
| Validation Accuracy | 92.3% |
| Inference Speed (GPU) | 30+ FPS |
| Model Size | ~5.9 MB |
| Training Time | ~45 min (200 epochs) |

## 🏗️ Repository Structure

```
github-projects/
├── yolos-cpp/                  # C++ Detection Project
│   ├── src/                    # Source code (3 inference modes)
│   ├── include/                # Header files
│   ├── CMakeLists.txt         # Build configuration
│   ├── README.md              # Full documentation
│   └── build.sh               # Automated build script
│
├── fruit-quality/              # Python Classification Project
│   ├── scripts/               # Training & inference
│   ├── notebooks/             # Jupyter notebooks
│   ├── data/                  # Sample images
│   ├── README.md              # Full documentation
│   ├── requirements.txt       # Python dependencies
│   └── setup.sh               # Environment setup
│
├── LICENSE                     # MIT License
└── README.md                  # This file
```

## 🎓 Educational Background

- **University**: Hochschule Niederrhein (University of Applied Sciences)
- **Program**: Bachelor of Science in IT
- **Specialization**: Deep Learning & Computer Vision
- **Supervisor**: Prof. Dr. Steffen Goebbels (iPattern Institute)

## 💼 Practical Experience

### Praxisphase (Internship)
- **Duration**: 06.01.2025 - 23.03.2025 (12 weeks)
- **Organization**: iPattern Institute, Hochschule Niederrhein
- **Project**: "3D Building Model Calculation for Cadastre from Remote Sensing Data"
- **Tasks**: Data preparation, model training, evaluation, C++ implementation

## 📚 Related Work

### Bachelor Thesis
Title: Deep Learning Models for Window and Door Detection

**Scope**:
- Comparison of YOLO variants (v3, v4, v5, v8, v10, v11)
- EfficientDet, Faster R-CNN, SSD implementations
- Building façade element segmentation
- Performance analysis and optimization

### Publications & Reports
- Window/Door Detection Report
- YOLOv8+ Comparison Analysis  
- GPU Cluster Performance Study
- Deep Learning Best Practices

## 🤝 Collaboration & Attribution

### Advisors
- **Prof. Dr. Steffen Goebbels** - Hochschule Niederrhein
- Expertise: Computer Vision, Optimization, 3D Reconstruction

### Partners
- **City of Dortmund** - Surveying & Cadastral Office
- Application partner for building model projects

### Open Source Projects Used
- **Ultralytics YOLO** - Object detection framework
- **OpenCV** - Computer vision library
- **ONNX Runtime** - Model inference engine
- **PyTorch** - Deep learning framework

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ (for Fruit Quality)
- C++17 Compiler (for YOLOs-CPP)
- Git

### Clone & Explore
```bash
git clone https://github.com/yourusername/deep-learning-portfolio.git
cd deep-learning-portfolio

# Explore projects
cd yolos-cpp && cat README.md
cd ../fruit-quality && cat README.md
```

### Run Projects
Each project has its own setup guide. See individual README files:
- [YOLOs-CPP Setup](yolos-cpp/README.md#-quick-start)
- [Fruit Quality Setup](fruit-quality/README.md#-installation)

## 📈 Project Statistics

| Metric | YOLOs-CPP | Fruit Quality | Total |
|--------|-----------|---------------|-------|
| Lines of Code | ~500 | ~1000 | 1500+ |
| Languages | C++ | Python | 2 |
| Models | 3+ (YOLO variants) | 1 (YOLOv11) | 4+ |
| Test Cases | Inference modes | Class balancing | Multi-mode |
| Documentation | Comprehensive | Comprehensive | Complete |

## 🎯 Achievements

- ✅ Implemented production-grade C++ inference pipeline
- ✅ Trained deep learning models with 92%+ accuracy
- ✅ Real-time inference (30+ FPS)
- ✅ Cross-platform compatibility (Linux, macOS, Windows)
- ✅ Comprehensive documentation and examples
- ✅ Professional code quality (clean, optimized, tested)
- ✅ Model deployment in multiple formats (PyTorch, ONNX, TensorFlow)

## 📞 Contact & Support

- **Email**: wailchlih@gmail.com
- **Tel**: +4917624179500

## 📄 License

All projects are licensed under the MIT License - See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hochschule Niederrhein for educational support
- Prof. Dr. Steffen Goebbels for guidance and mentorship
- City of Dortmund for real-world application scenarios
- Open-source community for excellent tools and frameworks

---

**Last Updated**: April 2026  
**Version**: 1.0.0  
**Status**: Production Ready ✅

