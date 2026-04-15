# YOLOs-CPP Dependencies

## Required

### CMake
- Version: 3.15 or higher
- Download: https://cmake.org/download/

### Compiler
- GCC 7+ (Linux)
- Clang 5+ (macOS)
- MSVC 2019+ (Windows)

### OpenCV
- Version: 4.5 or higher
- Installation:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install libopencv-dev

  # macOS
  brew install opencv

  # Windows
  # Download from https://opencv.org/releases/
  ```

### ONNX Runtime
- Version: 1.20.0
- Download:
  ```bash
  # Linux
  wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-linux-x64-1.20.0.tgz
  tar -xzf onnxruntime-linux-x64-1.20.0.tgz

  # macOS (ARM64)
  wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.0/onnxruntime-osx-arm64-1.20.0.tgz

  # Windows
  # Download from releases page
  ```

## Optional

### CUDA (for GPU acceleration)
- NVIDIA CUDA Toolkit 11.0+
- cuDNN 8.0+
- Download: https://developer.nvidia.com/cuda-downloads

### TensorRT
- For optimized inference on NVIDIA GPUs
- Version: 8.0+

## Verify Installation

```bash
cd yolos-cpp
mkdir build && cd build
cmake -DONNXRUNTIME_DIR="../onnxruntime-linux-x64-1.20.0" ..
make -j$(nproc)

# Test
./image_inference ../data/test_image.jpg
```
