#!/bin/bash

# YOLOs-CPP Build Script
# Automates CMake build process

set -e

echo "=========================================="
echo "YOLOs-CPP Build Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
ONNXRUNTIME_DIR="${PROJECT_DIR}/onnxruntime-linux-x64-1.20.0"

echo -e "${YELLOW}Project Directory: ${PROJECT_DIR}${NC}"
echo -e "${YELLOW}Build Directory: ${BUILD_DIR}${NC}"
echo -e "${YELLOW}ONNX Runtime: ${ONNXRUNTIME_DIR}${NC}"

# Check if build directory exists
if [ -d "$BUILD_DIR" ]; then
    echo -e "${YELLOW}Cleaning old build directory...${NC}"
    rm -rf "$BUILD_DIR"
fi

# Create build directory
echo -e "${YELLOW}Creating build directory...${NC}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: CMake not found. Please install CMake 3.15 or higher.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ CMake found: $(cmake --version | head -n1)${NC}"

# Check C++ compiler
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo -e "${RED}Error: C++ compiler not found. Please install GCC or Clang.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ C++ Compiler found${NC}"

# Check OpenCV
pkg-config --modversion opencv4 > /dev/null 2>&1 || pkg-config --modversion opencv > /dev/null 2>&1
if [ $? -eq 0 ]; then
    OPENCV_VERSION=$(pkg-config --modversion opencv4 2>/dev/null || pkg-config --modversion opencv)
    echo -e "${GREEN}✓ OpenCV found: ${OPENCV_VERSION}${NC}"
else
    echo -e "${RED}Error: OpenCV not found. Please install libopencv-dev.${NC}"
    exit 1
fi

# Check ONNX Runtime
if [ ! -d "$ONNXRUNTIME_DIR" ]; then
    echo -e "${RED}Error: ONNX Runtime not found at ${ONNXRUNTIME_DIR}${NC}"
    echo -e "${YELLOW}Please download it from:${NC}"
    echo "https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0"
    exit 1
fi
echo -e "${GREEN}✓ ONNX Runtime found${NC}"

# Run CMake
echo -e "${YELLOW}Running CMake...${NC}"
cmake -DONNXRUNTIME_DIR="${ONNXRUNTIME_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-O3 -march=native" \
      "$PROJECT_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed.${NC}"
    exit 1
fi

# Build
echo -e "${YELLOW}Building...${NC}"
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed.${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================="
echo "✓ Build successful!"
echo "==========================================${NC}"

# Copy libraries
echo -e "${YELLOW}Copying runtime libraries...${NC}"
cp "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so" "${BUILD_DIR}/bin/" 2>/dev/null || true

# Summary
echo ""
echo -e "${GREEN}Build Summary:${NC}"
echo "  Executables: ${BUILD_DIR}/bin/"
echo "    - image_inference"
echo "    - video_inference"
echo "    - camera_inference"
echo ""
echo -e "${YELLOW}Usage:${NC}"
echo "  ./bin/image_inference ../data/test_image.jpg"
echo "  ./bin/video_inference ../data/test_video.mp4"
echo "  ./bin/camera_inference 0"
echo ""
