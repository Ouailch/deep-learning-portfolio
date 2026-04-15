#!/bin/bash

# Fruit Quality Classification Setup Script

set -e

echo "=========================================="
echo "Fruit Quality Setup"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${YELLOW}Project Directory: ${PROJECT_DIR}${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found.${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} found${NC}"

# Create virtual environment
echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv "${PROJECT_DIR}/venv"
source "${PROJECT_DIR}/venv/bin/activate"

echo -e "${GREEN}✓ Virtual environment created${NC}"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r "${PROJECT_DIR}/requirements.txt"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}Error: Failed to install dependencies${NC}"
    exit 1
fi

# Download pre-trained model (optional)
echo ""
echo -e "${YELLOW}Setup Options:${NC}"
echo "1. Download pre-trained model (recommended)"
echo "2. Train new model"
echo "3. Skip (you can train/download later)"
echo ""
read -p "Choose option (1/2/3): " option

case $option in
    1)
        echo -e "${YELLOW}Downloading pre-trained model...${NC}"
        # Note: Replace with actual download URL
        echo "Model download functionality to be implemented"
        echo "For now, place model in: ${PROJECT_DIR}/models/"
        ;;
    2)
        echo -e "${YELLOW}Starting training...${NC}"
        echo "Ensure dataset is in: ${PROJECT_DIR}/fruit_quality_raw/"
        python scripts/train_model.py
        ;;
    3)
        echo -e "${YELLOW}Skipping model setup${NC}"
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        ;;
esac

echo ""
echo -e "${GREEN}=========================================="
echo "✓ Setup complete!"
echo "==========================================${NC}"

echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Activate environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run inference:"
echo "   python -c \"from scripts.inference import FruitQualityClassifier; classifier = FruitQualityClassifier('fruit_quality_model.pt'); classifier.predict_camera()\""
echo ""
echo "3. For more information, see README.md"
echo ""
