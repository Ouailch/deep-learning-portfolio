"""
Fruit Quality Classification Training

This script trains a YOLOv11 model to classify fruit quality from images.
Uses transfer learning for efficient training on custom dataset.

Author: Ouail Chlih
Date: 2025
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FruitQualityDataset(Dataset):
    """Custom dataset for fruit quality classification."""
    
    def __init__(self, data_dir, transform=None, classes=None):
        """
        Args:
            data_dir: Path to dataset directory
            transform: Optional transforms to apply
            classes: List of class names
        """
        self.data_dir = Path(data_dir)
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        self.classes = classes or ['good', 'bad', 'rotten']
        self.images = []
        self.labels = []
        self._load_data()
    
    def _load_data(self):
        """Load image paths and labels from directory."""
        for class_idx, class_name in enumerate(self.classes):
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(str(img_path))
                    self.labels.append(class_idx)
                logger.info(f"Loaded {len([x for x in self.images if class_name in x])} images for class '{class_name}'")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """Get image and label at index."""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def train_yolo_model(data_yaml_path, epochs=200, batch_size=16, img_size=640):
    """
    Train YOLOv11 model for fruit quality classification.
    
    Args:
        data_yaml_path: Path to dataset YAML configuration
        epochs: Number of training epochs
        batch_size: Training batch size
        img_size: Input image size
    
    Returns:
        Trained model
    """
    logger.info("Initializing YOLOv11 model...")
    model = YOLO('yolov11n.pt')
    
    logger.info("Starting training...")
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=20,
        device=0,  # GPU device (0 for first GPU, -1 for CPU)
        workers=4,
        augment=True,
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        degrees=10,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        save=True,
        save_period=10,
        exist_ok=False,
        pretrained=True,
        optimizer='SGD',
        close_mosaic=10,
        resume=False,
    )
    
    logger.info("Training completed!")
    return model


def evaluate_model(model, val_dataset, batch_size=16):
    """
    Evaluate model on validation dataset.
    
    Args:
        model: Trained model
        val_dataset: Validation dataset
        batch_size: Batch size for evaluation
    
    Returns:
        Metrics dictionary
    """
    logger.info("Evaluating model...")
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    logger.info(f"Validation Accuracy: {accuracy:.2f}%")
    
    return {'accuracy': accuracy, 'correct': correct, 'total': total}


def predict_single_image(model, image_path, class_names):
    """
    Make prediction on single image.
    
    Args:
        model: Trained model
        image_path: Path to image file
        class_names: List of class names
    
    Returns:
        Prediction result with confidence
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    
    results = model.predict(image, conf=0.5)
    
    if results:
        top_pred = results[0]
        class_id = int(top_pred.probs.top1)
        confidence = float(top_pred.probs.top1conf)
        class_name = class_names[class_id]
        
        logger.info(f"Prediction: {class_name} (Confidence: {confidence:.2f})")
        return {'class': class_name, 'confidence': confidence}
    
    return None


def create_data_yaml(data_dir, output_path='data.yaml'):
    """
    Create YAML configuration for dataset.
    
    Args:
        data_dir: Root data directory
        output_path: Output YAML file path
    """
    yaml_content = f"""
path: {data_dir}
train: train
val: val
test: test

nc: 3
names: ['good', 'bad', 'rotten']
"""
    
    with open(output_path, 'w') as f:
        f.write(yaml_content.strip())
    
    logger.info(f"Data configuration saved to {output_path}")


def main():
    """Main training pipeline."""
    
    # Configuration
    data_dir = './fruit_quality_raw'
    epochs = 200
    batch_size = 16
    img_size = 640
    
    logger.info("=" * 50)
    logger.info("Fruit Quality Classification Training")
    logger.info("=" * 50)
    
    # Create data configuration
    create_data_yaml(data_dir)
    
    # Train model
    model = train_yolo_model('data.yaml', epochs=epochs, batch_size=batch_size)
    
    # Evaluate
    val_dataset = FruitQualityDataset(f'{data_dir}/val')
    metrics = evaluate_model(model, val_dataset, batch_size=batch_size)
    
    # Save model
    model.save('fruit_quality_model.pt')
    logger.info("Model saved to fruit_quality_model.pt")
    
    # Export to ONNX
    model.export(format='onnx')
    logger.info("Model exported to ONNX format")
    
    logger.info("=" * 50)
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 50)


if __name__ == '__main__':
    main()
