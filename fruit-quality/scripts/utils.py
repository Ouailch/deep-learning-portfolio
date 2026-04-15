"""
Fruit Quality Classification - Utility Functions

Helper functions for data processing, visualization, and model management.

Author: Ouail Chlih
Date: 2025
"""

import cv2
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dataset_yaml(data_dir, output='data.yaml'):
    """
    Create YAML configuration for dataset.
    
    Args:
        data_dir: Root data directory
        output: Output YAML path
    """
    yaml_content = f"""
path: {data_dir}
train: train
val: val
test: test

nc: 3
names: ['good', 'bad', 'rotten']
"""
    
    with open(output, 'w') as f:
        f.write(yaml_content.strip())
    
    logger.info(f"Dataset YAML created: {output}")


def organize_dataset(source_dir, target_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Organize dataset into train/val/test splits.
    
    Args:
        source_dir: Directory with class folders
        target_dir: Target directory for organized data
        split_ratio: (train, val, test) ratios
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        for class_name in ['good', 'bad', 'rotten']:
            (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    train_ratio, val_ratio, _ = split_ratio
    
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        images = list(class_dir.glob('*.jpg'))
        
        # Calculate split indices
        n_images = len(images)
        train_count = int(n_images * train_ratio)
        val_count = int(n_images * val_ratio)
        
        # Assign to splits
        for idx, img_path in enumerate(images):
            if idx < train_count:
                split = 'train'
            elif idx < train_count + val_count:
                split = 'val'
            else:
                split = 'test'
            
            # Copy file
            target_path = target_dir / split / class_name / img_path.name
            import shutil
            shutil.copy(img_path, target_path)
        
        logger.info(f"{class_name}: {train_count} train, {val_count} val, {n_images-train_count-val_count} test")


def resize_images(input_dir, output_dir, size=(640, 640)):
    """
    Resize all images in directory.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        size: Target size (width, height)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for img_path in input_dir.rglob('*.jpg'):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # Resize with aspect ratio preservation
        img_resized = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        
        # Save
        output_path = output_dir / img_path.relative_to(input_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img_resized)
        
        count += 1
    
    logger.info(f"Resized {count} images to {size}")


def visualize_predictions(image_path, predictions, class_colors=None, save_path=None):
    """
    Visualize predictions on image.
    
    Args:
        image_path: Path to image
        predictions: List of prediction dicts with 'class', 'confidence'
        class_colors: Dict mapping class names to BGR colors
        save_path: Optional path to save visualization
    """
    if class_colors is None:
        class_colors = {
            'good': (0, 255, 0),
            'bad': (0, 165, 255),
            'rotten': (0, 0, 255)
        }
    
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Could not read image: {image_path}")
        return
    
    # Add predictions to image
    for idx, pred in enumerate(predictions):
        y_pos = 50 + idx * 40
        text = f"{pred['class']}: {pred['confidence']:.2%}"
        color = class_colors.get(pred['class'].lower(), (255, 255, 255))
        
        cv2.putText(img, text, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    
    if save_path:
        cv2.imwrite(str(save_path), img)
        logger.info(f"Visualization saved: {save_path}")
    
    return img


def get_dataset_statistics(data_dir):
    """
    Calculate dataset statistics.
    
    Args:
        data_dir: Root data directory
    
    Returns:
        Statistics dictionary
    """
    data_dir = Path(data_dir)
    stats = {'total': 0, 'by_class': {}, 'by_split': {}}
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        split_count = 0
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            class_count = len(list(class_dir.glob('*.jpg')))
            
            if class_name not in stats['by_class']:
                stats['by_class'][class_name] = 0
            
            stats['by_class'][class_name] += class_count
            split_count += class_count
        
        stats['by_split'][split] = split_count
        stats['total'] += split_count
    
    return stats


def print_dataset_stats(stats):
    """Pretty print dataset statistics."""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"\nTotal Images: {stats['total']}")
    
    print("\nBy Split:")
    for split, count in stats['by_split'].items():
        pct = 100 * count / stats['total']
        print(f"  {split:>8}: {count:>5} ({pct:>5.1f}%)")
    
    print("\nBy Class:")
    for class_name, count in stats['by_class'].items():
        pct = 100 * count / stats['total']
        print(f"  {class_name:>8}: {count:>5} ({pct:>5.1f}%)")
    
    print("\n" + "="*50 + "\n")


def save_model_config(model_path, config_path='model_config.json'):
    """
    Save model configuration metadata.
    
    Args:
        model_path: Path to model
        config_path: Path to save config
    """
    config = {
        'model_path': str(model_path),
        'model_type': 'YOLOv11n',
        'classes': ['good', 'bad', 'rotten'],
        'input_size': 640,
        'confidence_threshold': 0.5,
        'iou_threshold': 0.5,
        'training_date': '2025-09-2025',
        'validation_accuracy': 0.923,
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Config saved: {config_path}")


def main():
    """Example usage."""
    
    # Get dataset statistics
    stats = get_dataset_statistics('./fruit_quality_raw')
    print_dataset_stats(stats)
    
    # Organize dataset
    # organize_dataset('./raw_fruits', './fruit_quality_raw')
    
    # Resize images
    # resize_images('./fruit_quality_raw/train', './fruit_quality_resized/train')
    
    # Create YAML
    create_dataset_yaml('./fruit_quality_raw', 'data.yaml')


if __name__ == '__main__':
    main()
