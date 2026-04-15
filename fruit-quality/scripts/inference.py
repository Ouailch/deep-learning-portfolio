"""
Fruit Quality Classification - Inference Pipeline

This script provides inference functions for the trained fruit quality model.
Supports single image, batch, and real-time video inference.

Author: Ouail Chlih
Date: 2025
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FruitQualityClassifier:
    """Fruit quality classification inference class."""
    
    def __init__(self, model_path, conf_threshold=0.5):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to trained YOLO model
            conf_threshold: Confidence threshold for predictions
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = ['Good', 'Bad', 'Rotten']
        logger.info(f"Model loaded from {model_path}")
    
    def predict_image(self, image_path):
        """
        Predict quality for single image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with prediction results
        """
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not read image from {image_path}")
            return None
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.predict(image_rgb, conf=self.conf_threshold)
        
        if results and len(results) > 0:
            pred = results[0]
            top_class = int(pred.probs.top1)
            confidence = float(pred.probs.top1conf)
            
            prediction = {
                'class': self.class_names[top_class],
                'class_id': top_class,
                'confidence': confidence,
                'image_path': image_path
            }
            
            logger.info(f"Predicted: {prediction['class']} (Confidence: {confidence:.2%})")
            return prediction
        
        return None
    
    def predict_batch(self, image_dir, file_extension='*.jpg'):
        """
        Predict quality for batch of images.
        
        Args:
            image_dir: Directory containing images
            file_extension: File extension to search for
        
        Returns:
            List of prediction results
        """
        image_dir = Path(image_dir)
        image_paths = list(image_dir.glob(file_extension))
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        predictions = []
        for idx, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing {idx}/{len(image_paths)}: {image_path.name}")
            pred = self.predict_image(str(image_path))
            if pred:
                predictions.append(pred)
        
        return predictions
    
    def predict_video(self, video_path, output_path=None):
        """
        Perform inference on video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
        
        Returns:
            Statistics dictionary
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {frame_width}x{frame_height} @ {fps} FPS ({total_frames} frames)")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        predictions = {class_name: 0 for class_name in self.class_names}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get prediction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model.predict(frame_rgb, conf=self.conf_threshold, verbose=False)
            
            if results and len(results) > 0:
                pred = results[0]
                top_class = int(pred.probs.top1)
                confidence = float(pred.probs.top1conf)
                class_name = self.class_names[top_class]
                
                predictions[class_name] += 1
                
                # Draw annotation
                text = f"{class_name}: {confidence:.2%}"
                color = self._get_color(top_class)
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           1.2, color, 2)
            
            if writer:
                writer.write(frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        if writer:
            writer.release()
            logger.info(f"Output video saved to {output_path}")
        
        stats = {
            'total_frames': frame_count,
            'predictions': predictions,
            'percentages': {k: f"{v/frame_count*100:.1f}%" for k, v in predictions.items()}
        }
        
        logger.info(f"Video processing complete! Statistics: {stats}")
        return stats
    
    def predict_camera(self, camera_id=0):
        """
        Real-time inference from webcam.
        
        Args:
            camera_id: Camera device ID (default: 0)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return
        
        logger.info("Camera opened. Press 'q' to quit, 's' to save frame")
        
        frame_count = 0
        predictions = {class_name: 0 for class_name in self.class_names}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inference
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model.predict(frame_rgb, conf=self.conf_threshold, verbose=False)
            
            if results and len(results) > 0:
                pred = results[0]
                top_class = int(pred.probs.top1)
                confidence = float(pred.probs.top1conf)
                class_name = self.class_names[top_class]
                
                predictions[class_name] += 1
                
                # Draw results
                text = f"{class_name}: {confidence:.2%}"
                color = self._get_color(top_class)
                cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                           1.5, color, 2)
            
            cv2.imshow('Fruit Quality Classification', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"fruit_capture_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                logger.info(f"Frame saved to {filename}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        logger.info(f"Session complete! Processed {frame_count} frames")
        logger.info(f"Statistics: {predictions}")
    
    @staticmethod
    def _get_color(class_id):
        """Get BGR color for class."""
        colors = {
            0: (0, 255, 0),      # Good - Green
            1: (0, 165, 255),    # Bad - Orange
            2: (0, 0, 255)       # Rotten - Red
        }
        return colors.get(class_id, (255, 255, 255))


def main():
    """Example usage."""
    
    # Initialize classifier
    classifier = FruitQualityClassifier('fruit_quality_model.pt', conf_threshold=0.5)
    
    # Single image prediction
    # result = classifier.predict_image('path/to/image.jpg')
    
    # Batch prediction
    # results = classifier.predict_batch('./fruit_quality_raw/test')
    
    # Video inference
    # stats = classifier.predict_video('input.mp4', 'output.mp4')
    
    # Real-time camera
    classifier.predict_camera(camera_id=0)


if __name__ == '__main__':
    main()
