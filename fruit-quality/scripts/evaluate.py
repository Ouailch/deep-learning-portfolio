"""
Fruit Quality Classification - Model Evaluation

Comprehensive evaluation metrics, confusion matrix, and reporting.

Author: Ouail Chlih
Date: 2025
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation toolkit."""
    
    def __init__(self, model_path, class_names=None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            class_names: List of class names
        """
        self.model = YOLO(model_path)
        self.class_names = class_names or ['Good', 'Bad', 'Rotten']
        logger.info(f"Model loaded: {model_path}")
    
    def evaluate_dataset(self, test_dir, conf_threshold=0.5):
        """
        Evaluate model on entire test dataset.
        
        Args:
            test_dir: Directory with test images
            conf_threshold: Confidence threshold
        
        Returns:
            Metrics dictionary
        """
        test_dir = Path(test_dir)
        
        y_true = []
        y_pred = []
        
        # Process each class directory
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = test_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Directory not found: {class_dir}")
                continue
            
            for img_path in class_dir.glob('*.jpg'):
                # Ground truth
                y_true.append(class_idx)
                
                # Prediction
                results = self.model.predict(str(img_path), conf=conf_threshold, verbose=False)
                if results and len(results) > 0:
                    pred_class = int(results[0].probs.top1)
                    y_pred.append(pred_class)
                else:
                    y_pred.append(-1)  # Failed prediction
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred)
        
        logger.info("\n" + "="*50)
        logger.info("EVALUATION METRICS")
        logger.info("="*50)
        
        for key, value in metrics.items():
            if isinstance(value, dict):
                logger.info(f"\n{key}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v:.4f}")
            elif isinstance(value, (int, float)):
                logger.info(f"{key}: {value:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics."""
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        }
        
        # Per-class metrics
        per_class = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        metrics['per_class'] = per_class
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.class_names)))
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def plot_confusion_matrix(self, metrics, save_path='confusion_matrix.png'):
        """
        Plot and save confusion matrix.
        
        Args:
            metrics: Metrics dictionary from evaluate_dataset
            save_path: Path to save plot
        """
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"Confusion matrix saved to: {save_path}")
        plt.close()
    
    def plot_metrics(self, metrics, save_path='metrics.png'):
        """
        Plot comprehensive metrics.
        
        Args:
            metrics: Metrics dictionary
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Metrics', fontsize=16)
        
        # Accuracy
        ax = axes[0, 0]
        ax.bar(['Accuracy'], [metrics['accuracy']], color='skyblue')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Score')
        ax.set_title('Overall Accuracy')
        
        # Per-class F1 scores
        ax = axes[0, 1]
        f1_scores = [metrics['per_class'][name]['f1-score'] for name in self.class_names]
        ax.bar(self.class_names, f1_scores, color='lightcoral')
        ax.set_ylim([0, 1])
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Scores')
        
        # Per-class Precision
        ax = axes[1, 0]
        precisions = [metrics['per_class'][name]['precision'] for name in self.class_names]
        ax.bar(self.class_names, precisions, color='lightgreen')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Precision')
        ax.set_title('Per-Class Precision')
        
        # Per-class Recall
        ax = axes[1, 1]
        recalls = [metrics['per_class'][name]['recall'] for name in self.class_names]
        ax.bar(self.class_names, recalls, color='lightyellow')
        ax.set_ylim([0, 1])
        ax.set_ylabel('Recall')
        ax.set_title('Per-Class Recall')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        logger.info(f"Metrics plot saved to: {save_path}")
        plt.close()
    
    def generate_report(self, test_dir, output_file='evaluation_report.txt'):
        """
        Generate comprehensive evaluation report.
        
        Args:
            test_dir: Test dataset directory
            output_file: Path to save report
        """
        metrics = self.evaluate_dataset(test_dir)
        
        # Build report
        report = []
        report.append("="*60)
        report.append("FRUIT QUALITY CLASSIFICATION - EVALUATION REPORT")
        report.append("="*60)
        report.append("")
        
        # Overview
        report.append("OVERALL METRICS")
        report.append("-" * 60)
        report.append(f"Accuracy:           {metrics['accuracy']:.4f}")
        report.append(f"Macro Precision:    {metrics['macro_precision']:.4f}")
        report.append(f"Macro Recall:       {metrics['macro_recall']:.4f}")
        report.append(f"Macro F1 Score:     {metrics['macro_f1']:.4f}")
        report.append("")
        
        # Per-class metrics
        report.append("PER-CLASS METRICS")
        report.append("-" * 60)
        for class_name in self.class_names:
            class_metrics = metrics['per_class'][class_name]
            report.append(f"\n{class_name}:")
            report.append(f"  Precision: {class_metrics['precision']:.4f}")
            report.append(f"  Recall:    {class_metrics['recall']:.4f}")
            report.append(f"  F1-Score:  {class_metrics['f1-score']:.4f}")
            report.append(f"  Support:   {int(class_metrics['support'])}")
        
        # Confusion matrix
        report.append("\n" + "="*60)
        report.append("CONFUSION MATRIX")
        report.append("-" * 60)
        cm = metrics['confusion_matrix']
        report.append(f"{'':15} " + " ".join(f"{name:>10}" for name in self.class_names))
        for i, class_name in enumerate(self.class_names):
            row_str = f"{class_name:>15} "
            row_str += " ".join(f"{cm[i][j]:>10}" for j in range(len(self.class_names)))
            report.append(row_str)
        
        report.append("")
        report.append("="*60)
        report.append(f"Report generated at: {Path.cwd()}")
        report.append("="*60)
        
        # Save report
        report_text = "\n".join(report)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        # Print and log
        print(report_text)
        logger.info(f"Report saved to: {output_file}")
        
        return metrics


def main():
    """Example usage."""
    evaluator = ModelEvaluator('fruit_quality_model.pt')
    
    # Evaluate
    metrics = evaluator.evaluate_dataset('./fruit_quality_raw/test')
    
    # Generate visualizations
    evaluator.plot_confusion_matrix(metrics)
    evaluator.plot_metrics(metrics)
    
    # Generate report
    evaluator.generate_report('./fruit_quality_raw/test')


if __name__ == '__main__':
    main()
