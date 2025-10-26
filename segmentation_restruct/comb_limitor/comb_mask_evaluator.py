"""
CombMaskEvaluator: Evaluate binary comb mask predictions against ground truth.

This module provides comprehensive evaluation metrics for binary segmentation tasks,
comparing predicted comb masks against ground truth annotations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


class CombMaskEvaluator:
    """
    Evaluate binary comb mask predictions against ground truth labels.

    This class computes various performance metrics for binary segmentation,
    including accuracy, precision, recall, F1-score, IoU, and confusion matrix
    components (TP, FP, TN, FN).

    Designed for evaluating comb segmentation where:
    - 0 = background/non-comb pixels
    - 1 (or 255) = comb pixels
    """

    def __init__(self, threshold: Optional[int] = None):
        """
        Initialize the CombMaskEvaluator.

        Parameters
        ----------
        threshold : Optional[int], default=None
            Threshold for binarizing masks. Pixels > threshold are considered
            as comb (class 1), pixels <= threshold as background (class 0).

            If None (default), automatically detects the format:
            - If max value > 1, assumes 0/255 format and uses threshold=127
            - If max value <= 1, assumes 0/1 format and uses threshold=0

            Manual threshold examples:
            - threshold=0: For 0/1 masks (ground truth format)
            - threshold=127: For 0/255 masks (standard image format)

        Example
        -------
        >>> # Auto-detect format (recommended)
        >>> evaluator = CombMaskEvaluator()
        >>>
        >>> # Manual threshold for 0/1 ground truth
        >>> evaluator = CombMaskEvaluator(threshold=0)
        >>>
        >>> # Manual threshold for 0/255 predictions
        >>> evaluator = CombMaskEvaluator(threshold=127)
        """
        self.threshold = threshold
        self.auto_threshold = threshold is None

    def evaluate(
        self,
        predicted_mask: np.ndarray,
        ground_truth_mask: np.ndarray,
        return_confusion_matrix: bool = True,
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics for a single mask pair.

        Parameters
        ----------
        predicted_mask : np.ndarray
            Predicted binary mask (H, W) with values 0/1 or 0/255.
        ground_truth_mask : np.ndarray
            Ground truth binary mask (H, W) with values 0/1 or 0/255.
        return_confusion_matrix : bool, default=True
            If True, includes TP, TN, FP, FN counts in the results.

        Returns
        -------
        Dict[str, float]
            Dictionary containing evaluation metrics:
            - 'accuracy': Overall pixel accuracy (TP + TN) / (TP + TN + FP + FN)
            - 'precision': TP / (TP + FP) - how many predicted comb pixels are correct
            - 'recall': TP / (TP + FN) - how many actual comb pixels were found
            - 'f1_score': 2 * (precision * recall) / (precision + recall)
            - 'iou': Intersection over Union - TP / (TP + FP + FN)
            - 'dice': Dice coefficient - 2*TP / (2*TP + FP + FN) [same as F1-score]
            - 'specificity': TN / (TN + FP) - true negative rate
            - 'tp': True Positives (if return_confusion_matrix=True)
            - 'tn': True Negatives (if return_confusion_matrix=True)
            - 'fp': False Positives (if return_confusion_matrix=True)
            - 'fn': False Negatives (if return_confusion_matrix=True)

        Notes
        -----
        All metrics are in range [0, 1] except counts (TP, TN, FP, FN).

        The method automatically handles different mask formats:
        - Ground truth: 0/1 format (from CVAT labels)
        - Predictions: 0/255 format (from generate_comb_mask)
        Both are normalized to 0/1 before comparison.

        Definitions:
        - TP (True Positive): Correctly predicted comb pixels
        - TN (True Negative): Correctly predicted background pixels
        - FP (False Positive): Background pixels incorrectly predicted as comb
        - FN (False Negative): Comb pixels incorrectly predicted as background

        Example
        -------
        >>> evaluator = CombMaskEvaluator()
        >>> # Works with both 0/1 and 0/255 masks automatically
        >>> metrics = evaluator.evaluate(predicted_mask, gt_mask)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        >>> print(f"IoU: {metrics['iou']:.3f}")
        >>> print(f"F1 Score: {metrics['f1_score']:.3f}")
        """
        # Binarize masks (handles both 0/1 and 0/255 formats)
        pred_binary = self._binarize(predicted_mask)
        gt_binary = self._binarize(ground_truth_mask)

        # Ensure same shape
        if pred_binary.shape != gt_binary.shape:
            raise ValueError(
                f"Shape mismatch: predicted {pred_binary.shape} vs ground truth {gt_binary.shape}"
            )

        # Convert to torch tensors for efficient computation
        pred_tensor = torch.from_numpy(pred_binary).float()
        gt_tensor = torch.from_numpy(gt_binary).float()

        # Compute confusion matrix components
        tp = torch.sum((pred_tensor == 1) & (gt_tensor == 1)).item()
        tn = torch.sum((pred_tensor == 0) & (gt_tensor == 0)).item()
        fp = torch.sum((pred_tensor == 1) & (gt_tensor == 0)).item()
        fn = torch.sum((pred_tensor == 0) & (gt_tensor == 1)).item()

        # Compute metrics with epsilon to avoid division by zero
        eps = 1e-7

        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1_score = 2 * (precision * recall) / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        dice = 2 * tp / (2 * tp + fp + fn + eps)
        specificity = tn / (tn + fp + eps)

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "iou": float(iou),
            "dice": float(dice),
            "specificity": float(specificity),
        }

        if return_confusion_matrix:
            metrics.update(
                {
                    "tp": int(tp),
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                }
            )

        return metrics

    def evaluate_batch_from_arrays(
        self,
        predicted_masks: List[np.ndarray],
        ground_truth_masks: List[np.ndarray],
        image_names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Evaluate multiple mask pairs from in-memory arrays (no file I/O).

        This method is useful when you already have masks in memory and don't
        want to save/load files. Perfect for small datasets or notebooks.

        Parameters
        ----------
        predicted_masks : List[np.ndarray]
            List of predicted masks (each H x W).
        ground_truth_masks : List[np.ndarray]
            List of ground truth masks (each H x W).
        image_names : Optional[List[str]], default=None
            Optional list of image names for reference in results.
            If None, uses indices: "image_0", "image_1", etc.

        Returns
        -------
        Tuple[Dict[str, float], List[Dict[str, float]]]
            - Aggregate metrics (mean, std, min, max) across all images
            - List of per-image metrics with 'filename' field

        Raises
        ------
        ValueError
            If predicted_masks and ground_truth_masks have different lengths.

        Example
        -------
        >>> # Generate predictions for multiple images
        >>> pred_masks = [generator.generate_comb_mask(seg) for seg in segmentations]
        >>> gt_masks = [cv2.imread(f, 0) for f in gt_files]
        >>> names = [f.name for f in gt_files]
        >>>
        >>> # Evaluate all at once
        >>> evaluator = CombMaskEvaluator()
        >>> aggregate, per_image = evaluator.evaluate_batch_from_arrays(
        ...     pred_masks, gt_masks, names
        ... )
        >>> print(f"Mean IoU: {aggregate['iou_mean']:.3f}")
        """
        if len(predicted_masks) != len(ground_truth_masks):
            raise ValueError(
                f"Number of predicted masks ({len(predicted_masks)}) must match "
                f"number of ground truth masks ({len(ground_truth_masks)})"
            )

        if image_names is None:
            image_names = [f"image_{i}" for i in range(len(predicted_masks))]
        elif len(image_names) != len(predicted_masks):
            raise ValueError(
                f"Number of image names ({len(image_names)}) must match "
                f"number of masks ({len(predicted_masks)})"
            )

        per_image_metrics = []

        for pred_mask, gt_mask, name in zip(
            predicted_masks, ground_truth_masks, image_names
        ):
            # Evaluate this pair
            metrics = self.evaluate(pred_mask, gt_mask, return_confusion_matrix=True)
            metrics["filename"] = name
            per_image_metrics.append(metrics)

        # Compute aggregate statistics
        aggregate_metrics = self._compute_aggregate_metrics(per_image_metrics)

        return aggregate_metrics, per_image_metrics

    def evaluate_batch(
        self,
        predicted_dir: Union[str, Path],
        ground_truth_dir: Union[str, Path],
        file_extension: str = "*.png",
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Evaluate multiple mask pairs and compute aggregate statistics.

        Parameters
        ----------
        predicted_dir : Union[str, Path]
            Directory containing predicted mask files.
        ground_truth_dir : Union[str, Path]
            Directory containing ground truth mask files.
        file_extension : str, default="*.png"
            File pattern to match (e.g., "*.png", "*.jpg").

        Returns
        -------
        Tuple[Dict[str, float], List[Dict[str, float]]]
            - Aggregate metrics (mean, std) across all images
            - List of per-image metrics

        Example
        -------
        >>> evaluator = CombMaskEvaluator()
        >>> aggregate, per_image = evaluator.evaluate_batch(
        ...     predicted_dir='output/predicted_masks/',
        ...     ground_truth_dir='data/ground_truth/',
        ... )
        >>> print(f"Mean IoU: {aggregate['iou_mean']:.3f} ± {aggregate['iou_std']:.3f}")
        >>> print(f"Mean F1: {aggregate['f1_score_mean']:.3f}")
        """
        pred_dir = Path(predicted_dir)
        gt_dir = Path(ground_truth_dir)

        # Get all prediction files
        pred_files = sorted(pred_dir.glob(file_extension))

        if not pred_files:
            raise ValueError(f"No files found matching {file_extension} in {pred_dir}")

        per_image_metrics = []

        for pred_file in pred_files:
            # Find corresponding ground truth file
            gt_file = gt_dir / pred_file.name

            if not gt_file.exists():
                print(
                    f"Warning: Ground truth not found for {pred_file.name}, skipping..."
                )
                continue

            # Load masks
            pred_mask = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)

            if pred_mask is None or gt_mask is None:
                print(f"Warning: Failed to load {pred_file.name}, skipping...")
                continue

            # Evaluate
            metrics = self.evaluate(pred_mask, gt_mask)
            metrics["filename"] = pred_file.name
            per_image_metrics.append(metrics)

        # Compute aggregate statistics
        aggregate_metrics = self._compute_aggregate_metrics(per_image_metrics)

        return aggregate_metrics, per_image_metrics

    def visualize_prediction(
        self,
        predicted_mask: np.ndarray,
        ground_truth_mask: np.ndarray,
        input_image: Optional[np.ndarray] = None,
        show_metrics: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize prediction, ground truth, and error map side-by-side.

        Parameters
        ----------
        predicted_mask : np.ndarray
            Predicted binary mask (H, W).
        ground_truth_mask : np.ndarray
            Ground truth binary mask (H, W).
        input_image : Optional[np.ndarray], default=None
            Original input image (H, W) or (H, W, 3).
        show_metrics : bool, default=True
            If True, display metrics in the plot title.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Matplotlib figure and axes objects.

        Example
        -------
        >>> evaluator = CombMaskEvaluator()
        >>> fig, axes = evaluator.visualize_prediction(pred_mask, gt_mask, input_img)
        >>> plt.show()
        """
        # Binarize masks
        pred_binary = self._binarize(predicted_mask)
        gt_binary = self._binarize(ground_truth_mask)

        # Create error map: TP (green), FP (red), FN (blue), TN (black)
        error_map = self._create_error_map(pred_binary, gt_binary)

        # Determine subplot layout
        num_plots = 3 + (1 if input_image is not None else 0)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

        idx = 0

        # Plot input image if provided
        if input_image is not None:
            axes[idx].imshow(
                input_image, cmap="gray" if len(input_image.shape) == 2 else None
            )
            axes[idx].set_title("Input Image", fontsize=12)
            axes[idx].axis("off")
            idx += 1

        # Plot ground truth
        axes[idx].imshow(gt_binary, cmap="gray")
        axes[idx].set_title("Ground Truth", fontsize=12)
        axes[idx].axis("off")
        idx += 1

        # Plot prediction
        axes[idx].imshow(pred_binary, cmap="gray")
        axes[idx].set_title("Prediction", fontsize=12)
        axes[idx].axis("off")
        idx += 1

        # Plot error map
        axes[idx].imshow(error_map)
        axes[idx].set_title("Error Map\n(Green=TP, Red=FP, Blue=FN)", fontsize=12)
        axes[idx].axis("off")

        # Add metrics to title if requested
        if show_metrics:
            metrics = self.evaluate(
                predicted_mask, ground_truth_mask, return_confusion_matrix=False
            )
            title = (
                f"Accuracy: {metrics['accuracy']:.3f} | "
                f"IoU: {metrics['iou']:.3f} | "
                f"F1: {metrics['f1_score']:.3f} | "
                f"Precision: {metrics['precision']:.3f} | "
                f"Recall: {metrics['recall']:.3f}"
            )
            fig.suptitle(title, fontsize=14, y=1.02)

        plt.tight_layout()
        return fig, axes

    def visualize_batch(
        self,
        predicted_masks: List[np.ndarray],
        ground_truth_masks: List[np.ndarray],
        image_names: Optional[List[str]] = None,
        input_images: Optional[List[np.ndarray]] = None,
        max_cols: int = 2,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize multiple prediction/GT pairs in a grid layout.

        Creates a grid showing input image (optional), ground truth, prediction,
        and error map for each image in the batch.

        Parameters
        ----------
        predicted_masks : List[np.ndarray]
            List of predicted masks.
        ground_truth_masks : List[np.ndarray]
            List of ground truth masks.
        image_names : Optional[List[str]], default=None
            Optional list of image names for subplot titles.
        input_images : Optional[List[np.ndarray]], default=None
            Optional list of input images to display.
        max_cols : int, default=2
            Maximum number of image sets per row.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Figure and array of axes.

        Example
        -------
        >>> fig, axes = evaluator.visualize_batch(
        ...     predicted_masks, gt_masks, image_names, input_images
        ... )
        >>> plt.savefig('batch_evaluation.png', dpi=150, bbox_inches='tight')
        >>> plt.show()
        """
        n_images = len(predicted_masks)

        if image_names is None:
            image_names = [f"Image {i+1}" for i in range(n_images)]

        # Determine subplot layout
        show_input = input_images is not None
        n_cols_per_image = 4 if show_input else 3
        n_cols = min(max_cols, n_images) * n_cols_per_image
        n_rows = (n_images + max_cols - 1) // max_cols

        fig = plt.figure(
            figsize=(n_cols_per_image * 4 * min(max_cols, n_images), 4 * n_rows)
        )

        for idx in range(n_images):
            pred_mask = predicted_masks[idx]
            gt_mask = ground_truth_masks[idx]
            name = image_names[idx]

            # Calculate base position in grid
            row = idx // max_cols
            col = idx % max_cols
            base_idx = row * max_cols * n_cols_per_image + col * n_cols_per_image + 1

            # Binarize masks
            pred_binary = self._binarize(pred_mask)
            gt_binary = self._binarize(gt_mask)
            error_map = self._create_error_map(pred_binary, gt_binary)

            # Compute metrics for this pair
            metrics = self.evaluate(pred_mask, gt_mask, return_confusion_matrix=False)

            subplot_idx = base_idx

            # Plot input image if provided
            if show_input:
                ax = plt.subplot(n_rows, n_cols, subplot_idx)
                input_img = input_images[idx]
                ax.imshow(input_img, cmap="gray" if len(input_img.shape) == 2 else None)
                ax.set_title(f"{name}\nInput", fontsize=10)
                ax.axis("off")
                subplot_idx += 1

            # Plot GT
            ax = plt.subplot(n_rows, n_cols, subplot_idx)
            ax.imshow(gt_binary, cmap="gray", vmin=0, vmax=1)
            ax.set_title(
                f"{name}\nGround Truth" if not show_input else "GT", fontsize=10
            )
            ax.axis("off")
            subplot_idx += 1

            # Plot prediction
            ax = plt.subplot(n_rows, n_cols, subplot_idx)
            ax.imshow(pred_binary, cmap="gray", vmin=0, vmax=1)
            ax.set_title("Prediction", fontsize=10)
            ax.axis("off")
            subplot_idx += 1

            # Plot error map
            ax = plt.subplot(n_rows, n_cols, subplot_idx)
            ax.imshow(error_map)
            ax.set_title(f'Error Map\nIoU={metrics["iou"]:.3f}', fontsize=10)
            ax.axis("off")

        plt.suptitle(
            "Batch Evaluation Results (Green=TP, Red=FP, Blue=FN)", fontsize=14, y=0.995
        )
        plt.tight_layout()

        return fig, plt.gcf().axes

    def _binarize(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert mask to binary (0/1) format.

        Automatically detects mask format and applies appropriate threshold:
        - If max value > 1: assumes 0/255 format, uses threshold
        - If max value <= 1: assumes 0/1 format, already binary
        """
        # Check if mask is already binary (0/1)
        max_val = mask.max()

        if max_val <= 1:
            # Already in 0/1 format (ground truth)
            return mask.astype(np.uint8)
        else:
            # In 0/255 or similar format (prediction)
            if self.auto_threshold:
                # Auto-detect: use middle value
                threshold = max_val // 2
            else:
                threshold = self.threshold

            return (mask > threshold).astype(np.uint8)

    def _create_error_map(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Create RGB error visualization map.

        Green: True Positive (correct comb)
        Red: False Positive (predicted comb, actually background)
        Blue: False Negative (predicted background, actually comb)
        Black: True Negative (correct background)
        """
        h, w = pred.shape
        error_map = np.zeros((h, w, 3), dtype=np.uint8)

        # TP: Green
        tp_mask = (pred == 1) & (gt == 1)
        error_map[tp_mask] = [0, 255, 0]

        # FP: Red
        fp_mask = (pred == 1) & (gt == 0)
        error_map[fp_mask] = [255, 0, 0]

        # FN: Blue
        fn_mask = (pred == 0) & (gt == 1)
        error_map[fn_mask] = [0, 0, 255]

        # TN: Black (already 0)

        return error_map

    def _compute_aggregate_metrics(
        self, per_image_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute mean and std for all metrics across images, including confusion matrix totals."""
        if not per_image_metrics:
            return {}

        # Extract metric names (exclude filename, but INCLUDE confusion matrix counts for totals)
        metric_names = [k for k in per_image_metrics[0].keys() if k != "filename"]

        aggregate = {}

        for metric_name in metric_names:
            values = [m[metric_name] for m in per_image_metrics]

            # For confusion matrix counts (tp, tn, fp, fn), compute sum and mean
            if metric_name in ["tp", "tn", "fp", "fn"]:
                aggregate[f"{metric_name}_total"] = int(np.sum(values))
                aggregate[f"{metric_name}_mean"] = float(np.mean(values))
            else:
                # For percentage metrics, compute mean, std, min, max
                aggregate[f"{metric_name}_mean"] = float(np.mean(values))
                aggregate[f"{metric_name}_std"] = float(np.std(values))
                aggregate[f"{metric_name}_min"] = float(np.min(values))
                aggregate[f"{metric_name}_max"] = float(np.max(values))

        aggregate["num_images"] = len(per_image_metrics)

        return aggregate

    def print_metrics(
        self, metrics: Dict[str, float], title: str = "Evaluation Metrics"
    ) -> None:
        """
        Pretty print evaluation metrics.

        Parameters
        ----------
        metrics : Dict[str, float]
            Metrics dictionary from evaluate() or evaluate_batch().
        title : str, default="Evaluation Metrics"
            Title for the printout.

        Example
        -------
        >>> evaluator = CombMaskEvaluator()
        >>> metrics = evaluator.evaluate(pred_mask, gt_mask)
        >>> evaluator.print_metrics(metrics)
        """
        print("\n" + "=" * 60)
        print(f"{title:^60}")
        print("=" * 60)

        # Main metrics
        if "accuracy" in metrics:
            print("\nMain Metrics:")
            print(
                f"Accuracy:    {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)"
            )
            print(f"IoU:         {metrics['iou']:.4f}  ({metrics['iou']*100:.2f}%)")
            print(f"Dice:        {metrics['dice']:.4f}  ({metrics['dice']*100:.2f}%)")
            print(
                f"F1 Score:    {metrics['f1_score']:.4f}  ({metrics['f1_score']*100:.2f}%)"
            )

        # Precision and Recall
        if "precision" in metrics:
            print("\nPrecision & Recall:")
            print(
                f"Precision:   {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)"
            )
            print(
                f"Recall:      {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)"
            )
            print(
                f"Specificity: {metrics['specificity']:.4f}  ({metrics['specificity']*100:.2f}%)"
            )

        # Confusion matrix
        if "tp" in metrics:
            print("\nConfusion Matrix:")
            print(f"True Positives  (TP): {metrics['tp']:,}")
            print(f"True Negatives  (TN): {metrics['tn']:,}")
            print(f"False Positives (FP): {metrics['fp']:,}")
            print(f"False Negatives (FN): {metrics['fn']:,}")
            total = metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"]
            print(f"Total pixels:         {total:,}")

        # Aggregate statistics
        if "iou_mean" in metrics:
            print(f"\nAggregate Statistics (n={metrics.get('num_images', 0)} images):")
            print("\n  Main Metrics:")
            print(
                f"IoU:      {metrics['iou_mean']:.4f} +/- {metrics['iou_std']:.4f}  "
                f"(range: {metrics['iou_min']:.4f} - {metrics['iou_max']:.4f})"
            )
            print(
                f"Dice/F1:  {metrics['dice_mean']:.4f} +/- {metrics['dice_std']:.4f}  "
                f"(range: {metrics['dice_min']:.4f} - {metrics['dice_max']:.4f})"
            )
            print(
                f"Accuracy: {metrics['accuracy_mean']:.4f} +/- {metrics['accuracy_std']:.4f}  "
                f"(range: {metrics['accuracy_min']:.4f} - {metrics['accuracy_max']:.4f})"
            )

            if "precision_mean" in metrics:
                print("\n  Precision & Recall:")
                print(
                    f"Precision:   {metrics['precision_mean']:.4f} +/- {metrics['precision_std']:.4f}  "
                    f"(range: {metrics['precision_min']:.4f} - {metrics['precision_max']:.4f})"
                )
                print(
                    f"Recall:      {metrics['recall_mean']:.4f} +/- {metrics['recall_std']:.4f}  "
                    f"(range: {metrics['recall_min']:.4f} - {metrics['recall_max']:.4f})"
                )
                print(
                    f"Specificity: {metrics['specificity_mean']:.4f} +/- {metrics['specificity_std']:.4f}  "
                    f"(range: {metrics['specificity_min']:.4f} - {metrics['specificity_max']:.4f})"
                )

            if "tp_total" in metrics:
                print("\n  Confusion Matrix Totals:")
                print(
                    f"True Positives  (TP): {metrics['tp_total']:,}  (avg: {metrics['tp_mean']:,.0f} per image)"
                )
                print(
                    f"True Negatives  (TN): {metrics['tn_total']:,}  (avg: {metrics['tn_mean']:,.0f} per image)"
                )
                print(
                    f"False Positives (FP): {metrics['fp_total']:,}  (avg: {metrics['fp_mean']:,.0f} per image)"
                )
                print(
                    f"False Negatives (FN): {metrics['fn_total']:,}  (avg: {metrics['fn_mean']:,.0f} per image)"
                )
                total = (
                    metrics["tp_total"]
                    + metrics["tn_total"]
                    + metrics["fp_total"]
                    + metrics["fn_total"]
                )
                print(f"Total pixels:         {total:,}")

        print("=" * 60 + "\n")

    def save_results(
        self, metrics: Dict[str, float], output_path: Union[str, Path]
    ) -> None:
        """
        Save evaluation metrics to a JSON file.

        Parameters
        ----------
        metrics : Dict[str, float]
            Metrics dictionary to save.
        output_path : Union[str, Path]
            Path where JSON file should be saved.

        Example
        -------
        >>> evaluator = CombMaskEvaluator()
        >>> metrics = evaluator.evaluate(pred_mask, gt_mask)
        >>> evaluator.save_results(metrics, 'results/evaluation_metrics.json')
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Metrics saved to: {output_path}")

    def save_batch_results(
        self,
        aggregate_metrics: Dict[str, float],
        per_image_metrics: List[Dict[str, float]],
        output_dir: Union[str, Path],
        run_name: str,
        save_figure: bool = True,
        predicted_masks: Optional[List[np.ndarray]] = None,
        ground_truth_masks: Optional[List[np.ndarray]] = None,
        input_images: Optional[List[np.ndarray]] = None,
        image_names: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """
        Save batch evaluation results with a meaningful run name.

        Creates a directory structure to organize results from a batch evaluation:
        - {output_dir}/{run_name}/
          - aggregate_metrics.json: Overall statistics across all images
          - per_image_metrics.json: Detailed metrics for each image
          - visualization.png: Grid visualization (if save_figure=True)

        Parameters
        ----------
        aggregate_metrics : Dict[str, float]
            Aggregate statistics from evaluate_batch_from_arrays().
        per_image_metrics : List[Dict[str, float]]
            Per-image metrics from evaluate_batch_from_arrays().
        output_dir : Union[str, Path]
            Base directory where results will be saved.
        run_name : str
            Meaningful name for this evaluation run (e.g., "kernel7_iter3",
            "baseline_unet_effb0", "test_run_2024-10-25").
        save_figure : bool, default=True
            Whether to generate and save visualization grid.
        predicted_masks : Optional[List[np.ndarray]], default=None
            Required if save_figure=True.
        ground_truth_masks : Optional[List[np.ndarray]], default=None
            Required if save_figure=True.
        input_images : Optional[List[np.ndarray]], default=None
            Optional input images for visualization.
        image_names : Optional[List[str]], default=None
            Optional image names for visualization.

        Returns
        -------
        Dict[str, Path]
            Dictionary mapping result types to their file paths:
            - 'aggregate': Path to aggregate_metrics.json
            - 'per_image': Path to per_image_metrics.json
            - 'visualization': Path to visualization.png (if save_figure=True)

        Raises
        ------
        ValueError
            If save_figure=True but masks are not provided.

        Example
        -------
        >>> # Run batch evaluation
        >>> aggregate, per_image = evaluator.evaluate_batch_from_arrays(
        ...     pred_masks, gt_masks, names
        ... )
        >>>
        >>> # Save results with meaningful name
        >>> paths = evaluator.save_batch_results(
        ...     aggregate_metrics=aggregate,
        ...     per_image_metrics=per_image,
        ...     output_dir='results',
        ...     run_name='unet_effb0_kernel7_2024-10-25',
        ...     save_figure=True,
        ...     predicted_masks=pred_masks,
        ...     ground_truth_masks=gt_masks,
        ...     input_images=input_imgs,
        ...     image_names=names
        ... )
        >>> print(f"Results saved to: {paths['aggregate'].parent}")
        """
        import json
        from datetime import datetime

        output_dir = Path(output_dir)
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # Save aggregate metrics
        aggregate_path = run_dir / "aggregate_metrics.json"
        with open(aggregate_path, "w", encoding="utf-8") as f:
            json.dump(aggregate_metrics, f, indent=2)
        saved_paths["aggregate"] = aggregate_path
        print(f"✓ Aggregate metrics saved to: {aggregate_path}")

        # Save per-image metrics
        per_image_path = run_dir / "per_image_metrics.json"
        with open(per_image_path, "w", encoding="utf-8") as f:
            json.dump(per_image_metrics, f, indent=2)
        saved_paths["per_image"] = per_image_path
        print(f"✓ Per-image metrics saved to: {per_image_path}")

        # Save visualization if requested
        if save_figure:
            if predicted_masks is None or ground_truth_masks is None:
                raise ValueError(
                    "predicted_masks and ground_truth_masks must be provided when save_figure=True"
                )

            fig, axes = self.visualize_batch(
                predicted_masks=predicted_masks,
                ground_truth_masks=ground_truth_masks,
                image_names=image_names,
                input_images=input_images,
                max_cols=2,
            )

            viz_path = run_dir / "visualization.png"
            fig.savefig(viz_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_paths["visualization"] = viz_path
            print(f"✓ Visualization saved to: {viz_path}")

        # Create a summary text file
        summary_path = run_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Batch Evaluation Results: {run_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")

            # Write aggregate statistics
            f.write(
                f"Aggregate Statistics (n={aggregate_metrics.get('num_images', 0)} images):\n\n"
            )
            f.write("Main Metrics:\n")
            f.write(
                f"  IoU:      {aggregate_metrics['iou_mean']:.4f} +/- {aggregate_metrics['iou_std']:.4f}\n"
            )
            f.write(
                f"  Dice/F1:  {aggregate_metrics['dice_mean']:.4f} +/- {aggregate_metrics['dice_std']:.4f}\n"
            )
            f.write(
                f"  Accuracy: {aggregate_metrics['accuracy_mean']:.4f} +/- {aggregate_metrics['accuracy_std']:.4f}\n\n"
            )

            if "precision_mean" in aggregate_metrics:
                f.write("Precision & Recall:\n")
                f.write(
                    f"  Precision:   {aggregate_metrics['precision_mean']:.4f} +/- {aggregate_metrics['precision_std']:.4f}\n"
                )
                f.write(
                    f"  Recall:      {aggregate_metrics['recall_mean']:.4f} +/- {aggregate_metrics['recall_std']:.4f}\n"
                )
                f.write(
                    f"  Specificity: {aggregate_metrics['specificity_mean']:.4f} +/- {aggregate_metrics['specificity_std']:.4f}\n\n"
                )

            # Write per-image results
            f.write("\nPer-Image Results:\n")
            f.write("-" * 60 + "\n")
            for metrics in per_image_metrics:
                f.write(f"\n{metrics['filename']}:\n")
                f.write(
                    f"  IoU: {metrics['iou']:.4f} | Dice: {metrics['dice']:.4f} | Accuracy: {metrics['accuracy']:.4f}\n"
                )
                f.write(
                    f"  Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}\n"
                )

        saved_paths["summary"] = summary_path
        print(f"✓ Summary saved to: {summary_path}")

        print(f"\n{'='*60}")
        print(f"All results saved to: {run_dir}")
        print(f"{'='*60}\n")

        return saved_paths
