"""
CombMaskGeneratorSAM: Generate binary comb masks using Segment Anything Model (SAM).

This module uses SAM to detect the wooden frame boundaries in honeybee comb images
and estimates the comb region based on frame detection. This approach is robust
because wooden frames have consistent visual characteristics across images.
"""

from typing import Optional, Tuple, List
import numpy as np
import cv2
import matplotlib.pyplot as plt


class CombMaskGeneratorSAM:
    """
    Generate binary comb masks using Segment Anything Model (SAM).

    This class uses SAM to detect wooden frame boundaries in honeybee comb images.
    The comb region is estimated as the area inside the frame, which provides a
    robust segmentation approach since wooden frames have consistent appearance.

    The workflow:
    1. Load SAM model (automatic mask generation or prompted segmentation)
    2. Detect wooden frame boundaries
    3. Extract the inner region (comb area)
    4. Post-process to create clean binary mask
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
    ):
        """
        Initialize the SAM-based comb mask generator.

        Parameters
        ----------
        model_type : str, default="vit_h"
            SAM model variant to use:
            - "vit_h" (huge): Most accurate, slowest
            - "vit_l" (large): Good balance
            - "vit_b" (base): Fastest, less accurate
        checkpoint_path : Optional[str], default=None
            Path to SAM checkpoint file (.pth).
            If None, you'll need to download it first.
            Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints
        device : str, default="cuda"
            Device to run inference on ("cuda" or "cpu").
        points_per_side : int, default=32
            Number of points per side for automatic mask generation.
            Higher = more detailed but slower.
        pred_iou_thresh : float, default=0.88
            IoU threshold for filtering predicted masks.
        stability_score_thresh : float, default=0.95
            Stability score threshold for mask filtering.

        Example
        -------
        >>> # Initialize with downloaded checkpoint
        >>> generator = CombMaskGeneratorSAM(
        ...     model_type="vit_h",
        ...     checkpoint_path="models/sam_vit_h_4b8939.pth",
        ...     device="cuda"
        ... )
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.points_per_side = points_per_side
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh

        # Lazy loading - only import and load when needed
        self.sam = None
        self.mask_generator = None
        self.predictor = None

    def _load_sam(self):
        """Lazy load SAM model when first needed."""
        if self.sam is not None:
            return

        try:
            from segment_anything import (
                sam_model_registry,
                SamAutomaticMaskGenerator,
                SamPredictor,
            )
        except ImportError:
            raise ImportError(
                "segment-anything is not installed. Install it with:\n"
                "  pip install segment-anything\n"
                "Or from source:\n"
                "  pip install git+https://github.com/facebookresearch/segment-anything.git"
            )

        if self.checkpoint_path is None:
            raise ValueError(
                "checkpoint_path must be provided. Download SAM checkpoints from:\n"
                "https://github.com/facebookresearch/segment-anything#model-checkpoints\n"
                "Available models:\n"
                "  - vit_h: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth\n"
                "  - vit_l: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth\n"
                "  - vit_b: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            )

        # Load SAM model
        self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        self.sam.to(device=self.device)

        # Initialize mask generator for automatic segmentation
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
        )

        # Initialize predictor for prompted segmentation
        self.predictor = SamPredictor(self.sam)

        print(f"✓ SAM model loaded: {self.model_type}")

    def generate_comb_mask_auto(
        self,
        image: np.ndarray,
        select_largest: bool = True,
        min_area_ratio: float = 0.3,
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Generate comb mask using automatic mask generation.

        SAM automatically generates multiple masks for the image, and we select
        the one(s) most likely to represent the comb region (typically the largest
        central region excluding the wooden frame).

        Parameters
        ----------
        image : np.ndarray
            Input image (H, W, 3) in RGB format or (H, W) grayscale.
            Will be converted to RGB if grayscale.
        select_largest : bool, default=True
            If True, select only the largest mask (likely the comb).
            If False, return all masks above the area threshold.
        min_area_ratio : float, default=0.3
            Minimum mask area as ratio of total image area.
            Masks smaller than this are filtered out.

        Returns
        -------
        Tuple[np.ndarray, List[dict]]
            - Binary mask (H, W) with values 0 or 1
            - List of mask dictionaries with metadata (area, bbox, etc.)

        Example
        -------
        >>> generator = CombMaskGeneratorSAM(
        ...     checkpoint_path="models/sam_vit_h_4b8939.pth"
        ... )
        >>> image = cv2.imread("comb_image.png")
        >>> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        >>> comb_mask, masks = generator.generate_comb_mask_auto(image)
        >>> print(f"Generated {len(masks)} masks")
        """
        self._load_sam()

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Generate masks
        print("Generating masks with SAM (this may take a moment)...")
        masks = self.mask_generator.generate(image)
        print(f"✓ Generated {len(masks)} initial masks")

        # Filter masks by area
        total_area = image.shape[0] * image.shape[1]
        min_area = total_area * min_area_ratio

        filtered_masks = [m for m in masks if m["area"] >= min_area]
        print(
            f"✓ Filtered to {len(filtered_masks)} masks (area >= {min_area_ratio*100:.1f}%)"
        )

        if len(filtered_masks) == 0:
            print("⚠ No masks found above area threshold. Returning empty mask.")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8), []

        # Sort by area (largest first)
        filtered_masks = sorted(filtered_masks, key=lambda x: x["area"], reverse=True)

        if select_largest:
            # Use only the largest mask (most likely the comb region)
            selected_masks = [filtered_masks[0]]
            print(
                f"✓ Selected largest mask (area: {filtered_masks[0]['area']:,} pixels)"
            )
        else:
            selected_masks = filtered_masks

        # Combine selected masks into binary mask
        combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for mask_dict in selected_masks:
            combined_mask = np.logical_or(
                combined_mask, mask_dict["segmentation"]
            ).astype(np.uint8)

        return combined_mask, selected_masks

    def generate_comb_mask_box(
        self,
        image: np.ndarray,
        box: Optional[np.ndarray] = None,
        margin_ratio: float = 0.05,
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate comb mask using box prompt (for frame detection).

        If you know approximately where the frame is, you can provide a bounding box
        and SAM will segment the region inside it. If no box is provided, uses
        automatic detection with a central box.

        Parameters
        ----------
        image : np.ndarray
            Input image (H, W, 3) in RGB format or (H, W) grayscale.
        box : Optional[np.ndarray], default=None
            Bounding box [x1, y1, x2, y2] in pixel coordinates.
            If None, uses a central box with margins.
        margin_ratio : float, default=0.05
            Margin ratio from image edges when auto-detecting box.
            0.05 = 5% margin on each side.

        Returns
        -------
        Tuple[np.ndarray, dict]
            - Binary mask (H, W) with values 0 or 1
            - Metadata dictionary with bbox and scores

        Example
        -------
        >>> # Automatic box detection
        >>> comb_mask, metadata = generator.generate_comb_mask_box(image)
        >>>
        >>> # Manual box specification
        >>> box = np.array([50, 50, 950, 950])  # [x1, y1, x2, y2]
        >>> comb_mask, metadata = generator.generate_comb_mask_box(image, box=box)
        """
        self._load_sam()

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Auto-generate box if not provided
        if box is None:
            h, w = image.shape[:2]
            margin_h = int(h * margin_ratio)
            margin_w = int(w * margin_ratio)
            box = np.array([margin_w, margin_h, w - margin_w, h - margin_h])
            print(f"Using automatic box with {margin_ratio*100:.1f}% margins: {box}")

        # Set image for predictor
        self.predictor.set_image(image)

        # Predict with box prompt
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=True,
        )

        # Select best mask (highest score)
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx].astype(np.uint8)

        metadata = {
            "box": box,
            "scores": scores,
            "best_score": float(scores[best_idx]),
        }

        print(f"✓ Generated mask with box prompt (score: {metadata['best_score']:.3f})")

        return best_mask, metadata

    def generate_comb_mask_points(
        self,
        image: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        height_ratio: float = 0.5,
        width_ratios: Optional[List[float]] = None,
        multimask_output: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        """
        Generate comb mask using point prompts (ideal for multiple frames).

        This method is perfect for images with multiple wooden frames (e.g., two
        frames side by side). You can specify point locations as ratios of image
        dimensions, and SAM will segment the objects at those locations.

        Parameters
        ----------
        image : np.ndarray
            Input image (H, W, 3) in RGB format or (H, W) grayscale.
        point_coords : Optional[np.ndarray], default=None
            Point coordinates as (N, 2) array [x, y] in pixels.
            If None, uses automatic positioning based on ratios.
        point_labels : Optional[np.ndarray], default=None
            Labels for points (1 = foreground, 0 = background).
            If None, all points are foreground (1).
        height_ratio : float, default=0.5
            Height position as ratio (0.5 = center vertically).
            Used only if point_coords is None.
        width_ratios : Optional[List[float]], default=[0.33, 0.67]
            Width positions as ratios for multiple frames.
            Default: [0.33, 0.67] for two frames (1/3 and 2/3 of width).
            Used only if point_coords is None.
        multimask_output : bool, default=False
            If True, returns 3 masks per point. If False, returns 1 best mask.

        Returns
        -------
        Tuple[np.ndarray, dict]
            - Binary mask (H, W) with values 0 or 1 (combined from all points)
            - Metadata dictionary with point information and scores

        Example
        -------
        >>> # Automatic positioning for 2 frames (default)
        >>> comb_mask, metadata = generator.generate_comb_mask_points(image)
        >>> # This places points at (h/2, w/3) and (h/2, 2w/3)
        >>>
        >>> # Custom positioning for 3 frames
        >>> comb_mask, metadata = generator.generate_comb_mask_points(
        ...     image,
        ...     height_ratio=0.5,
        ...     width_ratios=[0.25, 0.5, 0.75]
        ... )
        >>>
        >>> # Manual point specification
        >>> points = np.array([[100, 200], [300, 200]])  # Two points
        >>> labels = np.array([1, 1])  # Both are foreground
        >>> comb_mask, metadata = generator.generate_comb_mask_points(
        ...     image, point_coords=points, point_labels=labels
        ... )
        """
        self._load_sam()

        # Convert grayscale to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        h, w = image.shape[:2]

        # Auto-generate points if not provided
        if point_coords is None:
            if width_ratios is None:
                width_ratios = [0.33, 0.67]  # Default: 2 frames at 1/3 and 2/3

            # Calculate point positions
            point_coords = np.array(
                [[int(w * w_ratio), int(h * height_ratio)] for w_ratio in width_ratios]
            )
            print("Using automatic point placement:")
            for i, (x, y) in enumerate(point_coords):
                print(f"  Point {i+1}: ({x}, {y}) - width ratio {width_ratios[i]:.2f}")

        # Auto-generate labels if not provided (all foreground)
        if point_labels is None:
            point_labels = np.ones(len(point_coords), dtype=int)

        # Set image for predictor
        self.predictor.set_image(image)

        # Predict with point prompts
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            multimask_output=multimask_output,
        )

        # Combine all masks (take best mask if multimask_output)
        if multimask_output:
            # Select best mask (highest score)
            best_idx = np.argmax(scores)
            combined_mask = masks[best_idx].astype(np.uint8)
            best_score = scores[best_idx]
        else:
            # Single mask per point - combine all
            combined_mask = np.any(masks, axis=0).astype(np.uint8)
            best_score = np.mean(scores)

        metadata = {
            "point_coords": point_coords,
            "point_labels": point_labels,
            "num_points": len(point_coords),
            "scores": scores,
            "best_score": float(best_score),
            "masks_shape": masks.shape,
        }

        print(
            f"✓ Generated mask with {len(point_coords)} point prompt(s) (score: {best_score:.3f})"
        )

        return combined_mask, metadata

    def visualize_detection(
        self,
        image: np.ndarray,
        comb_mask: np.ndarray,
        masks_metadata: Optional[dict] = None,
        show_all_masks: bool = False,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Visualize SAM detection results.

        Shows the original image (with prompts if available), detected masks
        (with boundaries), and the final binary comb mask.

        Parameters
        ----------
        image : np.ndarray
            Original input image (H, W, 3) or (H, W).
        comb_mask : np.ndarray
            Generated binary comb mask (H, W).
        masks_metadata : Optional[dict], default=None
            Metadata dictionary from generate_comb_mask_*() methods.
            Can contain 'point_coords', 'box', etc.
        show_all_masks : bool, default=False
            If True and masks_metadata is a list of masks, show all detected masks.
            If False, only show the final combined mask.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Matplotlib figure and axes array.

        Example
        -------
        >>> # With point prompts
        >>> comb_mask, metadata = generator.generate_comb_mask_points(image)
        >>> fig, axes = generator.visualize_detection(image, comb_mask, metadata)
        >>> plt.show()
        >>>
        >>> # With automatic detection
        >>> comb_mask, masks = generator.generate_comb_mask_auto(image)
        >>> fig, axes = generator.visualize_detection(image, comb_mask, masks, show_all_masks=True)
        >>> plt.show()
        """
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_image = image.copy()

        # Determine if metadata is a list (from auto) or dict (from points/box)
        is_list_metadata = isinstance(masks_metadata, list)

        # Determine number of subplots
        n_plots = 3 if is_list_metadata and show_all_masks else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))

        if n_plots == 2:
            axes = [axes[0], axes[1]]

        idx = 0

        # Plot 1: Original image with prompts (if available)
        axes[idx].imshow(display_image)

        # Draw prompts on the image
        if masks_metadata and isinstance(masks_metadata, dict):
            # Draw point prompts
            if "point_coords" in masks_metadata:
                points = masks_metadata["point_coords"]
                labels = masks_metadata.get("point_labels", np.ones(len(points)))
                for (x, y), label in zip(points, labels):
                    color = "lime" if label == 1 else "red"
                    marker = "o" if label == 1 else "x"
                    axes[idx].plot(
                        x,
                        y,
                        marker=marker,
                        color=color,
                        markersize=15,
                        markeredgewidth=3,
                        markeredgecolor="white",
                    )
                axes[idx].set_title(
                    f"Input Image + {len(points)} Point Prompt(s)", fontsize=14
                )
            # Draw box prompt
            elif "box" in masks_metadata:
                box = masks_metadata["box"]
                x1, y1, x2, y2 = box
                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=3,
                )
                axes[idx].add_patch(rect)
                axes[idx].set_title("Input Image + Box Prompt", fontsize=14)
            else:
                axes[idx].set_title("Original Image", fontsize=14)
        else:
            axes[idx].set_title("Original Image", fontsize=14)

        axes[idx].axis("off")
        idx += 1

        # Plot 2: All detected masks (if requested and available)
        if is_list_metadata and show_all_masks:
            mask_overlay = display_image.copy()
            for i, mask_dict in enumerate(masks_metadata):
                # Create colored overlay
                color = np.random.random(3) * 255
                mask_overlay[mask_dict["segmentation"]] = (
                    mask_overlay[mask_dict["segmentation"]] * 0.5 + color * 0.5
                ).astype(np.uint8)

                # Draw bounding box
                bbox = mask_dict["bbox"]  # [x, y, w, h]
                x, y, w, h = bbox
                cv2.rectangle(
                    mask_overlay, (x, y), (x + w, y + h), color.astype(int).tolist(), 2
                )

            axes[idx].imshow(mask_overlay)
            axes[idx].set_title(
                f"Detected Masks (n={len(masks_metadata)})", fontsize=14
            )
            axes[idx].axis("off")
            idx += 1

        # Plot 3: Final binary mask with overlay
        overlay = display_image.copy()
        # Create green overlay for comb region
        overlay[comb_mask == 1] = (
            overlay[comb_mask == 1] * 0.6 + np.array([0, 255, 0]) * 0.4
        ).astype(np.uint8)

        # Draw contours
        contours, _ = cv2.findContours(
            comb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 3)

        axes[idx].imshow(overlay)
        axes[idx].set_title("Final Comb Mask", fontsize=14)
        axes[idx].axis("off")

        plt.tight_layout()
        return fig, axes

    def save_mask(self, comb_mask: np.ndarray, output_path: str) -> None:
        """
        Save the binary comb mask to disk.

        Parameters
        ----------
        comb_mask : np.ndarray
            Binary comb mask (H, W) with values 0 or 1.
        output_path : str
            Path where the mask should be saved.

        Example
        -------
        >>> generator.save_mask(comb_mask, "output/comb_mask_sam.png")
        """
        cv2.imwrite(output_path, comb_mask)
        print(f"✓ Comb mask saved to: {output_path}")
