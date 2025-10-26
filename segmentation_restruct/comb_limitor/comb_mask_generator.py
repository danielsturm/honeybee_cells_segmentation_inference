"""
CombMaskGenerator: Generate binary comb masks from segmentation predictions.

This module creates binary masks representing the entire honeybee comb structure
by combining all cell-type predictions and applying morphological operations to
fill gaps between individual cells.
"""

from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


class CombMaskGenerator:
    """
    Generate binary comb masks from honeybee comb segmentation predictions.

    This class takes segmentation masks (with multiple cell type classes) and
    creates a unified binary mask representing the entire comb structure. It
    filters out bees and background, keeping only actual cell structures, then
    applies morphological closing to fill small gaps between cells.

    The class is designed to work with images that have no bees or minimal bee
    presence, focusing on extracting the comb structure itself.
    """

    def __init__(
        self,
        cell_classes: Optional[list] = None,
        closing_kernel_size: int = 5,
        closing_iterations: int = 1,
        remove_outliers: bool = True,
        min_region_size: int = 500,
    ):
        """
        Initialize the CombMaskGenerator.

        Parameters
        ----------
        cell_classes : Optional[list], default=[2, 3, 4, 5, 6, 7]
            List of class indices that represent actual cells in the comb.
            Default includes all cell types from label_classes.json:
            - 2: empty_cell
            - 3: open_brood
            - 4: open_honey
            - 5: capped_honey
            - 6: capped_brood
            - 7: pollen
            Excludes: 0 (background), 1 (bees), 8 (bee_in_cell)
        closing_kernel_size : int, default=5
            Size of the kernel for morphological closing operation (erosion + dilation).
            Larger values fill bigger gaps between cells.
            Recommended range: 3-11 (odd numbers work best).
        closing_iterations : int, default=1
            Number of times to apply the closing operation.
            More iterations = more aggressive gap filling.
        remove_outliers : bool, default=True
            Whether to remove small isolated regions (outliers) from the mask.
            This removes noise and small artifacts that aren't part of the main comb.
        min_region_size : int, default=500
            Minimum size (in pixels) for a region to be kept.
            Regions smaller than this are considered outliers and removed.
            Typical values: 100-1000 pixels depending on image resolution.
            - Smaller values: Keep more regions, less aggressive filtering
            - Larger values: Remove more small regions, more aggressive filtering

        Example
        -------
        >>> # Default settings - include all cell types, mild closing, remove outliers
        >>> generator = CombMaskGenerator()
        >>>
        >>> # Custom settings - aggressive outlier removal
        >>> generator = CombMaskGenerator(
        ...     cell_classes=[3, 4, 5, 6],  # Only brood and honey
        ...     closing_kernel_size=9,
        ...     closing_iterations=2,
        ...     remove_outliers=True,
        ...     min_region_size=1000  # Remove regions smaller than 1000 pixels
        ... )
        >>>
        >>> # Disable outlier removal if you want to keep all regions
        >>> generator = CombMaskGenerator(remove_outliers=False)
        """
        # Default cell classes: all actual cell types (exclude background, bees, bee_in_cell)
        self.cell_classes = (
            cell_classes if cell_classes is not None else [2, 3, 4, 5, 6, 7]
        )
        self.closing_kernel_size = closing_kernel_size
        self.closing_iterations = closing_iterations
        self.remove_outliers = remove_outliers
        self.min_region_size = min_region_size

        # Create morphological kernel for closing operation
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size)
        )

    def _suppress_outliers(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Remove small isolated regions (outliers) from the binary mask.

        Uses connected component analysis to identify separate regions and
        removes those smaller than the minimum region size threshold. This
        effectively filters out noise and small artifacts.

        Parameters
        ----------
        binary_mask : np.ndarray
            Binary mask (H, W) with values 0 or 1.

        Returns
        -------
        np.ndarray
            Filtered binary mask with small regions removed.

        Process
        -------
        1. Find all connected components (separate regions)
        2. Calculate the size (number of pixels) of each region
        3. Keep only regions that meet the minimum size threshold
        4. Remove all smaller regions (outliers)

        Example
        -------
        >>> # Internal method, called automatically if remove_outliers=True
        >>> # Can also be used standalone:
        >>> generator = CombMaskGenerator()
        >>> noisy_mask = np.array([...])  # Binary mask with noise
        >>> clean_mask = generator._suppress_outliers(noisy_mask)
        """
        # Find all connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        # Create output mask (start with zeros)
        filtered_mask = np.zeros_like(binary_mask)

        # Iterate through all components (skip 0, which is background)
        for label in range(1, num_labels):
            # Get the area (number of pixels) of this component
            area = stats[label, cv2.CC_STAT_AREA]

            # Keep only regions that meet the minimum size threshold
            if area >= self.min_region_size:
                filtered_mask[labels == label] = 1

        return filtered_mask.astype(np.uint8)

    def generate_comb_mask(self, segmentation_mask: np.ndarray) -> np.ndarray:
        """
        Generate binary comb mask from segmentation predictions.

        Takes a multi-class segmentation mask and creates a binary mask where:
        - 1 = comb cells (any cell type)
        - 0 = background, bees, or bee_in_cell

        The method groups all cell classes together, then applies morphological
        closing to fill small gaps between adjacent cells.

        Parameters
        ----------
        segmentation_mask : np.ndarray
            Segmentation mask of shape (H, W) where each pixel value represents
            a class index (0-8 based on label_classes.json).
            Typically the output from HoneyBeeCombInferer.infer() or
            HoneyBeeCombInferer.infer_without_bees().

        Returns
        -------
        np.ndarray
            Binary mask of shape (H, W) with dtype uint8:
            - 1 = comb cell regions
            - 0 = non-comb regions (background, bees)

            Note: Values are 0 and 1 (NOT 0 and 255) to match ground truth format.
            This allows direct comparison with GT labels without conversion.

        Process
        -------
        1. Create binary mask by checking if pixels belong to cell classes
        2. Apply morphological closing (erosion followed by dilation)
           - Erosion: Shrinks the mask, removing small noise
           - Dilation: Expands it back, filling gaps between cells
        3. Optionally remove outliers (small isolated regions) if enabled
        4. Return cleaned binary mask

        Example
        -------
        >>> from honeybee_comb_inferer.inference import HoneyBeeCombInferer
        >>> from segmentation_restruct.comb_limitor import CombMaskGenerator
        >>>
        >>> # Get segmentation mask (without bees)
        >>> model = HoneyBeeCombInferer(model_name='unet_effnetb0',
        ...                              path_to_pretrained_models='models/',
        ...                              device='cuda')
        >>> seg_mask = model.infer_without_bees('data/frames/')
        >>>
        >>> # Generate binary comb mask
        >>> generator = CombMaskGenerator(closing_kernel_size=7)
        >>> comb_mask = generator.generate_comb_mask(seg_mask)
        >>>
        >>> # Visualize
        >>> import matplotlib.pyplot as plt
        >>> plt.imshow(comb_mask, cmap='gray')
        >>> plt.title('Binary Comb Mask')
        >>> plt.show()
        """
        # Step 1: Create binary mask - 1 for any cell class, 0 otherwise
        binary_mask = np.isin(segmentation_mask, self.cell_classes).astype(np.uint8)

        # Step 2: Apply morphological closing to fill gaps between cells
        # This connects nearby cells and creates a continuous comb structure
        closed_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=self.closing_iterations,
        )

        # Step 3: Remove outliers (small isolated regions) if enabled
        if self.remove_outliers:
            closed_mask = self._suppress_outliers(closed_mask)

        # Return as 0/1 binary mask (same format as ground truth)
        # This allows direct comparison with GT labels without conversion
        return closed_mask.astype(np.uint8)

    def visualize_comparison(
        self,
        segmentation_mask: np.ndarray,
        comb_mask: Optional[np.ndarray] = None,
        input_image: Optional[np.ndarray] = None,
        cmap_colors: Optional[dict] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize the original segmentation, binary comb mask, and optionally the input image.

        Parameters
        ----------
        segmentation_mask : np.ndarray
            Multi-class segmentation mask (H, W) with class indices 0-8.
        comb_mask : Optional[np.ndarray], default=None
            Binary comb mask (H, W). If None, will be generated automatically.
        input_image : Optional[np.ndarray], default=None
            Original input image (H, W) or (H, W, 3). If provided, displayed as first subplot.
        cmap_colors : Optional[dict], default=None
            Dictionary mapping class indices to RGB(A) colors for visualization.
            If None, uses default grayscale for segmentation mask.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Matplotlib figure and axes objects for further customization.

        Example
        -------
        >>> # Generate and visualize
        >>> generator = CombMaskGenerator()
        >>> comb_mask = generator.generate_comb_mask(seg_mask)
        >>> fig, ax = generator.visualize_comparison(seg_mask, comb_mask, input_image)
        >>> plt.show()
        """
        # Generate comb mask if not provided
        if comb_mask is None:
            comb_mask = self.generate_comb_mask(segmentation_mask)

        # Determine subplot layout
        num_plots = 2 + (1 if input_image is not None else 0)

        fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 8))
        if num_plots == 2:
            axes = [axes[0], axes[1]]
        else:
            axes = list(axes)

        idx = 0

        # Plot input image if provided
        if input_image is not None:
            axes[idx].imshow(
                input_image, cmap="gray" if len(input_image.shape) == 2 else None
            )
            axes[idx].set_title("Input Image", fontsize=14)
            axes[idx].axis("off")
            idx += 1

        # Plot segmentation mask
        if cmap_colors is not None:
            # Convert segmentation mask to color using provided colormap
            max_label = max(cmap_colors.keys())
            color_map = np.zeros((max_label + 1, 4), dtype=np.float32)
            for label, color in cmap_colors.items():
                color_map[label] = color
            seg_colored = color_map[segmentation_mask]
            axes[idx].imshow(seg_colored)
        else:
            axes[idx].imshow(segmentation_mask, cmap="tab10")
        axes[idx].set_title("Segmentation Mask (Multi-class)", fontsize=14)
        axes[idx].axis("off")
        idx += 1

        # Plot binary comb mask
        axes[idx].imshow(comb_mask, cmap="gray")
        axes[idx].set_title("Binary Comb Mask (Cells Only)", fontsize=14)
        axes[idx].axis("off")

        plt.tight_layout()
        return fig, axes

    def get_comb_statistics(self, comb_mask: np.ndarray) -> dict:
        """
        Calculate statistics about the comb mask.

        Parameters
        ----------
        comb_mask : np.ndarray
            Binary comb mask (H, W) with values 0 or 1.

        Returns
        -------
        dict
            Dictionary containing:
            - 'total_pixels': Total number of pixels in image
            - 'comb_pixels': Number of pixels classified as comb
            - 'background_pixels': Number of pixels classified as background
            - 'comb_percentage': Percentage of image covered by comb
            - 'num_connected_components': Number of separate comb regions

        Example
        -------
        >>> generator = CombMaskGenerator()
        >>> comb_mask = generator.generate_comb_mask(seg_mask)
        >>> stats = generator.get_comb_statistics(comb_mask)
        >>> print(f"Comb coverage: {stats['comb_percentage']:.1f}%")
        >>> print(f"Number of comb regions: {stats['num_connected_components']}")
        """
        # Convert to binary if needed
        binary = (comb_mask > 0).astype(np.uint8)

        # Calculate pixel counts
        total_pixels = binary.size
        comb_pixels = np.sum(binary)
        background_pixels = total_pixels - comb_pixels
        comb_percentage = (comb_pixels / total_pixels) * 100

        # Find connected components
        num_components, _ = cv2.connectedComponents(binary)
        # Subtract 1 because background is counted as a component
        num_components -= 1

        return {
            "total_pixels": total_pixels,
            "comb_pixels": int(comb_pixels),
            "background_pixels": int(background_pixels),
            "comb_percentage": float(comb_percentage),
            "num_connected_components": int(num_components),
        }

    def save_mask(self, comb_mask: np.ndarray, output_path: str) -> None:
        """
        Save the binary comb mask to disk.

        Parameters
        ----------
        comb_mask : np.ndarray
            Binary comb mask (H, W) with values 0 or 1.
        output_path : str
            Path where the mask should be saved (e.g., 'output/comb_mask.png').

        Example
        -------
        >>> generator = CombMaskGenerator()
        >>> comb_mask = generator.generate_comb_mask(seg_mask)
        >>> generator.save_mask(comb_mask, 'data/masks/comb_binary_mask.png')
        """
        cv2.imwrite(output_path, comb_mask)
        print(f"Comb mask saved to: {output_path}")
