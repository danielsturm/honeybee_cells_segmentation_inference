"""
Example script demonstrating CombMaskGenerator and CombMaskGeneratorUNet usage.

This script shows how to generate binary comb masks from:
1. Multi-class segmentation predictions (CombMaskGenerator)
2. Custom-trained UNet model (CombMaskGeneratorUNet)
"""

from pathlib import Path
import sys

# Add project root to path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from honeybee_comb_inferer.inference import HoneyBeeCombInferer
from segmentation_restruct.comb_limitor import CombMaskGenerator
import matplotlib.pyplot as plt


def main():
    """
    Main example workflow:
    1. Load segmentation model
    2. Perform inference on bee-free image
    3. Generate binary comb mask
    4. Visualize and save results
    """

    # Configuration
    model_name = "unet_effnetb0"
    model_dir = root_dir / "models"
    device = "cuda"  # or 'cpu'

    # Path to your comb image (without bees or use infer_without_bees)
    image_path = root_dir / "data" / "images" / "your_comb_image.png"

    print("=" * 60)
    print("Binary Comb Mask Generation Example")
    print("=" * 60)

    # Step 1: Initialize segmentation model
    print("\n1. Loading segmentation model...")
    model = HoneyBeeCombInferer(
        model_name=model_name, path_to_pretrained_models=str(model_dir), device=device
    )
    print(f"   ✓ Model '{model_name}' loaded on {device}")

    # Step 2: Get segmentation prediction
    print("\n2. Running segmentation inference...")

    # Option A: Single image (may have bees)
    if image_path.exists():
        seg_mask = model.infer(str(image_path))
        print(f"   ✓ Segmented image: {image_path.name}")
    else:
        print(f"   ⚠ Image not found: {image_path}")
        print("   Please update 'image_path' variable with your image path")
        return

    # Option B: Bee-free segmentation from multiple frames (recommended)
    # frames_path = root_dir / 'data' / 'close_frames'
    # if frames_path.exists():
    #     seg_mask = model.infer_without_bees(str(frames_path))
    #     print(f"   ✓ Bee-free segmentation from {len(list(frames_path.iterdir()))} frames")

    # Step 3: Generate binary comb mask
    print("\n3. Generating binary comb mask...")
    generator = CombMaskGenerator(
        closing_kernel_size=7,  # Adjust for your images (3-11)
        closing_iterations=1,  # Usually 1 is sufficient
    )

    comb_mask = generator.generate_comb_mask(seg_mask)
    print(f"   ✓ Binary mask generated")
    print(f"   ✓ Closing kernel size: {generator.closing_kernel_size}")

    # Step 4: Get statistics
    print("\n4. Comb mask statistics:")
    stats = generator.get_comb_statistics(comb_mask)
    print(f"   • Image size: {stats['total_pixels']:,} pixels")
    print(f"   • Comb pixels: {stats['comb_pixels']:,}")
    print(f"   • Comb coverage: {stats['comb_percentage']:.1f}%")
    print(f"   • Number of regions: {stats['num_connected_components']}")

    # Step 5: Visualize
    print("\n5. Creating visualization...")
    fig, axes = generator.visualize_comparison(
        segmentation_mask=seg_mask, comb_mask=comb_mask, cmap_colors=model.cmap
    )
    plt.suptitle("Binary Comb Mask Generation", fontsize=16, y=1.02)

    # Step 6: Save results
    output_dir = root_dir / "data" / "comb_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_path = output_dir / f"binary_mask_{image_path.stem}.png"
    generator.save_mask(comb_mask, str(mask_path))
    print(f"   ✓ Mask saved to: {mask_path}")

    viz_path = output_dir / f"visualization_{image_path.stem}.png"
    plt.savefig(viz_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ Visualization saved to: {viz_path}")

    print("\n" + "=" * 60)
    print("✓ Complete! Showing visualization...")
    print("=" * 60)

    plt.show()

    # Optional: Experiment with different kernel sizes
    print("\n[Optional] Comparing different kernel sizes...")
    compare_kernel_sizes(seg_mask, output_dir, image_path.stem)


def compare_kernel_sizes(seg_mask, output_dir, image_stem):
    """Compare different morphological closing kernel sizes."""

    kernel_sizes = [3, 5, 7, 9, 11]
    fig, axes = plt.subplots(1, len(kernel_sizes), figsize=(20, 4))

    for idx, kernel_size in enumerate(kernel_sizes):
        gen = CombMaskGenerator(closing_kernel_size=kernel_size)
        mask = gen.generate_comb_mask(seg_mask)

        axes[idx].imshow(mask, cmap="gray")
        axes[idx].set_title(f"Kernel: {kernel_size}", fontsize=12)
        axes[idx].axis("off")

    plt.suptitle("Effect of Different Closing Kernel Sizes", fontsize=14)
    plt.tight_layout()

    comparison_path = output_dir / f"kernel_comparison_{image_stem}.png"
    plt.savefig(comparison_path, dpi=150, bbox_inches="tight")
    print(f"   ✓ Kernel comparison saved to: {comparison_path}")

    plt.show()


if __name__ == "__main__":
    main()
