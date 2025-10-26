# Comb Limitor - Binary Comb Mask Generator & Evaluator

This module provides **two approaches** for generating binary masks of honeybee comb structures:

1. **`CombMaskGenerator`**: Segmentation-based approach (combines cell predictions)
2. **`CombMaskGeneratorSAM`**: Frame detection approach using Segment Anything Model (SAM)

Both produce masks in **0/1 format** (0=background, 1=comb) to match CVAT ground truth labels.

---

## Approach 1: Segmentation-Based (`CombMaskGenerator`)

Uses trained segmentation models to identify cell types, then combines them into a binary comb mask.

### Purpose

When working with honeybee comb images without bees, you often need a simple binary mask showing where the comb is located. The `CombMaskGenerator` class:

1. **Combines all cell types** into a single "comb" category (ignoring bees and background)
2. **Applies morphological closing** to fill small gaps between individual cells
3. **Optionally removes outliers** (small isolated regions)
4. **Produces a clean binary mask** suitable for further processing

### Label Classes (from `label_classes.json`)

- **Cell classes** (included in mask):
  - 2: `empty_cell`
  - 3: `open_brood`
  - 4: `open_honey`
  - 5: `capped_honey`
  - 6: `capped_brood`
  - 7: `pollen`

- **Non-cell classes** (excluded from mask):
  - 0: `background`
  - 1: `bees`
  - 8: `bee_in_cell`

---

## Approach 2: SAM-Based (`CombMaskGeneratorSAM`)

Uses Segment Anything Model to detect wooden frame boundaries and extract the comb region.

### Purpose

Honeybee comb images typically have consistent structure: wooden frames containing the comb. SAM excels at detecting these frame boundaries without requiring training data.

### Advantages of SAM Approach

✅ **No training required**: Works out-of-the-box with pretrained SAM
✅ **Robust**: Handles different lighting, orientations, and comb types
✅ **Fast setup**: No need to train segmentation models
✅ **Frame-aware**: Directly detects physical frame boundaries
✅ **Flexible**: Can use automatic detection or manual prompts

### When to Use SAM vs Segmentation

| Feature | SAM Approach | Segmentation Approach |
|---------|-------------|----------------------|
| **Setup** | Download checkpoint (~2.4GB) | Train/load segmentation model |
| **Training data** | None needed | Requires labeled dataset |
| **Speed** | 2-3 sec/image (GPU) | 0.5-1 sec/image (GPU) |
| **Robustness** | Excellent for frames | Good for cells |
| **Cell detail** | No | Yes (identifies cell types) |
| **Best for** | Frame detection, quick prototyping | Cell-level analysis, detailed segmentation |

**Recommendation**: Use SAM for robust frame detection, then optionally use segmentation inside the detected region for cell-level analysis.

---

## Class Structure

## Usage Example

```python
from honeybee_comb_inferer.inference import HoneyBeeCombInferer
from segmentation_restruct.comb_limitor import CombMaskGenerator

# Step 1: Get segmentation mask (preferably without bees)
model = HoneyBeeCombInferer(
    model_name='unet_effnetb0',
    path_to_pretrained_models='models/',
    device='cuda'
)

# Option A: Single image inference
seg_mask = model.infer('path/to/comb_image.png')

# Option B: Bee-free inference from multiple frames (recommended)
seg_mask = model.infer_without_bees('path/to/frames/')

# Step 2: Generate binary comb mask
generator = CombMaskGenerator(
    closing_kernel_size=7,  # Larger = more aggressive gap filling
    closing_iterations=1
)
comb_mask = generator.generate_comb_mask(seg_mask)  # Returns 0/1 format

# Step 3: Visualize or save
generator.visualize_comparison(seg_mask, comb_mask)
generator.save_mask(comb_mask, 'output/comb_mask.png')

# Get statistics
stats = generator.get_comb_statistics(comb_mask)
print(f"Comb coverage: {stats['comb_percentage']:.1f}%")
```

## Parameters

### `CombMaskGenerator.__init__()`

- **`cell_classes`** (list): Class indices to include as comb cells
  - Default: `[2, 3, 4, 5, 6, 7]` (all cell types)
  - Custom: `[3, 4, 5, 6]` (only brood/honey cells, exclude empty cells and pollen)

- **`closing_kernel_size`** (int): Size of morphological kernel
  - Default: `5`
  - Range: `3-11` (odd numbers recommended)
  - Larger values fill bigger gaps

- **`closing_iterations`** (int): Number of closing operations
  - Default: `1`
  - More iterations = more aggressive gap filling

- **`remove_outliers`** (bool): Enable outlier suppression
  - Default: `True`
  - Removes small isolated regions (noise, artifacts)
  - Set to `False` to keep all regions

- **`min_region_size`** (int): Minimum region size in pixels
  - Default: `500`
  - Regions smaller than this are removed (if `remove_outliers=True`)
  - Recommended range: `100-1500` depending on image resolution
  - Smaller values: Less aggressive filtering, keep more regions
  - Larger values: More aggressive filtering, remove more noise
  - Increase for more aggressive gap filling

## Methods

### `generate_comb_mask(segmentation_mask)`
Main method to create binary comb mask.

**Input**: Multi-class segmentation mask (H, W) with values 0-8

**Output**: Binary mask (H, W) with values **0 or 1** (matches ground truth format)

**Process**:
1. Extract all cell classes (ignoring background/bees)
2. Apply morphological closing to fill gaps
3. Remove outliers (if enabled) - filters small isolated regions
4. Return clean binary mask

### `visualize_comparison(segmentation_mask, comb_mask, input_image, cmap_colors)`
Create side-by-side comparison plots.

### `get_comb_statistics(comb_mask)`
Calculate coverage statistics and number of connected components.

### `save_mask(comb_mask, output_path)`
Save binary mask to disk as PNG.

## Outlier Suppression

**What are outliers?**
Small isolated regions that aren't part of the main comb structure - typically noise, segmentation artifacts, or misclassified pixels.

**How it works:**
1. Identifies all separate connected regions in the mask
2. Measures the size (number of pixels) of each region
3. Removes regions smaller than `min_region_size` threshold
4. Keeps only significant comb structures

**Example configurations:**

```python
# Conservative - keep smaller regions
generator = CombMaskGenerator(
    closing_kernel_size=7,
    closing_iterations=1,
    remove_outliers=True,
    min_region_size=200  # Small threshold
)

# Balanced (default) - good for most cases
generator = CombMaskGenerator(
    closing_kernel_size=7,
    closing_iterations=1,
    remove_outliers=True,
    min_region_size=500  # Default
)

# Aggressive - only keep large regions
generator = CombMaskGenerator(
    closing_kernel_size=7,
    closing_iterations=1,
    remove_outliers=True,
    min_region_size=1500  # Large threshold
)

# No filtering - keep everything
generator = CombMaskGenerator(
    closing_kernel_size=7,
    closing_iterations=1,
    remove_outliers=False
)
```

**Benefits:**
- ✅ Removes segmentation noise
- ✅ Eliminates small artifacts
- ✅ Produces cleaner masks
- ✅ Improves evaluation metrics (less false positives)

**When to disable:**
- If you need to preserve all detected regions
- If working with highly fragmented combs
- For debugging segmentation quality

## Morphological Closing Explained

Closing = **Dilation** followed by **Erosion**

1. **Dilation**: Expands white regions, connecting nearby cells
2. **Erosion**: Shrinks back to original size, preserving connections

**Effect**: Small gaps between cells are filled while maintaining overall shape.

**Visual Example**:
```
Before:  ●  ●  ●  (individual cells with gaps)
After:   ●●●●●●  (connected comb structure)
```

## When to Use

### Segmentation Approach (`CombMaskGenerator`)

✅ **Best for**:
- Images with no bees or minimal bee presence
- Cell-level detail is important
- You have trained segmentation models
- Distinguishing between cell types matters
- Creating comb region masks for cropping
- Estimating total comb area
- Preprocessing for cell counting
- Removing noise and outliers

❌ **Not ideal for**:
- Images with many bees (use `infer_without_bees()` first)
- Quick prototyping without trained models
- When you only care about frame boundaries

### SAM Approach (`CombMaskGeneratorSAM`)

✅ **Best for**:
- Robust frame detection
- Quick setup (no training required)
- Handling lighting/orientation variations
- Prototyping and experimentation
- When segmentation models aren't available
- Detecting physical frame boundaries
- General-purpose comb region extraction

❌ **Not ideal for**:
- Fine-grained cell-level analysis
- Distinguishing between cell types
- When you need cell-specific information

---

## SAM Setup & Usage

### Installation

```bash
# Install segment-anything
pip install segment-anything

# Or from source
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Download SAM Checkpoint

Choose a model variant and download:

| Model | Size | Speed | Accuracy | Download Link |
|-------|------|-------|----------|---------------|
| **vit_h** | 2.4 GB | Slow | Best | [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| **vit_l** | 1.2 GB | Medium | Good | [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) |
| **vit_b** | 375 MB | Fast | OK | [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |

Save the checkpoint in your `models/` directory.

### Basic Usage

```python
from segmentation_restruct.comb_limitor import CombMaskGeneratorSAM
import cv2

# Initialize SAM generator
generator = CombMaskGeneratorSAM(
    model_type="vit_h",
    checkpoint_path="models/sam_vit_h_4b8939.pth",
    device="cuda"  # or "cpu"
)

# Load image (RGB format)
image = cv2.imread("comb_image.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Method 1: Automatic detection (finds all objects, selects largest)
comb_mask, masks = generator.generate_comb_mask_auto(
    image_rgb,
    select_largest=True,
    min_area_ratio=0.3
)

# Method 2: Box-prompted detection (faster if frame position is consistent)
comb_mask, metadata = generator.generate_comb_mask_box(
    image_rgb,
    margin_ratio=0.05  # 5% margin from edges
)

# Visualize results
fig, axes = generator.visualize_detection(image_rgb, comb_mask, masks)
plt.show()

# Save mask
generator.save_mask(comb_mask, "output/comb_mask_sam.png")
```

### SAM Parameters

**`CombMaskGeneratorSAM.__init__()`**:
- `model_type`: "vit_h" (best), "vit_l" (balanced), or "vit_b" (fastest)
- `checkpoint_path`: Path to downloaded .pth file
- `device`: "cuda" or "cpu"
- `points_per_side`: Grid density for auto generation (16, 32, 64)
- `pred_iou_thresh`: IoU threshold for filtering (0.7-0.95)
- `stability_score_thresh`: Stability threshold (0.7-0.99)

**`generate_comb_mask_auto()` parameters**:
- `select_largest`: Keep only largest region (typical for single comb)
- `min_area_ratio`: Minimum mask size (0.3 = 30% of image)

**`generate_comb_mask_box()` parameters**:
- `box`: Manual box [x1, y1, x2, y2] or None for auto
- `margin_ratio`: Margin from edges when auto-detecting (0.05 = 5%)

### Tips for SAM

1. **Start with automatic detection** to see what SAM finds
2. **Use vit_h for best results**, vit_b for prototyping
3. **Increase `min_area_ratio`** (e.g., 0.5) to filter small detections
4. **Use box prompts** if frame position is consistent (faster)
5. **Combine approaches**: SAM for frame → segmentation for cells

---

## Tips (Segmentation Approach)

1. **For cleaner results**: Use `infer_without_bees()` to get bee-free segmentation first
2. **Adjust closing size**: Experiment with `closing_kernel_size` (5-9 works well for most images)
3. **Check statistics**: Use `get_comb_statistics()` to verify reasonable comb coverage
4. **Multiple regions**: If `num_connected_components > 1`, you have separate comb sections
5. **Outlier removal**: Enable `remove_outliers=True` and tune `min_region_size`

---

# CombMaskEvaluator - Performance Evaluation

Evaluate binary comb mask predictions (from either approach) against ground truth labels.

## Features

✅ **Comprehensive Metrics**:
- Accuracy, Precision, Recall, F1-Score
- IoU (Intersection over Union)
- Dice Coefficient
- Specificity
- Confusion Matrix (TP, TN, FP, FN)

✅ **Visualization**:
- Side-by-side comparison (GT vs Prediction)
- Error map (color-coded: Green=TP, Red=FP, Blue=FN)

✅ **Batch Processing**:
- Evaluate multiple images at once
- Aggregate statistics (mean, std, min, max)
- Save results with meaningful names

✅ **PyTorch-based**: Efficient computation using PyTorch tensors

✅ **Works with both approaches**: Evaluates masks from segmentation or SAM

## Usage Example

### Single Image Evaluation

```python
from segmentation_restruct.comb_limitor import CombMaskGenerator, CombMaskEvaluator

# Generate prediction (outputs 0/1 format - matches GT!)
generator = CombMaskGenerator(closing_kernel_size=7)
predicted_mask = generator.generate_comb_mask(segmentation_mask)

# Load ground truth (0/1 format from CVAT labels)
gt_mask = cv2.imread('ground_truth.png', cv2.IMREAD_GRAYSCALE)

# Evaluate - both are now in the same 0/1 format!
evaluator = CombMaskEvaluator()
metrics = evaluator.evaluate(predicted_mask, gt_mask)

# Display results
evaluator.print_metrics(metrics)
# Output:
# ============================================================
#                    Evaluation Metrics
# ============================================================
#
# 📊 Main Metrics:
#   • Accuracy:    0.9856  (98.56%)
#   • IoU:         0.8923  (89.23%)
#   • Dice:        0.9431  (94.31%)
#   • F1 Score:    0.9431  (94.31%)
#
# 🎯 Precision & Recall:
#   • Precision:   0.9512  (95.12%)
#   • Recall:      0.9352  (93.52%)
#   • Specificity: 0.9921  (99.21%)
#
# 🔢 Confusion Matrix:
#   • True Positives  (TP): 1,234,567
#   • True Negatives  (TN): 3,456,789
#   • False Positives (FP): 54,321
#   • False Negatives (FN): 87,654
```

### Visualization

```python
# Visualize prediction vs ground truth with error map
fig, axes = evaluator.visualize_prediction(
    predicted_mask,
    gt_mask,
    input_image=original_image,
    show_metrics=True
)
plt.show()
```

**Error Map Colors**:
- 🟢 **Green**: True Positives (correct comb prediction)
- 🔴 **Red**: False Positives (over-segmentation - predicted comb, actually background)
- 🔵 **Blue**: False Negatives (under-segmentation - predicted background, actually comb)
- ⚫ **Black**: True Negatives (correct background)

### Batch Evaluation

#### Option 1: From Files (when predictions are already saved)

```python
# Evaluate all images in directories
aggregate_metrics, per_image_metrics = evaluator.evaluate_batch(
    predicted_dir='output/predicted_masks/',
    ground_truth_dir='data/ground_truth_labels/'
)

# Display aggregate statistics
evaluator.print_metrics(aggregate_metrics)
# Output:
# 📈 Aggregate Statistics (n=50 images):
#   • IoU:      0.8923 ± 0.0234
#   • F1 Score: 0.9431 ± 0.0156
#   • Accuracy: 0.9856 ± 0.0078

# Access per-image results
for img_metrics in per_image_metrics:
    print(f"{img_metrics['filename']}: IoU = {img_metrics['iou']:.3f}")

# Save results to JSON
evaluator.save_results(aggregate_metrics, 'results/evaluation.json')
```

#### Option 2: From In-Memory Arrays (recommended for small datasets)

```python
from honeybee_comb_inferer.inference import HoneyBeeCombInferer
from segmentation_restruct.comb_limitor import CombMaskGenerator, CombMaskEvaluator

# Initialize
model = HoneyBeeCombInferer(model_name='unet_effnetb0',
                             path_to_pretrained_models='models/',
                             device='cuda')
generator = CombMaskGenerator(closing_kernel_size=7)
evaluator = CombMaskEvaluator()

# Process all images
image_files = sorted(Path('images/').glob('*.png'))
predicted_masks = []
ground_truth_masks = []
image_names = []

for img_file in image_files:
    # Inference
    seg_mask = model.infer(str(img_file))
    pred_mask = generator.generate_comb_mask(seg_mask)

    # Load GT
    gt_file = Path('labels') / img_file.name
    gt_mask = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)

    # Store
    predicted_masks.append(pred_mask)
    ground_truth_masks.append(gt_mask)
    image_names.append(img_file.name)

# Evaluate all at once (no file I/O!)
aggregate, per_image = evaluator.evaluate_batch_from_arrays(
    predicted_masks, ground_truth_masks, image_names
)

evaluator.print_metrics(aggregate)

# Visualize all results in a grid
fig, axes = evaluator.visualize_batch(
    predicted_masks, ground_truth_masks, image_names
)
plt.show()
```

## Metrics Explained

| Metric | Formula | Meaning | Best Value |
|--------|---------|---------|------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness | 1.0 (100%) |
| **Precision** | TP / (TP + FP) | How many predicted comb pixels are correct | 1.0 (100%) |
| **Recall** | TP / (TP + FN) | How many actual comb pixels were found | 1.0 (100%) |
| **F1 Score** | 2×(Prec×Rec) / (Prec+Rec) | Harmonic mean of precision & recall | 1.0 (100%) |
| **Dice Coefficient** | 2×TP / (2×TP + FP + FN) | **Same as F1 Score** (identical formula) | 1.0 (100%) |
| **IoU** | TP / (TP + FP + FN) | Overlap between prediction & ground truth | 1.0 (100%) |
| **Specificity** | TN / (TN + FP) | True negative rate | 1.0 (100%) |

### Confusion Matrix

```
                   Ground Truth
                 Comb    Background
Predicted  Comb   TP        FP
        Background FN        TN
```

- **TP (True Positive)**: Correctly predicted comb pixels
- **TN (True Negative)**: Correctly predicted background pixels
- **FP (False Positive)**: Background incorrectly predicted as comb (over-segmentation)
- **FN (False Negative)**: Comb incorrectly predicted as background (under-segmentation)

## Ground Truth Format

Your ground truth masks should be:
- Binary images (PNG, JPG, etc.)
- **0** = background/non-comb pixels
- **1** = comb pixels (values 0 and 1, NOT 0 and 255!)
- Same dimensions as predicted masks

**Format Consistency**: Both predictions and ground truth now use **0/1 format** ✅
- **Ground Truth**: 0/1 (from CVAT labels after color conversion)
- **Predictions**: 0/1 (from `generate_comb_mask()`)
- **Direct comparison** without format conversion needed!

The evaluator can still handle 0/255 format if needed (auto-detects), but both
predicted and ground truth masks are now consistently 0/1.

### Example: Creating Ground Truth from CVAT

If you export from CVAT with RGB colors, convert to binary 0/1 format:

```python
import cv2
import numpy as np
from pathlib import Path

# Define your CVAT label colors
label_colors = {
    (0, 0, 0): 0,        # Background → 0
    (61, 245, 61): 1,    # Comb → 1
}

mask_dir = Path("cvat_masks_download/SegmentationClass")
out_dir = Path("labels")
out_dir.mkdir(exist_ok=True)

for mask_file in mask_dir.glob("*.png"):
    rgb = cv2.imread(str(mask_file))
    h, w, _ = rgb.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Convert RGB colors to class indices
    for color, index in label_colors.items():
        match = np.all(rgb == color, axis=-1)
        mask[match] = index

    # Save as binary 0/1 mask
    cv2.imwrite(str(out_dir / mask_file.name), mask)
    print(f"✓ Converted {mask_file.name}: values {np.unique(mask)}")
```

Example ground truth location:
```
E:\Bachelorarbeit\comb_limitation_dataset\labels\
    ├── image_001.png  (binary: 0=background, 1=comb)
    ├── image_002.png
    └── ...
```

## Parameters

### `CombMaskEvaluator.__init__(threshold=None)`

- **`threshold`** (Optional[int]): Binarization threshold
  - Default: `None` (auto-detects format) ✅ **Recommended**
  - If `None`: Automatically detects whether mask is 0/1 or 0/255
  - Manual options:
    - `threshold=0`: For 0/1 masks (ground truth)
    - `threshold=127`: For 0/255 masks (predictions)

**How Auto-Detection Works**:
```python
# Checks max value in mask
if max_value <= 1:
    # Mask is already 0/1 format (ground truth)
    return mask as-is
else:
    # Mask is 0/255 format (prediction)
    threshold = max_value // 2  # Use 127 for 0/255
    return mask > threshold
```

## Methods

### `evaluate(predicted_mask, ground_truth_mask)`
Evaluate single mask pair, returns metrics dict.

### `evaluate_batch(predicted_dir, ground_truth_dir)`
Evaluate multiple images from disk directories.

### `evaluate_batch_from_arrays(predicted_masks, ground_truth_masks, image_names)`
Evaluate multiple images from in-memory arrays (no file I/O). **Recommended for small datasets**.

### `visualize_prediction(predicted, ground_truth, input_image)`
Create visualization with error map for a single image.

### `visualize_batch(predicted_masks, ground_truth_masks, image_names, input_images)`
Create grid visualization for multiple images. Shows input, GT, prediction, and error map for each.

### `print_metrics(metrics)`
Pretty print metrics to console.

### `save_results(metrics, output_path)`
Save metrics to JSON file.

## Complete Workflow

```python
from honeybee_comb_inferer.inference import HoneyBeeCombInferer
from segmentation_restruct.comb_limitor import CombMaskGenerator, CombMaskEvaluator
import cv2

# 1. Segment image
model = HoneyBeeCombInferer(
    model_name='unet_effnetb0',
    path_to_pretrained_models='models/',
    device='cuda'
)
seg_mask = model.infer('image.png')

# 2. Generate binary comb mask
generator = CombMaskGenerator(closing_kernel_size=7)
comb_mask = generator.generate_comb_mask(seg_mask)

# 3. Evaluate against ground truth
gt_mask = cv2.imread('ground_truth.png', cv2.IMREAD_GRAYSCALE)
evaluator = CombMaskEvaluator()
metrics = evaluator.evaluate(comb_mask, gt_mask)

# 4. Display and save results
evaluator.print_metrics(metrics)
evaluator.visualize_prediction(comb_mask, gt_mask)
evaluator.save_results(metrics, 'results.json')
```

## Interpretation Guide

### Good Results (IoU > 0.85)
- Model accurately captures comb structure
- Minimal over/under-segmentation
- Ready for downstream tasks

### Moderate Results (IoU 0.70-0.85)
- Check error map for systematic issues
- Adjust `closing_kernel_size` parameter
- May need better segmentation model

### Poor Results (IoU < 0.70)
- Significant segmentation errors
- Review segmentation quality first
- Check ground truth alignment

### Common Issues

**High FP (Red regions)**: Over-segmentation
- Solution: Increase closing kernel size
- Solution: Exclude more cell classes

**High FN (Blue regions)**: Under-segmentation
- Solution: Decrease closing kernel size
- Solution: Include more cell classes

**Low specificity**: Background misclassified as comb
- Solution: Check segmentation model performance
- Solution: Adjust class weights in segmentation
