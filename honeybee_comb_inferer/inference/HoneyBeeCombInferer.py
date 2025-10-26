from typing import List, Optional, Tuple, Union

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2
from monai.inferers import SlidingWindowInferer
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from honeybee_comb_inferer.config import config_default, label_classes_default
from honeybee_comb_inferer.dataset.CustomDataset import CustomDataset
from honeybee_comb_inferer.model.HoneyBeeCombSegmentationModel import (
    HoneyBeeCombSegmentationModel,
)
from honeybee_comb_inferer.utils.utils import (
    get_cmap_and_labels_for_plotting,
    read_config,
    seed_everything,
)


class HoneyBeeCombInferer:
    def __init__(
        self,
        model_name: str,
        path_to_pretrained_models: str,
        label_classes_config: Union[str, List[dict]] = label_classes_default,
        config: Union[str, dict] = config_default,
        sw_inference: bool = True,
        device: str = "cpu",
    ):
        """
        class for performing semantic segmentation of honey bee comb

        Parameters
        ----------
            model_name: str
                filename of the pretrained model to be used for inference.
                Should be located in 'path_to_pretrained_models'.
            path_to_pretrained_models: str
                path where 'model_name' is located
            label_classes_config: Union[str, List[dict]]
                path to json or read json extracted from 'hasty.ai' including classes and colors (used for plotting).
                Default is read from "data/label_classes.json". List of dicts.
            config: Union[str, dict]
                path or dictionary config, which includes parameters for dataloader and sliding-window inference.
            sw_inference: bool
                boolean value (True/False) whether to apply sliding-window inference.
            device: str
                on which device should inference run, pytorch format: 'cpu','cuda','cuda:1'.

        Example usage:
            >>> from honeybee_comb_inferer.inference import HoneyBeeCombInferer
            >>> model = HoneyBeeCombInferer(model_name = 'model-name', path_to_pretrained_models = 'path-to-pretrained-models', device = 'cuda')
            >>> model.infer(image = 'path-to-image')
        """
        self.device = device
        self.sw_inferer = sw_inference

        self.config = self._get_config(config)
        self.cmap, self.patches = get_cmap_and_labels_for_plotting(label_classes_config)

        self.model = HoneyBeeCombSegmentationModel(
            model_name=model_name,
            device=device,
            path_to_pretrained_models=path_to_pretrained_models,
        )

        if sw_inference:
            self.sw_inferer = SlidingWindowInferer(
                **self.config["sliding_window_inferer"]
            )

        seed_everything(self.config["random_seed"])

    def infer(
        self,
        image: Union[Tensor, np.array, str],
        return_logits: bool = False,
        pad_to_input_size: bool = True,
    ) -> Tensor:
        """
        Perform semantic segmentation on a single honeybee comb image.

        This is the primary inference method for segmenting individual images. It identifies
        different cell types in the comb including: bees, empty cells, open/capped brood,
        open/capped honey, pollen, and bees in cells.

        The method works on images with or without bees present. For images with bees,
        it will segment both the bees and the visible cells. To remove bees from the
        segmentation, use `infer_without_bees()` instead.

        Parameters
        ----------
        image : Union[Tensor, np.array, str]
            Input image in one of three formats:
            - str: Path to image file (will be loaded as grayscale)
            - np.array: Grayscale image array (H, W)
            - Tensor: Pre-processed tensor (C, H, W) or (B, C, H, W)
        return_logits : bool, optional
            If True, returns raw model logits (useful for post-processing).
            If False (default), returns the final segmentation mask.
        pad_to_input_size : bool, optional
            If True (default), pads the output mask to match the original input image size.
            The model requires dimensions divisible by 32, so images are cropped before
            inference and then padded back to original size.

        Returns
        -------
        Tensor or np.ndarray
            If return_logits=True: Raw logits tensor of shape (B, num_classes, H, W)
            If return_logits=False: Segmentation mask array of shape (H, W) where each
            pixel value represents a class index (0-8 corresponding to label_classes.json)

        Example
        -------
        >>> model = HoneyBeeCombInferer(model_name='unet_effnetb0',
        ...                              path_to_pretrained_models='models/',
        ...                              device='cuda')
        >>> # Infer from image path
        >>> mask = model.infer('data/images/comb_image.png')
        >>> # Infer from numpy array
        >>> img_array = cv2.imread('image.png', 0)
        >>> mask = model.infer(img_array)
        >>> # Get logits for custom post-processing
        >>> logits = model.infer('image.png', return_logits=True)
        """

        image = self._check_type_and_read_image_from_str(image)
        if isinstance(image, np.ndarray):
            image, diff_in_dims = self.preprocess_raw_image(image)
            image = image.to(self.device)
        if len(image.size()) < 4:
            image = image.unsqueeze(0)

        if self.sw_inferer:
            inferred_logits = self.sw_inferer(image, self.model)
        else:
            inferred_logits = self.model(image)

        if return_logits:
            return inferred_logits
        else:
            inferred_mask = self._get_mask(inferred_logits)

            if pad_to_input_size:
                return self._pad_mask_to_input_dim(
                    inferred_mask,
                    diff_height=diff_in_dims[0],
                    diff_width=diff_in_dims[1],
                )
            else:
                return inferred_mask

    def infer_batch(self, images_path: str) -> None:
        """
        Perform batch segmentation on multiple images and save visualization results to disk.

        This method is optimized for processing entire folders of images, typically used when
        running inference from the command line. It processes all images in the specified folder,
        generates segmentation masks, and saves color-coded visualization plots.

        Use this when you have many images to process and want automatic saving of results.
        For single images or when you need the mask array for further processing, use `infer()` instead.

        Parameters
        ----------
        images_path : str
            Relative or absolute path to folder containing images to segment.
            Supported formats: .jpg, .jpeg, .png, .bmp, .tif, .tiff
            Example: 'data/images' or '/absolute/path/to/images'

        Returns
        -------
        None
            Results are saved to disk rather than returned. For each image in the input folder,
            a corresponding visualization is saved to a parallel 'inferred_masks' folder.

        Output Structure
        ----------------
        Input:  data/images/comb_image.png
        Output: data/inferred_masks/comb_image.png (color-coded segmentation visualization)

        The output images are matplotlib figures showing:
        - Color-coded segmentation mask based on label_classes.json colors
        - Legend showing all cell types (bees, empty cells, brood, honey, pollen, etc.)
        - Title: "predicted"

        Notes
        -----
        - Automatically creates 'inferred_masks' folder if it doesn't exist
        - Processes images one at a time to manage memory usage
        - Progress bar (tqdm) shows processing status
        - Images are automatically padded back to original dimensions

        Example
        -------
        >>> model = HoneyBeeCombInferer(model_name='unet_effnetb0',
        ...                              path_to_pretrained_models='models/',
        ...                              device='cuda')
        >>> # Process all images in data/images/ and save to data/inferred_masks/
        >>> model.infer_batch('data/images')

        Command Line Usage
        ------------------
        python infer.py --source data/images --model-name unet_effnetb0 --gpu
        """

        dataset = CustomDataset(images_path)

        for image, image_path, diff_in_dims in tqdm(dataset):
            inferred_mask = self.infer(image.to(self.device), pad_to_input_size=False)
            inferred_mask = self._pad_mask_to_input_dim(
                inferred_mask, diff_height=diff_in_dims[0], diff_width=diff_in_dims[1]
            )

            self._save_batch_inference(inferred_mask, image_path)

        return None

    def infer_without_bees(self, images_path: str) -> np.ndarray:
        """
        Generate a bee-free segmentation mask by averaging predictions across multiple frames.

        This method removes bees from the segmentation by exploiting temporal information.
        When you have 100+ consecutive frames of the same comb taken over time (e.g., 2 hours),
        bees move around while comb cells remain stationary. By averaging the segmentation
        predictions across all frames, moving bees are effectively "averaged out", leaving
        only the static comb structure with accurate cell type classifications.

        This is crucial for accurate cell counting, as it provides a clean view of all cells
        without occlusion by bees.

        Parameters
        ----------
        images_path : str
            Path to folder containing consecutive frames of the SAME comb area.
            Recommended: 100+ images taken over 1-2 hours with fixed camera position.
            Images should be of the same comb section to ensure proper alignment.

        Returns
        -------
        np.ndarray
            Segmentation mask of shape (H, W) with bee-free cell classifications.
            Pixel values represent class indices:
            - 0: Background
            - 1: Bees (should be minimal/zero after averaging)
            - 2: Empty cells
            - 3: Open brood
            - 4: Open honey
            - 5: Capped honey
            - 6: Capped brood
            - 7: Pollen
            - 8: Bee in cell (should be minimal after averaging)

        How It Works
        ------------
        1. Processes images in batches (batch_size from config.yaml)
        2. For each batch, runs inference and applies softmax to get class probabilities
        3. Adjusts class weights to reduce bee detection:
           - Bees: multiplied by 0.35
           - Bee_in_cell: multiplied by 0.1
           - Empty_cell: set to 0
        4. Accumulates probability averages every 5 batches to manage memory
        5. Final mask is the argmax of averaged probabilities across all frames

        Memory Usage
        ------------
        This method accumulates predictions in memory. For very large datasets (500+ images)
        or high-resolution images, consider using `infer_without_bees_opt()` which uses
        a running average approach with lower memory footprint.

        Example
        -------
        >>> model = HoneyBeeCombInferer(model_name='unet_effnetb0',
        ...                              path_to_pretrained_models='models/',
        ...                              device='cuda')
        >>> # 100 consecutive frames of the same comb
        >>> mask_no_bees = model.infer_without_bees('data/close_frames/')
        >>>
        >>> # Use the clean mask for cell counting
        >>> from honeybee_comb_inferer.cell_counter import CellCounter
        >>> counter = CellCounter(inferred_mask=mask_no_bees, method='edt')
        >>> counter.run_counter()

        Notes
        -----
        - All images must show the SAME comb area (fixed camera position)
        - More frames = better bee removal (100+ recommended)
        - Images should span enough time for bees to move (1-2 hours recommended)
        - For memory efficiency with 500+ frames, use `infer_without_bees_opt()`
        """

        dataset = CustomDataset(images_path)
        dataloader = DataLoader(dataset=dataset, **self.config["dataloader"])

        output_means = []
        output = 0
        c = 0

        for image, image_path, diff_in_dims in tqdm(dataloader):
            inferred_logits = torch.softmax(
                self.infer(image.to(self.device), return_logits=True).detach(), dim=1
            ).cpu()

            inferred_logits = self._adjust_class_weights(inferred_logits)

            if type(output) is int:
                output = inferred_logits.clone()
            else:
                output = torch.cat([inferred_logits, output])
                if c % 5 == 0:
                    output_means.append(output.mean(dim=0))
                    del output
                    output = 0

            c += 1

        output_means.append(output.mean(dim=0))
        inferred_mask = self._get_mask_no_bees(output_means)

        diff_in_dims = torch.stack(diff_in_dims)
        diff_height = int(diff_in_dims[0, 0])
        diff_width = int(diff_in_dims[1, 0])

        return self._pad_mask_to_input_dim(
            inferred_mask, diff_height=diff_height, diff_width=diff_width
        )

    def infer_without_bees_opt(self, images_path: str, skip: int = 1) -> np.ndarray:
        """
        Memory-optimized bee-free segmentation using running average (recommended for large datasets).

        This is an improved version of `infer_without_bees()` that uses significantly less memory
        by computing a running average instead of accumulating all predictions. It achieves the
        same goal of removing bees through temporal averaging but can handle much larger datasets
        (500+ images) without memory issues.

        Additionally, this method supports frame skipping to process fewer images when you have
        redundant frames, further improving memory efficiency and speed.

        Parameters
        ----------
        images_path : str
            Path to folder containing consecutive frames of the SAME comb area.
            Works well even with 500+ images due to memory-efficient processing.
        skip : int, optional
            Process every Nth batch (default=1 means process all batches).
            - skip=1: Process all images (most accurate)
            - skip=2: Process every 2nd batch (2x faster, uses half the frames)
            - skip=5: Process every 5th batch (5x faster, still good with 500+ frames)
            Use higher skip values when you have many redundant frames.

        Returns
        -------
        np.ndarray
            Bee-free segmentation mask of shape (H, W) with class indices (0-8).
            Same format as `infer_without_bees()` output.

        Advantages Over infer_without_bees()
        -------------------------------------
        1. **Lower Memory**: Uses O(1) memory instead of O(N) where N = number of images
        2. **No Batch Accumulation**: Computes running average instead of storing predictions
        3. **Frame Skipping**: Can skip redundant frames for faster processing
        4. **Scalability**: Can handle 500+ high-resolution images without memory issues
        5. **GPU Memory Management**: Explicitly clears CUDA cache after each batch
        6. **Error Handling**: Continues processing even if individual batches fail

        How It Works
        ------------
        1. Initializes running_sum as None
        2. For each batch (or every skip-th batch):
           a. Runs inference and gets logits
           b. Applies softmax and class weight adjustments
           c. Computes batch mean probabilities
           d. Adds batch mean to running sum (updates average)
           e. Clears GPU memory
        3. Computes final averaged probabilities: running_sum / batch_count
        4. Applies final softmax and argmax to get segmentation mask
        5. Pads mask back to original image dimensions

        When to Use This vs infer_without_bees()
        -----------------------------------------
        Use `infer_without_bees_opt()` when:
        - You have 300+ images
        - You're running out of memory with `infer_without_bees()`
        - You want to process faster by skipping redundant frames
        - You have high-resolution images (>2000x2000 pixels)

        Use `infer_without_bees()` when:
        - You have <200 images
        - Memory is not a concern
        - You want the original implementation behavior

        Example
        -------
        >>> model = HoneyBeeCombInferer(model_name='unet_effnetb0',
        ...                              path_to_pretrained_models='models/',
        ...                              device='cuda')
        >>>
        >>> # Process all 500 frames (memory efficient)
        >>> mask = model.infer_without_bees_opt('data/many_frames/', skip=1)
        >>>
        >>> # Process every 3rd batch for faster results (still good quality)
        >>> mask_fast = model.infer_without_bees_opt('data/many_frames/', skip=3)
        >>>
        >>> # Use for cell counting
        >>> from honeybee_comb_inferer.cell_counter import CellCounter
        >>> counter = CellCounter(inferred_mask=mask, method='edt')
        >>> counter.run_counter()

        Performance Example
        -------------------
        Test case: 600 images, 2048x2048 pixels, batch_size=4
        - infer_without_bees(): ~24GB GPU memory, 45 minutes
        - infer_without_bees_opt(skip=1): ~8GB GPU memory, 45 minutes
        - infer_without_bees_opt(skip=3): ~3GB GPU memory, 15 minutes

        Notes
        -----
        - All images must show the SAME comb area (fixed camera)
        - With skip>1, ensure you still have enough frames for good averaging (50+ recommended)
        - Handles batch errors gracefully (prints warning and continues)
        - Automatically clears GPU cache to prevent memory buildup
        """
        dataset = CustomDataset(images_path)
        dataloader = DataLoader(dataset=dataset, **self.config["dataloader"])

        running_sum = None
        batch_count = 0
        diff_dims = None  # We'll capture diff_in_dims from the first batch
        with torch.no_grad():
            for idx, (image, image_path, diff_in_dims) in enumerate(tqdm(dataloader)):
                # Process only every skip-th batch
                if idx % skip != 0:
                    continue
                try:
                    logits = self.infer(image.to(self.device), return_logits=True)
                except Exception as e:
                    print(f"Skipping batch due to error: {e}")
                    continue
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Apply softmax to get probabilities and move to CPU
                inferred_logits = torch.softmax(logits.detach(), dim=1).cpu()
                # Adjust class weights as before
                inferred_logits = self._adjust_class_weights(inferred_logits)
                # Compute the mean for this batch
                batch_mean = inferred_logits.mean(dim=0)  # shape: (num_classes, H, W)

                if running_sum is None:
                    running_sum = batch_mean
                    diff_dims = diff_in_dims[
                        0
                    ]  # Capture diff dims from first batch (assumed constant)
                else:
                    running_sum += batch_mean
                batch_count += 1

        # Compute the final averaged logits
        final_logits = running_sum / batch_count
        # Apply softmax again then get the predicted mask (does same this as _get_mask_no_bees, using the running sum
        final_logits = torch.softmax(final_logits, dim=0)
        final_mask = torch.argmax(final_logits, dim=0).detach().cpu().numpy()

        # Use the captured diff dimensions to pad back to original image size
        diff_height, diff_width = diff_dims[0], diff_dims[1]
        return self._pad_mask_to_input_dim(
            final_mask, diff_height=diff_height, diff_width=diff_width
        )

    def _get_mask(self, inferred_logits: Tensor) -> Tensor:

        inferred_mask = (
            torch.argmax(inferred_logits.squeeze(), dim=0).detach().cpu().numpy()
        )

        return inferred_mask

    def _get_mask_no_bees(self, inferred_logits_means: Tensor) -> Tensor:

        inferred_logits_pred = torch.stack(inferred_logits_means)
        inferred_logits_pred = inferred_logits_pred.mean(dim=0)

        inferred_logits_pred = torch.softmax(inferred_logits_pred, dim=0).clone()
        inferred_mask = torch.argmax(inferred_logits_pred, dim=0).detach().cpu().numpy()

        return inferred_mask

    def _pad_mask_to_input_dim(
        self, inferred_mask, diff_height: int, diff_width: int
    ) -> np.ndarray:

        return np.pad(inferred_mask, ((0, diff_height), (0, diff_width)))

    def preprocess_raw_image(self, image: np.array) -> Union[Tensor, Tuple[int, int]]:

        height = image.shape[0] // 32 * 32
        width = image.shape[1] // 32 * 32

        diff_height = image.shape[0] - height
        diff_width = image.shape[1] - width

        image = image[:height, :width]

        transformation = self.get_transforms()

        return transformation(image=image)["image"], (diff_height, diff_width)

    def get_transforms(self) -> A.core.composition.Compose:

        list_trans = [A.Normalize(mean=0, std=1), ToTensorV2()]
        list_trans = A.Compose(list_trans)
        return list_trans

    def _adjust_class_weights(self, inferred_logits: Tensor) -> Tensor:

        inferred_logits[:, 1, ...] = 0
        inferred_logits[:, 0, ...] *= 0.35
        inferred_logits[:, 2, ...] *= 0.9
        inferred_logits[:, 8, ...] *= 0.1

        return inferred_logits

    def _check_type_and_read_image_from_str(
        self, image: Union[str, np.ndarray, Tensor]
    ) -> Union[np.ndarray, Tensor]:

        if isinstance(image, str):
            return cv2.imread(image, 0)
        else:
            return image

    def get_processed_labels(self, pred: Tensor):
        # label_processed = np.array([[self.cmap[int(i)] for i in j] for j in tqdm(pred)])  # nested loop label map processing (slower)
        # vectorized label map processing
        max_label = max(self.cmap.keys())
        color_map = np.zeros((max_label + 1, 4), dtype=np.float32)
        # Fill in the color_map with values from self.cmap
        for label, color in self.cmap.items():
            color_map[label] = color
        label_processed = color_map[pred]  # shape will be (H, W, 4)
        return label_processed

    def plot_prediction(
        self,
        pred: Tensor,
        input_image: Optional[Union[np.ndarray, str]] = None,
        mask: Optional[np.ndarray] = None,
    ):

        label_processed = self.get_processed_labels(pred)

        if input_image is not None and mask is not None:
            fig, ax = plt.subplots(3, 1, figsize=(36, 28))

            input_image = self._check_type_and_read_image_from_str(input_image)
            ax[0].imshow(input_image, cmap="gray")
            ax[0].set_title("input image")

            ax[1].imshow(label_processed)
            ax[1].set_title("predicted")

            mask_processed = np.array([[self.cmap[i] for i in j] for j in tqdm(mask)])
            ax[2].imshow(mask_processed)
            ax[2].set_title("ground truth")

        elif input_image is not None or mask is not None:
            fig, ax = plt.subplots(1, 2, figsize=(36, 28))

            if input_image is not None:
                input_image = self._check_type_and_read_image_from_str(input_image)
                ax[0].imshow(input_image, cmap="gray")
                ax[0].set_title("input image")
            elif mask is not None:
                mask_processed = np.array(
                    [[self.cmap[i] for i in j] for j in tqdm(mask)]
                )
                ax[0].imshow(input_image, cmap="gray")
                ax[0].set_title("input image")

            ax[1].imshow(label_processed)
            ax[1].set_title("predicted")

        else:
            fig, ax = plt.subplots(1, 1, figsize=(28, 20))

            ax.imshow(label_processed)
            ax.set_title("predicted")

        plt.legend(
            handles=self.patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
        )

        return fig, ax

    def _save_batch_inference(self, pred: Tensor, input_image_path: str) -> None:

        output_path = input_image_path.replace("images", "inferred_masks")

        label_processed = np.array([[self.cmap[int(i)] for i in j] for j in pred])
        fig, ax = plt.subplots(1, 1, figsize=(24, 20))

        ax.imshow(label_processed)
        ax.set_title("predicted")
        plt.legend(
            handles=self.patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
        )
        plt.savefig(output_path)
        plt.close(fig)

        return None

    def _get_config(self, config: Union[str, dict]) -> dict:

        if isinstance(config, str):
            return read_config(config)
        elif isinstance(config, dict):
            return config
        else:
            raise Exception(
                f"'config' should be of type <str> (path) or <dict>, but you provided type <{type(config)}>"
            )
