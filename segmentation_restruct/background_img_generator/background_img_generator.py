import cv2
import math
import numpy as np
import tempfile

from joblib import Parallel, delayed
from pathlib import Path, PurePath
from scipy.stats import mode
from tqdm import tqdm
from typing import List
from honeybee_comb_inferer.inference import HoneyBeeCombInferer


class BackgroundImageGenerator:
    def __init__(self, source_path: Path, output_path: Path):
        if not source_path.is_dir():
            raise NotADirectoryError(
                f"provided source path {source_path} is not a directory"
            )
        self.source_path = source_path
        if not output_path.is_dir():
            raise NotADirectoryError(
                f"provided source path {output_path} is not a directory"
            )
        self.output_path = output_path
        self.masked_img_dir, self.background_img_dir = self.create_output_dir()
        weights_path: Path = Path(__file__).parents[2] / "models"
        device = "cuda"  # TODO: move to config pydantic model
        model_name = "unet_effnetb0"  # TODO: move to config pydantic model (maybe)
        self.model = HoneyBeeCombInferer(
            model_name=model_name,
            path_to_pretrained_models=str(weights_path),
            device=device,
        )
        self.mask_out_bees()
        # self.create_background_masked_median_mode(use_median=False)

    def _find_images_by_path(self, path: Path) -> list[Path]:
        image_paths = sorted(path.glob("*.[pj][np][ge]*"))
        assert image_paths, "No images found."
        return image_paths

    def mask_out_bees(self) -> None:
        """
        Only for testing. later decide if there is an
        index to tell, wich images (the new ones) still
        have to be masked out.
        """
        # for file in self.masked_img_dir.iterdir():
        #     if file.is_file():
        #         file.unlink()

        image_files = self.find_unmasked_imgages()

        for source_img_path in tqdm(image_files):
            img = cv2.imread(source_img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: could not read {source_img_path}")
                continue
            pred_mask = self.model.infer(img, return_logits=False)
            bee_pixels = (pred_mask == 1) | (pred_mask == 8)
            img[bee_pixels] = 0
            out_path = self.masked_img_dir / f"masked_{PurePath(source_img_path).name}"
            cv2.imwrite(out_path, img)

    def find_unmasked_imgages(self) -> List[Path]:
        source_images = sorted(self.source_path.glob("*.[pj][np][ge]*"))
        masked_images = set(
            f.name.replace("masked_", "") for f in self.masked_img_dir.glob("masked_*")
        )
        unmasked_images = [
            img for img in source_images if img.name not in masked_images
        ]
        return unmasked_images

    def create_output_dir(self) -> tuple[Path, Path]:
        masked_img_dir: Path = self.output_path / "background_output" / "masked"
        Path.mkdir(masked_img_dir, parents=True, exist_ok=True)
        background_img_dir: Path = self.output_path / "background_output" / "background"
        Path.mkdir(background_img_dir, parents=True, exist_ok=True)
        return masked_img_dir, background_img_dir

    def create_background_image_version_2(self) -> None:
        images, img_name = self._load_grayscale_images(self.masked_img_dir, 10, 0)
        background = self._compute_background_image(images)
        self._save_image(background, self.background_img_dir / img_name)

    def _load_grayscale_images(
        self,
        folder: Path,
        window_size: int = 5,
        start_index: int = 0,
    ) -> list[np.ndarray]:
        image_paths = self._find_images_by_path(folder)
        selected_paths = image_paths[start_index : start_index + window_size]
        image_name = (
            selected_paths[-1].name.replace("masked", "background", 1)
            if selected_paths
            else None
        )
        return [
            cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in selected_paths
        ], image_name

    def _compute_background_image(self, images: list[np.ndarray]) -> np.ndarray:
        assert images, "No images provided."
        stacked = np.stack(images, axis=0)  # shape: (N, H, W)

        # Valid mask: non-zero pixels
        valid_mask = stacked != 0

        # Mask invalid pixels with NaN
        masked_data = np.where(valid_mask, stacked, np.nan)
        median_image = np.nanmedian(masked_data, axis=0)

        # Replace NaNs with 0 and convert to uint8
        background = np.nan_to_num(median_image, nan=0).astype(np.uint8)
        return background

    def _save_image(self, image: np.ndarray, output_path: Path) -> None:
        cv2.imwrite(str(output_path), image)
