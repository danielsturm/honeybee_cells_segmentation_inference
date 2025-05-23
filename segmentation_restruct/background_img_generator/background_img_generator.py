import cv2
import gc
import math
import matplotlib.pyplot as plt
import numpy as np
import tempfile

from joblib import Parallel, delayed
from pathlib import Path, PurePath
from scipy.stats import mode
from tqdm import tqdm
from typing import List, Literal
from honeybee_comb_inferer.inference import HoneyBeeCombInferer
from collections import deque
from typing import Deque
from segmentation_restruct.utils import timed, BgImageGenConfig

import time
import torch
import cupy as cp
import re


class BackgroundImageGenerator:
    def __init__(self, source_path: Path, output_path: Path, config: BgImageGenConfig):
        self._config = config
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
        self.model = HoneyBeeCombInferer(
            model_name=self._config.segmentation_model,
            path_to_pretrained_models=str(weights_path),
            device=self._config.device,
        )
        # self.mask_out_bees()
        self._memmap_file = Path(tempfile.gettempdir()) / "rolling_medians.dat"
        self._rolling_memmap = None

    # def _find_images_by_path(self, path: Path) -> list[Path]:
    #     return sorted(path.glob("*.[pj][np][ge]*"))
    def _find_images_by_path(
        self,
        path: Path,
        role: Literal["background", "masked"],
    ) -> list[Path]:
        # pattern_template = r"^{prefix}_(\d{{8}}T\d{{6}}\.\d{{6}}\.\d{{3}}Z)\.png$"
        # pattern_template = r"^{prefix}_(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{3}}Z)\.png$"
        pattern_template = (
            r"^{prefix}_cam-\d_(\d{{8}}T\d{{6}}\.\d{{1,6}}\.\d{{3}}Z)\.png$"
        )
        regex = re.compile(pattern_template.format(prefix=role))

        all_images = sorted(path.glob("*"))
        return [img for img in all_images if regex.match(img.name)]

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
            img = cv2.imread(str(source_img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: could not read {source_img_path}")
                continue
            pred_mask = self.model.infer(img, return_logits=False)
            bee_pixels = (pred_mask == 1) | (pred_mask == 8)
            # img[bee_pixels] = 0
            refined_mask = self._refine_mask(bee_pixels)
            img[refined_mask > 0] = 0
            out_path = self.masked_img_dir / f"masked_{PurePath(source_img_path).name}"
            cv2.imwrite(str(out_path), img)

    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        if not self._config.mask_dilation:
            return mask
        else:
            kernel_size = (self._config.mask_dilation, self._config.mask_dilation)
            kernel = np.ones(kernel_size, np.uint8)
            return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)

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

    def _read_image(self, filepath: Path) -> cv2.typing.MatLike:
        return cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)

    @timed("Rolling Background Generation")
    def process_rolling_backgrounds(
        self,
        sampling_rate: int,
        tile_size=(512, 512),
        use_median=True,
    ) -> None:
        masked_images = self._find_images_by_path(self.masked_img_dir, role="masked")
        print(self.masked_img_dir)
        if not masked_images:
            print("No masked images found")
            return
        background_images = self._find_images_by_path(
            self.background_img_dir, role="background"
        )

        image_queue: Deque[tuple[np.ndarray, Path]] = deque()

        last_processed_img_name = (
            background_images[-1].name.replace("background", "masked")
            if background_images
            else None
        )

        start_idx = 0
        if last_processed_img_name:
            try:
                last_index = masked_images.index(
                    self.masked_img_dir / last_processed_img_name
                )
                start_idx = last_index + sampling_rate
            except ValueError:
                print(f"Could not find masked image {last_processed_img_name}")
                return

        sampled_masked_paths = masked_images[start_idx::sampling_rate]

        if len(sampled_masked_paths) < self._config.window_size:
            print(
                f"Not enough images left for window. Found {len(sampled_masked_paths)}"
            )
            return

        if self._config.num_median_images:
            total_possible = len(sampled_masked_paths) - self._config.window_size + 1
            num_medians = min(self._config.num_median_images, total_possible)
            sampled_masked_paths = sampled_masked_paths[
                : num_medians + self._config.window_size - 1
            ]
        else:
            num_medians = len(sampled_masked_paths) - self._config.window_size + 1

        print(f"Will compute {num_medians} rolling median frames")

        # First image for shape
        first_img = self._read_image(sampled_masked_paths[0])
        H, W = first_img.shape
        assert H > 0 and W > 0, f"Invalid shape: H={H}, W={W}"

        self._rolling_memmap = np.memmap(
            self._memmap_file, dtype="uint8", mode="w+", shape=(num_medians, H, W)
        )

        for path in sampled_masked_paths[: self._config.window_size - 1]:
            img = self._read_image(path)
            image_queue.append((img, path))

        median_index = 0
        paths_to_process = sampled_masked_paths[self._config.window_size - 1 :]

        for path in tqdm(paths_to_process, desc="Rolling median"):
            if median_index >= num_medians:
                break
            next_img = self._read_image(path)
            image_queue.append((next_img, path))
            if len(image_queue) == self._config.window_size:
                window_imgs = [img for img, _ in image_queue]
                match self._config.median_computation:
                    case "cuda_support":
                        background = self._compute_background_image_cuda_support(
                            window_imgs, self._config.device
                        )
                    case "cupy":
                        background = self._compute_background_image_cupy(window_imgs)
                    case "masked_array":
                        background = self._compute_background_image(window_imgs)
                if self._config.apply_clahe == "intermediate":
                    background = self._apply_clahe(background)
                self._rolling_memmap[median_index, :, :] = background
                median_index += 1
                image_queue.popleft()

        self._rolling_memmap.flush()
        print("Rolling medians written to disk.")

        # === Tile-wise global median ===
        print("Starting global background computation by tile...")

        del self._rolling_memmap
        gc.collect()

        self._rolling_memmap = np.memmap(
            self._memmap_file, dtype="uint8", mode="r", shape=(num_medians, H, W)
        )

        background = np.zeros((H, W), dtype=np.uint8)
        results = Parallel(n_jobs=8)(
            delayed(self._process_tile_stack)(
                self._rolling_memmap, i, j, tile_size, use_median
            )
            for i in range(0, H, tile_size[0])
            for j in range(0, W, tile_size[1])
        )

        for i, i_end, j, j_end, tile_result in results:
            background[i:i_end, j:j_end] = tile_result

        if self._config.apply_clahe == "post":
            background = self._apply_clahe(background)

        # bg_img_name = sampled_masked_paths[0].name.replace("masked", "background")
        original_path = sampled_masked_paths[0]
        bg_img_name = f"{self._tmp_config_name_gen()}{original_path.suffix}"
        self._save_image(background, self.background_img_dir / bg_img_name)
        print("Final background saved to:", self.background_img_dir / bg_img_name)

        self._cleanup_memory()

    def _tmp_config_name_gen(self):
        config = self._config
        name = f"ws={config.window_size}_numimg={config.num_median_images}_clahe={config.apply_clahe}_dil={config.mask_dilation}_mdncomp={config.median_computation}"
        return name

    # @timed("Tile Processing")
    def _process_tile_stack(
        self,
        stack: np.memmap,
        i: int,
        j: int,
        tile_size: tuple[int, int],
        use_median: bool,
    ) -> tuple[int, int, int, int, np.ndarray]:
        H, W = stack.shape[1:]  # (N, H, W)
        i_end = min(i + tile_size[0], H)
        j_end = min(j + tile_size[1], W)

        tile_stack = stack[:, i:i_end, j:j_end]
        N, th, tw = tile_stack.shape
        tile_flat = tile_stack.reshape(N, -1)
        out_tile = np.zeros((th * tw,), dtype=np.uint8)

        for k in range(th * tw):
            pixel_values = tile_flat[:, k]
            nonzero = pixel_values[pixel_values != 0]
            if len(nonzero) == 0:
                out_tile[k] = 0
            else:
                if use_median:
                    out_tile[k] = np.median(nonzero).astype(np.uint8)
                else:
                    val, _ = mode(nonzero, keepdims=True)
                    out_tile[k] = val[0].astype(np.uint8)

        return i, i_end, j, j_end, out_tile.reshape(th, tw)

    def _cleanup_memory(self):
        del self._rolling_memmap
        gc.collect()
        if self._memmap_file.exists():
            try:
                self._memmap_file.unlink()
                print(f"Deleted temporary memmap file {self._memmap_file}")
            except PermissionError as e:
                print(f"Could not delete memmap file: {e}")

    def _compute_background_image(self, images: list[np.ndarray]) -> np.ndarray:
        assert images, "No images provided."
        stacked = np.stack(images, axis=0)  # shape: (N, H, W)

        masked = np.ma.masked_equal(stacked, 0)
        median = np.ma.median(masked, axis=0).filled(0).astype(np.uint8)
        return median

    def _compute_background_image_cuda_support(
        self, images: list[np.ndarray], device: str = "cpu"
    ) -> np.ndarray:
        assert images, "No images provided."

        tensors = [
            torch.from_numpy(img).to(device=device, dtype=torch.float32)
            for img in images
        ]
        stacked = torch.stack(tensors, dim=0)

        mask = stacked != 0
        stacked[~mask] = float("nan")

        median = torch.nanmedian(stacked, dim=0).values

        return median.nan_to_num(0).byte().cpu().numpy()

    def _compute_background_image_cupy(self, images: list[np.ndarray]) -> np.ndarray:
        assert images, "No images provided."

        stacked = cp.stack([cp.asarray(img, dtype=cp.uint8) for img in images], axis=0)
        stacked = stacked.astype(cp.float32)
        stacked[stacked == 0] = cp.nan

        median = cp.nanmedian(stacked, axis=0)

        result = cp.nan_to_num(median, nan=0).round().clip(0, 255).astype(cp.uint8)

        return cp.asnumpy(result)

    def _apply_clahe(self, img, clipLimit=2.0, tileGridSize=(8, 8)):

        if img.dtype in [np.float32, np.float64]:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
        return clahe.apply(img)

    def _save_image(self, image: np.ndarray, output_path: Path) -> None:
        cv2.imwrite(str(output_path), image)

    def create_background_masked_median_mode(
        self,
        # folder,
        window_size=10,
        tile_size=(512, 512),
        sampling_rate=5,
        use_median=True,
    ):

        file_list = self._find_images_by_path(self.masked_img_dir, role="masked")
        if not file_list:
            print("no files found")
            return
        # Filter out any existing background or unneeded files
        file_list = [f for f in file_list if "background" not in f.name.lower()]
        file_list = file_list[::sampling_rate]
        file_name = file_list[-1].name
        num_files = len(file_list)
        print("Number of masked images:", num_files)

        if num_files < window_size:
            raise ValueError(
                "Not enough images to apply rolling median; adjust window_size or sampling_rate."
            )

        def read_image(filepath) -> cv2.typing.MatLike:
            return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Read first image to get shape
        first_img = read_image(file_list[0])
        if first_img is None:
            raise ValueError("Cannot read the first masked image.")
        H, W = first_img.shape
        print(f"Image shape: {H} x {W}")

        num_medians = num_files - window_size + 1
        print("Number of median images to produce:", num_medians)

        # Create a memmap file to hold the 'median images'
        memmap_file = Path(tempfile.gettempdir()) / "median_images.dat"

        median_memmap = np.memmap(
            memmap_file, dtype="uint8", mode="w+", shape=(num_medians, H, W)
        )

        # Rolling median setup (ignoring black pixels in each window)
        window_imgs = []
        for f in file_list[: window_size - 1]:
            img = read_image(f)
            if img is not None:
                window_imgs.append(img)

        median_index = 0
        for f in tqdm(
            file_list[window_size - 1 :], desc="Computing rolling medians (masked)"
        ):
            img = read_image(f)
            if img is None:
                continue
            window_imgs.append(img)
            if len(window_imgs) == window_size:
                # Stack images: shape = (window_size, H, W)
                stack_ = np.stack(window_imgs, axis=0)
                # Create a masked array where pixels equal to 0 are masked out
                masked_stack = np.ma.masked_equal(stack_, 0)
                # Compute median along axis=0, ignoring masked (black) pixels.
                # For pixels where all values are masked, fill with 0.
                median_img = (
                    np.ma.median(masked_stack, axis=0).filled(0).astype(np.uint8)
                )
                median_memmap[median_index, :, :] = median_img
                median_index += 1
                window_imgs.pop(0)

        median_memmap.flush()
        print("Rolling median images computed.")

        # Now compute the final pixel ignoring black, either via median or mode
        median_memmap = np.memmap(
            memmap_file, dtype="uint8", mode="r", shape=(num_medians, H, W)
        )
        background = np.zeros((H, W), dtype=np.uint8)

        n_tiles_y = math.ceil(H / tile_size[0])
        n_tiles_x = math.ceil(W / tile_size[1])
        print(
            f"Processing background in {n_tiles_y} x {n_tiles_x} tiles, ignoring black=0 pixels..."
        )

        def process_tile(i, j):
            i_end = min(i + tile_size[0], H)
            j_end = min(j + tile_size[1], W)
            # Extract tile of shape (num_medians, tile_h, tile_w)
            tile_stack = median_memmap[:, i:i_end, j:j_end]
            N, th, tw = tile_stack.shape

            # Flatten each (th, tw) patch across N frames => shape (N, th*tw)
            tile_flat = tile_stack.reshape(N, -1)
            out_tile = np.zeros((th * tw,), dtype=np.uint8)

            for k in range(th * tw):
                pixel_values = tile_flat[:, k]
                # Filter out zeros
                nonzero = pixel_values[pixel_values != 0]
                if len(nonzero) == 0:
                    # No valid data => keep it black
                    out_tile[k] = 0
                else:
                    if use_median:
                        out_tile[k] = np.median(nonzero).astype(np.uint8)
                    else:
                        # Use mode from scipy, ignoring zeros
                        # The mode can return multiple values, but we only need the first
                        val, _ = mode(nonzero, keepdims=True)
                        out_tile[k] = val[0].astype(np.uint8)

            return i, i_end, j, j_end, out_tile.reshape(th, tw)

        # Parallel tile processing
        results = Parallel(n_jobs=8)(
            delayed(process_tile)(i, j)
            for i in range(0, H, tile_size[0])
            for j in range(0, W, tile_size[1])
        )

        for i, i_end, j, j_end, tile_result in results:
            background[i:i_end, j:j_end] = tile_result

        background = self._apply_clahe(background)
        # Display and save
        out_path = self.background_img_dir / f"background_{file_name}.png"
        cv2.imwrite(str(out_path), background)
        print("Masked background (ignoring black) saved to:", out_path)

        del median_memmap

        gc.collect()

        # Then safely delete the file if you want
        if memmap_file.exists():
            memmap_file.unlink()
            print(f"Deleted temporary memmap file {memmap_file}")

    def analyze_image_quality(self, folder_path: Path) -> None:
        image_files = self._find_images_by_path(folder_path, role="background")
        if not image_files:
            print("no files found")
            return
        image_path = image_files[0]
        img = self._read_image(image_path)

        # Brightness
        brightness = np.mean(img)

        # Contrast
        contrast = np.std(img)

        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = laplacian.var()

        # Histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        # Print metrics
        print(f"Analyzing: {image_path.name}")
        print(f"  Brightness (mean):  {brightness:.2f}")
        print(f"  Contrast (std):     {contrast:.2f}")
        print(f"  Sharpness (LapVar): {sharpness:.2f}")

        # Plot histogram
        plt.figure(figsize=(10, 4))
        plt.title(f"Histogram of {image_path.name}")
        plt.plot(hist, color="gray")
        plt.xlim([0, 256])
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def process_rolling_backgrounds_old(
        self, win_size: int, sampling_rate: int, stop_idx: int = 0, device: str = "cpu"
    ) -> None:

        masked_images = self._find_images_by_path(self.masked_img_dir, role="masked")
        background_images = self._find_images_by_path(
            self.background_img_dir, role="background"
        )

        image_queue: Deque[tuple[np.ndarray, Path]] = deque()

        last_processed_img_name = (
            background_images[-1].name.replace("background", "masked")
            if background_images
            else None
        )

        start_idx = 0
        if last_processed_img_name:
            try:
                last_index = masked_images.index(
                    self.masked_img_dir / last_processed_img_name
                )
                start_idx = last_index + sampling_rate
            except ValueError:
                print(f"Could not find masked image {last_processed_img_name}")
                return

        sampled_masked_paths = masked_images[start_idx::sampling_rate]
        bg_img_name = sampled_masked_paths[0].name.replace("masked", "background")

        if len(sampled_masked_paths) < win_size:
            print(
                f"Not enough images left for winow. Found {len(sampled_masked_paths)}"
            )
            return
        for path in sampled_masked_paths[: win_size - 1]:
            img = self._read_image(path)
            image_queue.append((img, path))

        paths_to_process = sampled_masked_paths[win_size - 1 :]
        if stop_idx:
            paths_to_process = paths_to_process[:stop_idx]
        for path in tqdm(paths_to_process):
            next_img = self._read_image(path)
            image_queue.append((next_img, path))
            if len(image_queue) == win_size:
                bg_img_name = image_queue[0][1].name.replace("masked", "background")
                window_imgs = [img for img, _ in image_queue]

                start = time.perf_counter()
                # background = self._compute_background_image(window_imgs)
                # background = self._compute_background_image_cuda_support(
                #     window_imgs, device
                # )
                background = self._compute_background_image_cupy(window_imgs)
                background = self._apply_clahe(background)
                print(
                    f"Background computation took {time.perf_counter() - start:.3f} s"
                )

                self._save_image(background, self.background_img_dir / bg_img_name)
                image_queue.popleft()
