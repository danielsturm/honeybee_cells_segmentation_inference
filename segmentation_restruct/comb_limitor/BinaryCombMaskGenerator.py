from pathlib import Path
from typing import Union, Optional
import numpy as np
import cv2
from honeybee_comb_inferer.inference import HoneyBeeCombInferer


class BinaryCombMaskGenerator:

    def __init__(
        self,
        model_name: str = "combSegmentor",
        model_dir: Optional[Union[str, Path]] = None,
        label_config: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        apply_closing: bool = True,
        apply_outlier_suppression: bool = True,
        closing_kernel_size: int = 5,
        min_contour_area: int = 1000
    ):
        # Set defaults if not provided
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent.parent.parent / "models"
        if label_config is None:
            label_config = Path(__file__).resolve().parent.parent.parent / "data" / "label_classes_binary.json"

        self.apply_closing = apply_closing
        self.apply_outlier_suppression = apply_outlier_suppression
        self.closing_kernel_size = closing_kernel_size
        self.min_contour_area = min_contour_area

        self.model = self._load_model(model_name, str(model_dir), str(label_config), device)

    def _load_model(self, model_name: str, model_dir: str, label_config: str, device: str) -> HoneyBeeCombInferer:
        return HoneyBeeCombInferer(
            model_name=model_name,
            path_to_pretrained_models=model_dir,
            label_classes_config=label_config,
            device=device
        )

    def generate_comb_mask(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        # Infer raw mask
        raw_mask = self._infer(image)

        # Apply post-processing
        mask = raw_mask.copy()
        if self.apply_closing:
            mask = self._apply_closing(mask)
        if self.apply_outlier_suppression:
            mask = self._suppress_outliers(mask)

        return mask

    def _infer(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        return self.model.infer(image=str(image) if isinstance(image, Path) else image)

    def _apply_closing(self, mask: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.closing_kernel_size, self.closing_kernel_size))
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    def _suppress_outliers(self, mask: np.ndarray) -> np.ndarray:
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cleaned_mask = np.zeros_like(mask_uint8)
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_contour_area:
                cv2.drawContours(cleaned_mask, [contour], -1, 1, -1)

        return cleaned_mask
