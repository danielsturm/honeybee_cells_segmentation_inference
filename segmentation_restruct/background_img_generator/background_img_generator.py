import cv2

from pathlib import Path, PurePath
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
        self.masked_img_dir: Path = self.create_output_dir()
        weights_path: Path = Path(__file__).parents[2] / "models"
        device = "cuda"  # TODO: move to config pydantic model
        model_name = "unet_effnetb0"  # TODO: move to config pydantic model (maybe)
        self.model = HoneyBeeCombInferer(
            model_name=model_name,
            path_to_pretrained_models=str(weights_path),
            device=device,
        )
        self.mask_out_bees()

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

    def create_output_dir(self) -> Path:
        masked_img_dir: Path = self.output_path / "background_output" / "masked"
        Path.mkdir(masked_img_dir, parents=True, exist_ok=True)
        return masked_img_dir
