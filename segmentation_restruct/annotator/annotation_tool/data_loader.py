from pathlib import Path
import numpy as np
import logging
import json
from skimage.io import imread


class DataLoader:
    def __init__(self, data_dir: Path, logger: logging.Logger) -> None:
        self.logger = logger
        self.data_dir = data_dir
        assert self.data_dir.exists(), "data dir does not exist"

        self.image_paths, self.label_paths = self._find_data(data_dir)

    def _find_data(self, data_dir: Path) -> tuple[list[Path], dict[str, Path]]:
        # TODO: Find only files with specific pattern
        image_paths = sorted(data_dir.glob("*.png"))
        label_paths = {p.stem: p for p in data_dir.glob("*.json")}
        return image_paths, label_paths

    # TODO: replace the return with AnnotationDTO
    def load_data(
        self, image_idx: int
    ) -> tuple[np.ndarray, str, list[str], np.ndarray, np.ndarray, list[str]]:
        image_path = self.image_paths[image_idx]
        image_name = image_path.stem
        image = imread(str(image_path))
        self.logger.info(f"loading image {image_path}")

        label_path = self.label_paths.get(image_path.stem)
        cells = []
        if label_path and label_path.exists():
            with open(label_path, "r") as f:
                cells = json.load(f)
        else:
            self.logger.warning(f"No labels json found for {image_path.stem}")

        """TODO: Instead use the cell dataclass and make a list of it.
        return AnnotationDTO. The DTO should provide functions to return
        ids, points etc as lists"""
        ids, points, radii, labels = [], [], [], []
        for cell in cells:
            id, cx, cy, radius, label = (
                cell["id"],
                cell["center_x"],
                cell["center_y"],
                cell["radius"],
                cell["label"],
            )
            ids.append(id)
            points.append([cy, cx])
            radii.append(radius * 2)
            labels.append(label)

        return (
            image,
            image_name,
            ids,
            np.array(points),
            np.array(radii, dtype=float),
            labels,
        )

    def export_annotated_cells(
        self,
        image_idx: int,
        ids: list[str],
        points: np.ndarray,
        radii: np.ndarray,
        labels: list[str],
    ) -> None:
        """TODO: Pass Annotation DTO instead and use the Cell data class"""
        exported = []
        for id, (y, x), r, label in zip(ids, points, radii, labels):
            exported.append(
                {
                    "id": id,
                    "center_x": int(x),
                    "center_y": int(y),
                    "radius": int(r / 2),
                    "label": label,
                }
            )

        curr_img_name = self.image_paths[image_idx].stem
        output_path = self.data_dir / f"{curr_img_name}.json"

        with open(output_path, "w") as f:
            json.dump(exported, f, indent=2)
        self.logger.info(f"Exporting {len(exported)} to {output_path}")
