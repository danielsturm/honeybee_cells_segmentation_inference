from pathlib import Path
import cv2
import json
import uuid
import numpy as np
from typing import Callable

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

from segmentation_restruct.annotator.cell_finder.utils import show_cells_on_image


class CellFinder:
    def __init__(self, input_dir: Path, output_path: Path | None = None) -> None:
        self.input_dir = input_dir
        assert self.input_dir.is_dir(), "images path does not exist"
        self.output_path = output_path if output_path else self.input_dir
        assert self.output_path.is_dir(), "output path does not exist"
        self.image_paths = self._find_images()

    def run_with_template_matching(self, threshold: float = 0.725, scale_factor: float = 0.425) -> None:
        template_folder = Path(r"C:\Users\sturmd\Desktop\Bachelorarbeit\pattern_matching")
        self.run("template_matching", template_folder=template_folder, threshold=threshold, scale_factor=scale_factor)

    def run_with_hough_transform(
        self,
        dp: float = 1.2,
        min_dist: float = 20,
        param1: float = 50,
        param2: float = 30,
        min_radius: int = 5,
        max_radius: int = 50,
    ) -> None:
        self.run(
            method="circle_hough_transform",
            dp=dp,
            min_dist=min_dist,
            param1=param1,
            param2=param2,
            min_radius=min_radius,
            max_radius=max_radius,
        )

    # def run(self, method: str = "template_matching", **method_kwargs) -> None:
    #     detection_fn = self._get_detection_function(method)

    #     if not self.image_paths:
    #         print("No images found to process.")
    #         return

    #     for img_path in self.image_paths:
    #         gray_image, _ = self._load_image_and_prepare(img_path)
    #         matches = detection_fn(gray_image, **method_kwargs)
    #         filtered_matches = self._non_max_suppression(matches)
    #         self._save_cells_to_json(filtered_matches, img_path)

    def run(self, method: str = "template_matching", max_workers: int = 4, **method_kwargs) -> None:
        detection_fn, supression_fn = self._get_detection_function(method)

        if not self.image_paths:
            print("No images found to process.")
            return

        def process_image(img_path: Path):
            start_time = time.time()
            gray_image, color_image = self._load_image_and_prepare(img_path)
            matches = detection_fn(gray_image, **method_kwargs)
            filtered_matches = supression_fn(matches)
            self._save_cells_to_json(filtered_matches, img_path)
            duration = time.time() - start_time
            return img_path.name, duration, len(filtered_matches), color_image, filtered_matches

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_image, img_path) for img_path in self.image_paths]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Running {method}"):
                img_name, duration, num_cells, color_image, filtered_matches = future.result()
                tqdm.write(f"Processed {img_name} in {duration:.2f}s - found {num_cells} cells")

                show_cells_on_image(color_image, filtered_matches)

    def _get_detection_function(self, method: str) -> tuple[Callable, Callable]:
        match method:
            case "template_matching":
                return self._template_matching, self._non_max_suppression
            case "circle_hough_transform":
                return self._circle_hough_transform, self._nms_circles
            case _:
                raise ValueError(
                    f"Detection method '{method}' is not supported. "
                    f"Choose from 'template_matching', 'circle_hough_transform'."
                )

    def _template_matching(
        self,
        gray_image,
        template_folder: Path,
        threshold: float = 0.725,
        scale_factor: float = 0.425,
    ) -> list[tuple]:
        assert template_folder.is_dir(), f"Template folder does not exist: {template_folder}"

        results = []
        for template_path in template_folder.glob("*.png"):
            template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
            if template is None:
                continue

            w, h = template.shape[::-1]
            result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)

            for pt in zip(*locations[::-1]):
                score = result[pt[1], pt[0]]
                center_x = pt[0] + w // 2
                center_y = pt[1] + h // 2
                radius = int(min(w, h) * scale_factor)
                results.append((center_x, center_y, radius, score))

        return results

    def _circle_hough_transform(
        self,
        gray_image,
        dp: float = 1.2,
        min_dist: float = 20,
        param1: float = 50,
        param2: float = 30,
        min_radius: int = 5,
        max_radius: int = 50,
    ) -> list[tuple]:
        detected_circles = cv2.HoughCircles(
            gray_image,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        results = []
        if detected_circles is not None and len(detected_circles) > 0:
            detected_circles = np.around(detected_circles[0, :]).astype(np.uint16)
            for x, y, r in detected_circles:
                results.append((x, y, r, 1.0))  # Score is placeholder for Hough

        return results

    def _nms_circles(self, matches, min_dist=20):
        if len(matches) == 0:
            return []
        # Sort by score descending (if available), here always 1.0 for Hough
        matches = sorted(matches, key=lambda x: x[3], reverse=True)
        kept = []
        for x, y, r, s in matches:
            if all(np.hypot(x - xk, y - yk) >= min_dist for xk, yk, _, _ in kept):
                kept.append((x, y, r, s))
        return kept

    def _non_max_suppression(self, matches, overlap_thresh=0.3):
        if len(matches) == 0:
            return []

        boxes = np.array([[x - r, y - r, x + r, y + r] for x, y, r, _ in matches])
        scores = np.array([score for *_, score in matches])

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= overlap_thresh)[0]
            order = order[inds + 1]

        return [matches[i] for i in keep]

    def _find_images(self) -> list[Path]:
        """TODO: This needs to be extended in order to find only
        background images. see background_img_annotator.py"""
        return sorted(self.input_dir.glob("*.png"))

    def _load_image_and_prepare(self, path: Path):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return gray, color

    def _save_cells_to_json(self, results, file_path: Path):
        cell_data = [
            {
                "id": str(uuid.uuid4()),
                "center_x": int(x),
                "center_y": int(y),
                "radius": int(r),
                "label": "unlabeled",
            }
            for x, y, r, s in results
        ]
        output_file = self.output_path / f"{file_path.stem}.json"
        with open(output_file, "w") as f:
            json.dump(cell_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run CellFinder with a chosen detection method.")
    parser.add_argument(
        "method", choices=["template_matching", "circle_hough_transform"], help="Detection method to use."
    )
    parser.add_argument("input_path", type=str, help="Path to input image directory.")
    parser.add_argument(
        "--output_path", type=str, default=None, help="Optional path to output directory. Defaults to input path."
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path) if args.output_path else input_path

    cell_finder = CellFinder(input_dir=input_path, output_path=output_path)

    if args.method == "template_matching":
        cell_finder.run_with_template_matching()
    elif args.method == "circle_hough_transform":
        cell_finder.run_with_hough_transform(min_dist=40, param2=35)


if __name__ == "__main__":
    main()
