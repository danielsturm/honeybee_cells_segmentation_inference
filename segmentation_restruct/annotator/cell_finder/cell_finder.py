import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json


def load_image_and_prepare(path: Path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return gray, color


def collect_template_matches(
    gray_image, threshold: float, scale_factor: float, template_folder: Path
):
    results = []
    for filename in os.listdir(template_folder):
        if not filename.lower().endswith((".png")):
            continue

        template_path = os.path.join(template_folder, filename)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
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


def non_max_suppression(matches, overlap_thresh=0.3):
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


def save_cells_to_json(results, output_path: Path):
    cell_data = [
        {
            "center_x": int(x),
            "center_y": int(y),
            "radius": int(r),
            "score": float(s),
        }
        for x, y, r, s in results
    ]
    with open(output_path, "w") as f:
        json.dump(cell_data, f, indent=2)


def draw_detected_cells(image_color, results):
    for center_x, center_y, radius, _ in results:
        cv2.circle(image_color, (center_x, center_y), radius, (0, 255, 0), 2)
    return image_color


def execute_cell_finder(path: Path, save_path: Path):
    threshold = 0.725
    scale_factor = 0.425

    template_folder = Path(r"C:\Users\sturmd\Desktop\Bachelorarbeit\pattern_matching")
    assert path.is_file(), "is no file"
    assert template_folder.is_dir(), "is no dir"

    gray_image, image_color = load_image_and_prepare(path)
    match_results = collect_template_matches(
        gray_image, threshold, scale_factor, template_folder
    )
    filtered_results = non_max_suppression(match_results)
    save_cells_to_json(filtered_results, save_path)
    image_with_circles = draw_detected_cells(image_color, filtered_results)

    plt.figure(figsize=(15, 15))
    plt.imshow(image_with_circles)
    plt.title(f"Matches after NMS: {len(filtered_results)}")
    plt.axis("off")
    plt.show()


image_path_1 = Path(
    r"C:\Users\sturmd\Desktop\Bachelorarbeit\ws=10_numimg=100_clahe=post_dil=0_mdncomp=cupy.png"
)
image_path_2 = Path(
    r"C:\Users\sturmd\Desktop\Bachelorarbeit\ws=10_numimg=100_clahe=intermediate_dil=15_mdncomp=cupy_dur=233.png"
)

save_dir = Path(__file__).parent / "cells.json"

execute_cell_finder(image_path_2, save_dir)
