import napari
from napari.layers import Points
from skimage.io import imread
import json
from pathlib import Path
import numpy as np
import pandas as pd
from magicgui.widgets import ComboBox, Container


def update_point_borders(points_layer: Points) -> None:
    selected = points_layer.selected_data
    border_colors = [
        "white" if i in selected else "black" for i in range(len(points_layer.data))
    ]
    points_layer.border_color = border_colors


def export_annotated_cells(points_layer, output_path: Path):
    points = points_layer.data
    sizes = points_layer.size  # array of radii
    labels = points_layer.features["cell_type"].tolist()

    exported = []
    for (y, x), r, label in zip(points, sizes, labels):
        exported.append(
            {"center_x": int(x), "center_y": int(y), "radius": int(r), "label": label}
        )

    with open(output_path, "w") as f:
        json.dump(exported, f, indent=2)


def create_label_menu(points_layer, labels):
    label_menu = ComboBox(label="cell_type", choices=labels)
    label_widget = Container(widgets=[label_menu])

    def update_label_menu(event):
        new_label = str(points_layer.feature_defaults["cell_type"][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    points_layer.events.feature_defaults.connect(update_label_menu)

    def label_changed(selected_label):
        # Assign label to selected points
        selected = list(points_layer.selected_data)
        if not selected:
            # No selection → update feature_defaults for new points
            feature_defaults = points_layer.feature_defaults
            feature_defaults["cell_type"] = selected_label
            points_layer.feature_defaults = feature_defaults
        else:
            points_layer.features.loc[selected, "cell_type"] = selected_label
        points_layer.refresh_colors()

    label_menu.changed.connect(label_changed)
    return label_widget


def start_napari_annotation(image_path: Path, json_in_path: Path, json_out_path: Path):
    image = imread(str(image_path))
    with open(json_in_path, "r") as f:
        cells = json.load(f)

    points = []
    radii = []
    labels = []

    for cell in cells:
        cx, cy, radius = cell["center_x"], cell["center_y"], cell["radius"]
        points.append([cy, cx])
        radii.append(radius)
        labels.append("unlabeled")

    points = np.array(points)
    radii = np.array(radii, dtype=float)

    label_categories = [
        "unlabeled",
        "honey",
        "larvae",
        "pollen",
        "empty",
        "capped_honey",
    ]
    features = pd.DataFrame(
        {"cell_type": pd.Categorical(labels, categories=label_categories)}
    )

    color_map = {
        "unlabeled": "blue",
        "honey": "gold",
        "larvae": "cyan",
        "pollen": "orange",
        "empty": "green",
        "capped_honey": "red",
    }

    viewer = napari.Viewer()
    viewer.add_image(image, name="Honeycomb")
    points_layer = viewer.add_points(
        points,
        features=features,
        face_color="cell_type",
        face_color_cycle=color_map,
        size=radii,
        name="Cells",
    )
    points_layer.face_color_mode = "cycle"
    points_layer.selected_data.events.items_changed.connect(
        lambda e: update_point_borders(points_layer)
    )

    brush_mask = viewer.add_labels(
        np.zeros(image.shape[:2], dtype=np.uint8), name="BrushMask"
    )

    # @viewer.bind_key("s")
    # def select_points_in_brush(viewer=None):
    #     mask_array = brush_mask.data
    #     selected = set()
    #     for i, (y, x) in enumerate(points_layer.data):
    #         if 0 <= int(y) < mask_array.shape[0] and 0 <= int(x) < mask_array.shape[1]:
    #             if mask_array[int(y), int(x)] == 1:
    #                 selected.add(i)
    #     points_layer.selected_data = selected
    #     print(f"Selected {len(selected)} points under brush.")

    def on_brush_mask_change(event):
        if brush_mask.mode == "paint":
            mask_array = brush_mask.data
            selected = set()
            for i, (y, x) in enumerate(points_layer.data):
                if (
                    0 <= int(y) < mask_array.shape[0]
                    and 0 <= int(x) < mask_array.shape[1]
                ):
                    if mask_array[int(y), int(x)] == 1:
                        selected.add(i)

            # print(points_layer.mode)
            points_layer.selected_data = selected

            update_point_borders(points_layer)

            # === Prevent infinite recursion ===
            brush_mask.events.paint.disconnect(on_brush_mask_change)
            brush_mask.data = np.zeros_like(brush_mask.data)
            brush_mask.events.paint.connect(on_brush_mask_change)

    brush_mask.events.paint.connect(on_brush_mask_change)

    label_widget = create_label_menu(points_layer, label_categories)
    viewer.window.add_dock_widget(label_widget, area="right")

    napari.run()

    export_annotated_cells(points_layer, json_out_path)


image_path_2 = Path(
    r"C:\Users\sturmd\Desktop\Bachelorarbeit\ws=10_numimg=100_clahe=intermediate_dil=15_mdncomp=cupy_dur=233.png"
)

json_dir = Path(__file__).parents[1] / "cell_finder"
json_in_dir = json_dir / "cells.json"
json_out_dir = json_dir / "cells_annotated.json"

start_napari_annotation(image_path_2, json_in_dir, json_out_dir)
