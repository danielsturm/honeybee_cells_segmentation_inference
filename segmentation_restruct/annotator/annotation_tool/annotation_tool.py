import napari
from napari.layers import Points, Labels
from skimage.io import imread
import json
from pathlib import Path
import numpy as np
import pandas as pd
from magicgui.widgets import ComboBox, Container


"""
TODO:   🧽 Add key "c" to clear the BrushMask layer after selection

        👀 Add visual feedback (e.g., print selected labels/counts)

        📝 Add save button in GUI instead of saving only on close

        🖱️ Add tooltips or points_layer.text to show the label on hover
"""


class HoneyCombAnnotator:

    def __init__(
        self, image_path: Path, label_in_path: Path, label_out_path: Path
    ) -> None:
        self.image_path = image_path
        self.label_in_path = label_in_path
        self.label_out_path = label_out_path

        self.label_categories = [
            "unlabeled",
            "honey",
            "larvae",
            "pollen",
            "empty",
            "eggs",
            "nectar",
            "capped_brood",
            "capped_honey",
        ]

        self.color_map = dict(
            zip(
                self.label_categories,
                [
                    "blue",
                    "gold",
                    "cyan",
                    "pink",
                    "green",
                    "red",
                    "orange",
                    "lime",
                    "brown",
                ],
            )
        )

        self.viewer = napari.Viewer()
        self.points_layer, self.brush_layer = self._register_layers()

    def _register_layers(self) -> tuple[Points, Labels]:
        image, points, radii, labels = self._load_data()

        self.viewer.add_image(image, name="Honeycomb")

        features = pd.DataFrame(
            {"cell_type": pd.Categorical(labels, categories=self.label_categories)}
        )
        points_layer = self.viewer.add_points(
            points,
            features=features,
            face_color="cell_type",
            face_color_cycle=self.color_map,
            size=radii,
            name="Cells",
        )
        points_layer.face_color_mode = "cycle"

        brush_layer = self.viewer.add_labels(
            np.zeros(image.shape[:2], dtype=np.uint8), name="BrushMask"
        )
        self._set_custom_brush_limit(brush_layer)

        return points_layer, brush_layer

    def _set_custom_brush_limit(self, layer: Labels, max_size: int = 150):
        self.viewer.window._qt_viewer.controls.widgets[
            layer
        ].brushSizeSlider.setMaximum(max_size)
        layer.brush_size = 40

    def _load_data(self):
        image = imread(str(self.image_path))
        with open(self.label_in_path, "r") as f:
            cells = json.load(f)

        points, radii, labels = [], [], []
        for cell in cells:
            cx, cy, radius = cell["center_x"], cell["center_y"], cell["radius"]
            points.append([cy, cx])
            radii.append(radius)
            labels.append("unlabeled")

        return image, np.array(points), np.array(radii, dtype=float), labels

    def _export_annotated_cells(self, points_layer: Points, output_path: Path):
        points = points_layer.data
        sizes = points_layer.size  # array of radii
        labels = points_layer.features["cell_type"].tolist()

        exported = []
        for (y, x), r, label in zip(points, sizes, labels):
            exported.append(
                {
                    "center_x": int(x),
                    "center_y": int(y),
                    "radius": int(r),
                    "label": label,
                }
            )

        with open(output_path, "w") as f:
            json.dump(exported, f, indent=2)

    def _update_point_borders(self) -> None:
        if len(self.points_layer.data) == 0:
            return

        selected = self.points_layer.selected_data
        border_colors = [
            "white" if i in selected else "black"
            for i in range(len(self.points_layer.data))
        ]
        self.points_layer.border_color = border_colors

    def _on_brush_mask_change(self):
        if self.brush_layer.mode == "paint":
            mask_array = self.brush_layer.data
            selected = set()
            for i, (y, x) in enumerate(self.points_layer.data):
                if (
                    0 <= int(y) < mask_array.shape[0]
                    and 0 <= int(x) < mask_array.shape[1]
                ):
                    if mask_array[int(y), int(x)] == 1:
                        selected.add(i)

            # print(points_layer.mode)
            self.points_layer.selected_data = selected

            self._update_point_borders()

            # === Prevent infinite recursion ===
            self.brush_layer.events.paint.disconnect(self._on_brush_mask_change)
            self.brush_layer.data = np.zeros_like(self.brush_layer.data)
            self.brush_layer.events.paint.connect(self._on_brush_mask_change)

    def _create_label_menu(self):
        label_menu = ComboBox(label="cell_type", choices=self.label_categories)
        label_widget = Container(widgets=[label_menu])

        self.points_layer.events.feature_defaults.connect(
            lambda event: self._update_label_menu(label_menu)
        )

        label_menu.changed.connect(self._label_changed)
        return label_widget

    def _label_changed(self, selected_label: str) -> None:
        # Assign label to selected points
        selected = list(self.points_layer.selected_data)
        if not selected:
            # No selection → update feature_defaults for new points
            feature_defaults = self.points_layer.feature_defaults
            feature_defaults["cell_type"] = selected_label
            self.points_layer.feature_defaults = feature_defaults
        else:
            self.points_layer.features.loc[selected, "cell_type"] = selected_label
        self.points_layer.refresh_colors()

    def _update_label_menu(self, label_menu: ComboBox) -> None:
        new_label = str(self.points_layer.feature_defaults["cell_type"][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    def run(self) -> None:
        self.points_layer.selected_data.events.items_changed.connect(
            lambda e: self._update_point_borders()
        )
        self.brush_layer.events.paint.connect(self._on_brush_mask_change)

        label_widget = self._create_label_menu()
        self.viewer.window.add_dock_widget(label_widget, area="right")
        self.test()

        napari.run()

        self._export_annotated_cells(self.points_layer, self.label_out_path)

    def test(self):
        pass


image_path_2 = Path(
    r"C:\Users\sturmd\Desktop\Bachelorarbeit\ws=10_numimg=100_clahe=intermediate_dil=15_mdncomp=cupy_dur=233.png"
)

json_dir = Path(__file__).parents[1] / "cell_finder"
json_in_dir = json_dir / "cells.json"
json_out_dir = json_dir / "cells_annotated.json"


annotator = HoneyCombAnnotator(image_path_2, json_in_dir, json_out_dir)
annotator.run()


# def update_point_borders(points_layer: Points) -> None:
#     selected = points_layer.selected_data
#     border_colors = [
#         "white" if i in selected else "black" for i in range(len(points_layer.data))
#     ]
#     points_layer.border_color = border_colors


# def export_annotated_cells(points_layer, output_path: Path):
#     points = points_layer.data
#     sizes = points_layer.size  # array of radii
#     labels = points_layer.features["cell_type"].tolist()

#     exported = []
#     for (y, x), r, label in zip(points, sizes, labels):
#         exported.append(
#             {"center_x": int(x), "center_y": int(y), "radius": int(r), "label": label}
#         )

#     with open(output_path, "w") as f:
#         json.dump(exported, f, indent=2)


# def create_label_menu(points_layer, labels):
#     label_menu = ComboBox(label="cell_type", choices=labels)
#     label_widget = Container(widgets=[label_menu])

#     def update_label_menu(event):
#         new_label = str(points_layer.feature_defaults["cell_type"][0])
#         if new_label != label_menu.value:
#             label_menu.value = new_label

#     points_layer.events.feature_defaults.connect(update_label_menu)

#     def label_changed(selected_label):
#         # Assign label to selected points
#         selected = list(points_layer.selected_data)
#         if not selected:
#             # No selection → update feature_defaults for new points
#             feature_defaults = points_layer.feature_defaults
#             feature_defaults["cell_type"] = selected_label
#             points_layer.feature_defaults = feature_defaults
#         else:
#             points_layer.features.loc[selected, "cell_type"] = selected_label
#         points_layer.refresh_colors()

#     label_menu.changed.connect(label_changed)
#     return label_widget


# def start_napari_annotation(image_path: Path, json_in_path: Path, json_out_path: Path):
#     image = imread(str(image_path))
#     with open(json_in_path, "r") as f:
#         cells = json.load(f)

#     points = []
#     radii = []
#     labels = []

#     for cell in cells:
#         cx, cy, radius = cell["center_x"], cell["center_y"], cell["radius"]
#         points.append([cy, cx])
#         radii.append(radius)
#         labels.append("unlabeled")

#     points = np.array(points)
#     radii = np.array(radii, dtype=float)

#     label_categories = [
#         "unlabeled",
#         "honey",
#         "larvae",
#         "pollen",
#         "empty",
#         "capped_honey",
#     ]
#     features = pd.DataFrame(
#         {"cell_type": pd.Categorical(labels, categories=label_categories)}
#     )

#     color_map = {
#         "unlabeled": "blue",
#         "honey": "gold",
#         "larvae": "cyan",
#         "pollen": "orange",
#         "empty": "green",
#         "capped_honey": "red",
#     }

#     viewer = napari.Viewer()
#     viewer.add_image(image, name="Honeycomb")
#     points_layer = viewer.add_points(
#         points,
#         features=features,
#         face_color="cell_type",
#         face_color_cycle=color_map,
#         size=radii,
#         name="Cells",
#     )
#     points_layer.face_color_mode = "cycle"
#     points_layer.selected_data.events.items_changed.connect(
#         lambda e: update_point_borders(points_layer)
#     )

#     brush_mask = viewer.add_labels(
#         np.zeros(image.shape[:2], dtype=np.uint8), name="BrushMask"
#     )

#     # @viewer.bind_key("s")
#     # def select_points_in_brush(viewer=None):
#     #     mask_array = brush_mask.data
#     #     selected = set()
#     #     for i, (y, x) in enumerate(points_layer.data):
#     #         if 0 <= int(y) < mask_array.shape[0] and 0 <= int(x) < mask_array.shape[1]:
#     #             if mask_array[int(y), int(x)] == 1:
#     #                 selected.add(i)
#     #     points_layer.selected_data = selected
#     #     print(f"Selected {len(selected)} points under brush.")

#     def on_brush_mask_change(event):
#         if brush_mask.mode == "paint":
#             mask_array = brush_mask.data
#             selected = set()
#             for i, (y, x) in enumerate(points_layer.data):
#                 if (
#                     0 <= int(y) < mask_array.shape[0]
#                     and 0 <= int(x) < mask_array.shape[1]
#                 ):
#                     if mask_array[int(y), int(x)] == 1:
#                         selected.add(i)

#             # print(points_layer.mode)
#             points_layer.selected_data = selected

#             update_point_borders(points_layer)

#             # === Prevent infinite recursion ===
#             brush_mask.events.paint.disconnect(on_brush_mask_change)
#             brush_mask.data = np.zeros_like(brush_mask.data)
#             brush_mask.events.paint.connect(on_brush_mask_change)

#     brush_mask.events.paint.connect(on_brush_mask_change)

#     label_widget = create_label_menu(points_layer, label_categories)
#     viewer.window.add_dock_widget(label_widget, area="right")

#     napari.run()

#     export_annotated_cells(points_layer, json_out_path)

# start_napari_annotation(image_path_2, json_in_dir, json_out_dir)
