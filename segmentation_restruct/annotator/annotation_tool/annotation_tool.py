import napari
from napari.layers import Points, Labels
from skimage.io import imread
import json
from pathlib import Path
import numpy as np
import pandas as pd
from magicgui.widgets import ComboBox, Container


"""
TODO:   - ✅ point size according to actual cell size
        - multiple images at once. how to load. switch between and save
        - key shortcuts (switch between layers, activate tools, opacity of points)
        - stop the user from using other tools. how to disable them?
        - ✅ increase size by scrolling (maybe alt + scrolling)
        - typing
        - totally custom layer to avoid points and labels layer
        - add point by hitting alt + doubleclick (probably not. can just go into add mode of point layer)

TODO:   🧽 Add key "c" to clear the BrushMask layer after selection

        👀 Add visual feedback (e.g., print selected labels/counts)

        📝 Add save button in GUI instead of saving only on close

        🖱️ Add tooltips or points_layer.text to show the label on hover
"""


class HoneyCombAnnotator:
    point_max_rad = 100
    point_min_rad = 1

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
            {
                "cell_type": pd.Categorical(labels, categories=self.label_categories),
                "radius": radii,
            }
        )
        points_layer = self.viewer.add_points(
            points,
            features=features,
            face_color="cell_type",
            face_color_cycle=self.color_map,
            size=features["radius"].values,
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
            radii.append(radius * 2)
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
                    "radius": int(r / 2),
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

    def _on_scroll_change_point_size(self, _, event):
        # print(type(event))
        # print(event.type)
        # print(event.delta)
        # print(event.modifiers)
        # print(event.is_dragging)
        # print(event.button)
        # print(event.position)

        if "Alt" not in event.modifiers:
            return

        delta = 1 if event.delta[0] > 0 else -1
        selected = list(self.points_layer.selected_data)
        if not selected:
            return

        for i in selected:
            current_radius = self.points_layer.features.at[i, "radius"]
            new_radius = current_radius + delta * 2
            new_radius = min(self.point_max_rad, max(self.point_min_rad, new_radius))
            self.points_layer.features.at[i, "radius"] = new_radius

        self.points_layer.size = self.points_layer.features["radius"].values

        self.points_layer.selected_data = set(selected)

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
        self.points_layer.mouse_wheel_callbacks.append(
            self._on_scroll_change_point_size
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
