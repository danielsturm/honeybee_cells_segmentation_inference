import napari
from napari.layers import Points, Labels
from skimage.io import imread
import json
from pathlib import Path
import numpy as np
import pandas as pd
from magicgui.widgets import ComboBox, Container, PushButton, Label

from utils import restrict_brush_layer_tools

"""
TODO:   - ✅ point size according to actual cell size
        - ✅ multiple images at once. how to load. switch between and save
        - key shortcuts (switch between layers, activate tools, opacity of points, next/prev image)
        - ✅ stop the user from using other tools. how to disable them?
        - ✅ increase size by scrolling (maybe alt + scrolling)
        - 🛠️ position the drop down and the buttons correctly
        - typing
        - 💤 totally custom layer to avoid points and labels layer
        - add point by hitting alt + doubleclick (probably not. can just go into add mode of point layer)
        - initial point size should be higher (50)
        - ✅ image name should be displayed in ui
        - all cells not activated, currently some are and some not
        - ✅ a filed should indicate image x/of n. between arrows
        - initial label type should be unlabled
        - Logging!!!

TODO:   - Add key "c" to clear the BrushMask layer after selection
        - Add visual feedback (e.g., print selected labels/counts)
        - Add save button in GUI instead of saving only on close
        - Add tooltips or points_layer.text to show the label on hover

BUG:    - adding new point on small label size. point seems to disappear. when saving it crashes with
        line 147, in _export_annotated_cells
        "radius": int(r / 2),
              ^^^^^^^^^^
        ValueError: cannot convert float NaN to integer
"""


class HoneyCombAnnotator:
    point_max_rad = 100
    point_min_rad = 1

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        assert self.data_dir.exists(), "data dir does not exist"
        self.image_paths, self.label_paths = self._find_data(data_dir)

        self.image_idx = 0
        self.image_layer_name = ""

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

    def _find_data(self, data_dir: Path) -> tuple[list[Path], dict[str, Path]]:
        # TODO: Find only files with specific pattern
        image_paths = sorted(data_dir.glob("*.png"))
        label_paths = {p.stem: p for p in data_dir.glob("*.json")}
        return image_paths, label_paths

    def _load_data(self):
        image_path = self.image_paths[self.image_idx]
        image_name = image_path.stem
        image = imread(str(image_path))

        label_path = self.label_paths.get(image_path.stem)
        cells = []
        if label_path and label_path.exists():
            with open(label_path, "r") as f:
                cells = json.load(f)

        points, radii, labels = [], [], []
        for cell in cells:
            cx, cy, radius, label = (
                cell["center_x"],
                cell["center_y"],
                cell["radius"],
                cell["label"],
            )
            points.append([cy, cx])
            radii.append(radius * 2)
            labels.append(label)

        return image, image_name, np.array(points), np.array(radii, dtype=float), labels

    def _register_layers(self) -> tuple[Points, Labels]:
        image, image_name, points, radii, labels = self._load_data()
        self.image_layer_name = image_name
        self.viewer.add_image(image, name=image_name)

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
        layer.brush_size = 70
        layer.mode = "paint"

    def _update_layers(self) -> None:
        image, image_name, points, radii, labels = self._load_data()
        self.viewer.layers[self.image_layer_name].data = image
        self.viewer.layers[self.image_layer_name].name = image_name
        self.image_layer_name = image_name

        features = pd.DataFrame(
            {
                "cell_type": pd.Categorical(labels, categories=self.label_categories),
                "radius": radii,
            }
        )
        self.points_layer.data = points

        self.points_layer.features = features
        self.points_layer.size = radii
        self.points_layer.selected_data = set()
        self._update_point_borders()
        self.brush_layer.data = np.zeros(image.shape[:2], dtype=np.uint8)

    def _export_annotated_cells(self, points_layer: Points) -> None:
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

        curr_img_name = self.image_paths[self.image_idx].stem
        output_path = self.data_dir / f"{curr_img_name}.json"

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

    def _on_points_data_change(self, event):
        max_idx = len(self.points_layer.data) - 1
        self.points_layer.selected_data = {
            i for i in self.points_layer.selected_data if i <= max_idx
        }

    def _create_navigation_buttons(self) -> Container[PushButton | Label]:
        prev_button = PushButton(label="← Previous")
        self.nav_label = Label(label=self._nav_label_text())
        next_button = PushButton(label="Next →")

        prev_button.clicked.connect(self._previous_image)
        next_button.clicked.connect(self._next_image)

        button_container = Container(
            widgets=[prev_button, self.nav_label, next_button], layout="horizontal"
        )
        return button_container

    def _nav_label_text(self) -> str:
        return f"{self.image_idx + 1}/{len(self.image_paths)} images"

    def _next_image(self) -> None:
        self._export_annotated_cells(self.points_layer)

        if self.image_idx + 1 < len(self.image_paths):
            self.image_idx += 1
            self._update_layers()
            self.nav_label.label = self._nav_label_text()

    def _previous_image(self) -> None:
        self._export_annotated_cells(self.points_layer)

        if self.image_idx - 1 >= 0:
            self.image_idx -= 1
            self._update_layers()
            self.nav_label.label = self._nav_label_text()

    def _connect_brush_auto_mode(self, brush_layer: Labels):
        def on_active_layer_change(event):
            if self.viewer.layers.selection.active == brush_layer:
                print("[DEBUG] Re-activating paint mode")
                brush_layer.mode = "paint"
                brush_layer.brush_size = 70

        self.viewer.layers.selection.events.active.connect(on_active_layer_change)

    def run(self) -> None:
        self.points_layer.selected_data.events.items_changed.connect(
            lambda e: self._update_point_borders()
        )
        self.points_layer.mouse_wheel_callbacks.append(
            self._on_scroll_change_point_size
        )
        self.brush_layer.events.paint.connect(self._on_brush_mask_change)

        label_widget = self._create_label_menu()
        self.viewer.window.add_dock_widget(label_widget, area="left")

        nav_buttons = self._create_navigation_buttons()
        self.viewer.window.add_dock_widget(nav_buttons, area="left")

        """maybe this need to be used when the error
        IndexError: index 3478 is out of bounds for axis 0 with size 3478
        appears again"""
        # self.points_layer.events.data.connect(self._on_points_data_change)

        restrict_brush_layer_tools(self.viewer, self.brush_layer, self.points_layer)
        self._connect_brush_auto_mode(self.brush_layer)

        napari.run()

        self._export_annotated_cells(self.points_layer)


data_dir = Path(
    r"C:\Users\sturmd\Documents\Development\Privates\honeybee_cells_segmentation_inference\segmentation_restruct\annotator\data\input"
)


annotator = HoneyCombAnnotator(data_dir=data_dir)
annotator.run()
