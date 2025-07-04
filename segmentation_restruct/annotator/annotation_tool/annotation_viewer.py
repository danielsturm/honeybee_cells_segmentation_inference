import napari
from napari.layers import Image
from napari.layers._data_protocols import LayerDataProtocol
import logging
import numpy as np
import pandas as pd
import uuid
from magicgui.widgets import ComboBox, Container, PushButton, Label
from utils import restrict_brush_layer_tools
from annotation_model import AnnotationDTO, Cell
from typing import cast

from qtpy.QtCore import QObject, Signal  # type: ignore


class AnnotationViewer(QObject):
    point_max_diameter = 100
    point_min_diameter = 1

    label_changed = Signal(str, list)
    point_data_changed = Signal(str, list)
    image_change = Signal(str)

    def __init__(
        self,
        label_map: dict[str, str],
        logger: logging.Logger,
    ) -> None:
        super().__init__()
        self.label_map = label_map
        self.logger = logger

        self.viewer = napari.Viewer()
        self.label_menu = self._create_label_menu()
        self.nav_buttons = self._create_navigation_buttons()

    def register_layers(self, data: AnnotationDTO, button_label: str) -> None:
        # image layer
        self.viewer.add_image(data.image, name=data.image_name)

        # points layer
        points_layer = self.viewer.add_points(
            data.points,
            features=self._construct_features(data.labels, data.point_diameters, data.ids),
            face_color="cell_type",
            face_color_cycle=self.label_map,
            size=data.point_diameters,  # type: ignore
            name="Cells",
        )
        points_layer.face_color_mode = "cycle"
        self.points_layer = points_layer
        self._set_init_point_size()
        self._update_feature_defaults_with_uuid()
        self._update_feature_defaults_with_point_size()
        self._update_point_borders()
        self.update_button_label(button_label)

        # brush layer
        self.brush_layer = self.viewer.add_labels(np.zeros(data.image.shape[:2], dtype=np.uint8), name="BrushMask")
        self._set_custom_brush_limit()
        restrict_brush_layer_tools(self.viewer, self.brush_layer, self.points_layer)

    def update_view(self, data: AnnotationDTO) -> None:
        # image layer
        self._remove_old_image_layer()
        self.viewer.add_image(data.image, name=data.image_name)

        # points layer
        self.points_layer.data = data.points
        self.points_layer.features = self._construct_features(data.labels, data.point_diameters, data.ids)
        self.points_layer.size = data.point_diameters
        self._update_point_borders()

        # brush layer
        new_data = np.zeros(data.image.shape[:2], dtype=np.uint8)
        self.brush_layer.data = cast(LayerDataProtocol, new_data)

    def _remove_old_image_layer(self) -> None:
        for layer in list(self.viewer.layers):
            if isinstance(layer, Image):
                self.viewer.layers.remove(layer)

    def print_layers(self):
        for layer in list(self.viewer.layers):
            print("layer type", type(layer))
            print("layer name", layer.name)

    @property
    def label_categories(self) -> list[str]:
        return list(self.label_map.keys())

    def _construct_features(self, labels: list[str], point_diameters: np.ndarray, ids: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cell_type": pd.Categorical(labels, categories=self.label_categories),
                "diameter": point_diameters,
                "id": ids,
            }
        )

    def _set_init_point_size(self, init_size: int = 46) -> None:
        self.points_layer.current_size = init_size

    def _set_custom_brush_limit(self, max_size: int = 150) -> None:
        self.viewer.window._qt_viewer.controls.widgets[self.brush_layer].brushSizeSlider.setMaximum(max_size)
        self.brush_layer.brush_size = 70
        self.brush_layer.mode = "paint"

    def _on_brush_mask_change(self) -> None:
        if self.brush_layer.mode == "paint":
            mask_array = self.brush_layer.data
            selected = set()
            for i, (y, x) in enumerate(self.points_layer.data):
                if 0 <= int(y) < mask_array.shape[0] and 0 <= int(x) < mask_array.shape[1]:
                    if mask_array[int(y), int(x)] == 1:  # type: ignore
                        selected.add(i)

            self.points_layer.selected_data = selected  # type: ignore

            self._update_point_borders()

            # === Prevent infinite recursion ===
            self.brush_layer.events.paint.disconnect(self._on_brush_mask_change)
            new_data = np.zeros_like(self.brush_layer.data)
            self.brush_layer.data = cast(LayerDataProtocol, new_data)
            # self.brush_layer.data = np.zeros_like(self.brush_layer.data)
            self.brush_layer.events.paint.connect(self._on_brush_mask_change)

    def _on_active_layer_change(self, event):
        if self.viewer.layers.selection.active == self.brush_layer:
            self.brush_layer.mode = "paint"
            self.brush_layer.brush_size = 70

    def _create_label_menu(self) -> ComboBox:
        label_menu = ComboBox(label="cell_type", choices=self.label_categories)
        label_widget = Container(widgets=[label_menu])
        self.viewer.window.add_dock_widget(label_widget, area="left")
        return label_menu

    def _update_label_menu(self, label_menu: ComboBox) -> None:
        new_label = str(self.points_layer.feature_defaults["cell_type"][0])
        if new_label != label_menu.value:
            """this is only to stop the ui from firing signal when
            only point is selected and label not changed. otherwise
            _label_changed() will be executed and spam signal"""
            with label_menu.changed.blocked():
                label_menu.value = new_label

    def _label_changed(self, selected_label: str) -> None:
        selected_points = list(self.points_layer.selected_data)
        if not selected_points:
            feature_defaults = self.points_layer.feature_defaults
            feature_defaults["cell_type"] = selected_label
            self.points_layer.feature_defaults = feature_defaults
        else:
            self._update_points_label(selected_label, selected_points)

            updated_cells = self._build_cell_payload_from_indices(selected_points)
            for cell in updated_cells:
                cell.label = selected_label

            self.label_changed.emit(selected_label, updated_cells)

    def _update_points_label(self, selected_label: str, selected_points: list[int]):
        self.points_layer.features.loc[selected_points, "cell_type"] = selected_label
        self.points_layer.refresh_colors()

    def _update_point_borders(self) -> None:
        if len(self.points_layer.data) == 0:
            return

        selected = self.points_layer.selected_data
        border_colors = ["white" if i in selected else "black" for i in range(len(self.points_layer.data))]
        self.points_layer.border_color = border_colors

    def _points_changed(self, event) -> None:
        if event.action not in {"added", "adding", "changed", "removing"}:
            return
        match event.action:
            case "adding":
                """Napari seems to copy/block the default features in the
                'added' stage. So the uuid change is ineffective there"""
                self._update_feature_defaults_with_uuid()
                """In case no points are selected the points size changed
                is not triggered"""
                self._update_feature_defaults_with_point_size()
            case "added":
                # TODO: replace with _build_cell_payload_from_indices, pass [-1]
                new_point = self.points_layer.data[-1]
                feat = self.points_layer.features.iloc[-1]
                result = []
                result.append(
                    Cell(
                        feat["id"],
                        new_point[1],
                        new_point[0],
                        feat["diameter"],
                        feat["cell_type"],
                    )
                )
                self.point_data_changed.emit("added", result)
            case "changed":
                # TODO: replace with _build_cell_payload_from_indices
                result = []
                moved_indices = list(event.data_indices)
                moved_points = self.points_layer.data[moved_indices]
                moved_feats = self.points_layer.features.iloc[moved_indices]
                for i, point in enumerate(moved_points):
                    result.append(
                        Cell(
                            moved_feats["id"].iloc[i],
                            point[1],
                            point[0],
                            moved_feats["diameter"].iloc[i],
                            moved_feats["cell_type"].iloc[i],
                        )
                    )
                self.point_data_changed.emit("changed", result)
            case "removing":
                # TODO: replace with _build_cell_payload_from_indices,
                result = []
                to_be_removed_indices = list(event.data_indices)
                to_be_removed_feats = self.points_layer.features.iloc[to_be_removed_indices]
                for idx, row in to_be_removed_feats.iterrows():
                    result.append(Cell(row["id"], -1.0, -1.0, -1.0, "None"))
                self.point_data_changed.emit("removed", result)
            case _:
                print("unsupported case")
                return

    def _update_feature_defaults_with_uuid(self) -> None:
        defaults = self.points_layer.feature_defaults
        defaults["id"] = str(uuid.uuid4())
        self.points_layer.feature_defaults = defaults

    def _update_feature_defaults_with_point_size(self) -> None:
        defaults = self.points_layer.feature_defaults
        defaults["diameter"] = self.points_layer.current_size
        self.points_layer.feature_defaults = defaults

    def _build_cell_payload_from_indices(self, indices: list[int]) -> list[Cell]:
        result = []
        feats = self.points_layer.features.iloc[indices]
        data = self.points_layer.data[indices]
        for i, idx in enumerate(indices):
            result.append(
                Cell(
                    feats["id"].iloc[i],
                    data[i][1],
                    data[i][0],
                    feats["diameter"].iloc[i],
                    feats["cell_type"].iloc[i],
                )
            )
        return result

    def _on_scroll_change_point_size(self, _, event) -> None:
        if "Alt" not in event.modifiers:
            return

        delta = 1 if event.delta[0] > 0 else -1
        selected = list(self.points_layer.selected_data)
        if not selected:
            return

        for idx in selected:
            current_point_diameter = self.points_layer.features.at[idx, "diameter"]
            new_point_diameter = current_point_diameter + delta * 2
            new_point_diameter = min(
                self.point_max_diameter,
                max(self.point_min_diameter, new_point_diameter),
            )
            self.points_layer.features.at[idx, "diameter"] = new_point_diameter

        self.points_layer.size = self.points_layer.features["diameter"].values

        self.points_layer.selected_data = set(selected)

        self._update_feature_defaults_with_point_size()
        result = self._build_cell_payload_from_indices(selected)
        self.point_data_changed.emit("changed", result)

    def _on_point_size_changed(self, event):
        selected = list(self.points_layer.selected_data)
        if not selected:
            return

        current_sizes = self.points_layer.size
        is_scalar = np.isscalar(current_sizes)

        for idx in selected:
            new_point_diameter = current_sizes if is_scalar else current_sizes[idx]
            self.points_layer.features.at[idx, "diameter"] = new_point_diameter

        self._update_feature_defaults_with_point_size()
        result = self._build_cell_payload_from_indices(selected)
        self.point_data_changed.emit("changed", result)

    def _create_navigation_buttons(self) -> Container[PushButton | Label]:
        self.prev_button = PushButton(label="← Previous")
        self.nav_label = Label(label="")
        self.next_button = PushButton(label="Next →")
        button_container = Container(
            widgets=[self.prev_button, self.nav_label, self.next_button],
            layout="horizontal",
        )
        self.viewer.window.add_dock_widget(button_container, area="left")
        return button_container

    def update_button_label(self, label: str) -> None:
        self.nav_label.label = label

    def connect_to_events(self) -> None:
        self.points_layer.events.feature_defaults.connect(lambda event: self._update_label_menu(self.label_menu))
        self.points_layer.selected_data.events.items_changed.connect(lambda e: self._update_point_borders())
        self.points_layer.events.data.connect(self._points_changed)

        self.label_menu.changed.connect(self._label_changed)

        self.points_layer.mouse_wheel_callbacks.append(self._on_scroll_change_point_size)
        self.points_layer.events.size.connect(self._on_point_size_changed)

        self.brush_layer.events.paint.connect(self._on_brush_mask_change)
        self.viewer.layers.selection.events.active.connect(self._on_active_layer_change)

        self.prev_button.clicked.connect(lambda: self.image_change.emit("prev"))
        self.next_button.clicked.connect(lambda: self.image_change.emit("next"))

    def run(self) -> None:
        napari.run()


"""
TODO: replace all complicated signatures to pass data with DTO objects
TODO: DTO should have serialization functions
TODO: Is the point size always in range?
TODO: Changing the label does not change the default value. Next cell is the old again
TODO: Point color should be the border and internal opaque. Then selected is just white border
TODO: Tooltips with information about label
"""
