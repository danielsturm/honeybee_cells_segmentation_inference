import napari
from napari.layers import Points, Labels, Image
from napari.layers._data_protocols import LayerDataProtocol
import logging
import numpy as np
import pandas as pd
from magicgui.widgets import ComboBox, Container
from annotation_model import HoneyCombAnnotationData
from utils import restrict_brush_layer_tools
from annotation_model import AnnotationDTO

from qtpy.QtCore import QObject, Signal  # type: ignore


class AnnotationViewer(QObject):
    point_max_rad = 100
    point_min_rad = 1

    label_changed = Signal(str, list)

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
        self.points_layer = Points | None

    def register_layers(self, data: AnnotationDTO) -> None:
        self.viewer.add_image(data.image, name=data.image_name)
        points_layer = self.viewer.add_points(
            data.points,
            features=self._construct_features(data.labels, data.radii),
            face_color="cell_type",
            face_color_cycle=self.label_map,
            size=data.radii,  # type: ignore
            name="Cells",
        )
        points_layer.face_color_mode = "cycle"
        self.points_layer = points_layer
        self._update_point_borders()
        print(data.ids)

    def update_view(self, data: AnnotationDTO) -> None:
        # image layer
        self._remove_old_image_layer()
        self.viewer.add_image(data.image, name=data.image_name)

        # points layer
        self.points_layer.data = data.points
        self.points_layer.features = self._construct_features(data.labels, data.radii)
        self.points_layer.size = data.radii
        self._update_point_borders()

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

    def _construct_features(self, labels, radii) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "cell_type": pd.Categorical(labels, categories=self.label_categories),
                "radius": radii,
            }
        )

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
            self.label_changed.emit(selected_label, selected_points)
            self._update_points_label(selected_label, selected_points)

    def _update_points_label(self, selected_label: str, selected_points: list[int]):
        self.points_layer.features.loc[selected_points, "cell_type"] = selected_label
        self.points_layer.refresh_colors()

    def _update_point_borders(self) -> None:
        if len(self.points_layer.data) == 0:
            return

        selected = self.points_layer.selected_data
        border_colors = [
            "white" if i in selected else "black"
            for i in range(len(self.points_layer.data))
        ]
        self.points_layer.border_color = border_colors

    def test_foo(self, event):
        print(event.source)
        selected = self.points_layer.selected_data.events.items_changed
        print("selected", selected)

    def test_bar(self, event):
        action = event.action
        indices = event.data_indices
        print(f"Action: {action}, Indices: {indices}")

    def connect_to_events(self) -> None:
        self.points_layer.events.feature_defaults.connect(
            lambda event: self._update_label_menu(self.label_menu)
        )
        self.points_layer.selected_data.events.items_changed.connect(
            lambda e: self._update_point_borders()
        )
        # self.points_layer.selected_data.events.items_changed.connect(self.test_foo)
        self.points_layer.events.data.connect(self.test_bar)
        self.label_menu.changed.connect(self._label_changed)

    def run(self) -> None:
        napari.run()
