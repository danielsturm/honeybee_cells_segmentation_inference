from annotation_viewer import AnnotationViewer
from data_loader import DataLoader
from annotation_model import HoneyCombAnnotationData
from pathlib import Path
import json
import argparse
from utils import setup_logger
from annotation_model import Cell


class AnnotationController:
    def __init__(self, data_dir: Path) -> None:
        self.logger = setup_logger(data_dir)
        self.label_map = self._load_label_map()
        self.loader = DataLoader(data_dir, self.logger)
        self.data = HoneyCombAnnotationData(list(self.label_map.keys()), self.logger)
        self.viewer = AnnotationViewer(self.label_map, self.logger)

        self.image_idx = 0

    def _load_label_map(self) -> dict[str, str]:
        curr_path = Path(__file__).parent  # TODO: currently hardcoded
        with open(curr_path / "label_map.json") as f:
            return json.load(f)

    def init_ui(self) -> None:
        assert self.image_idx == 0, "image index not at 0"
        self.data.update_data(*self.loader.load_data(image_idx=0))
        self.viewer.register_layers(self.data.full_data, self._nav_label_text())
        self.viewer.connect_to_events()
        self.connect_to_ui_signals()

    def load_new_image(self, image_idx: int) -> None:
        self.data.update_data(*self.loader.load_data(image_idx))
        self.viewer.update_view(self.data.full_data, self._nav_label_text())

    def _export_data(self) -> None:
        # TODO: change to AnnotationDTO everywhere and pass only DTO
        full_data = self.data.full_data
        self.loader.export_annotated_cells(
            self.image_idx,
            full_data.ids,
            full_data.points,
            full_data.point_diameters,
            full_data.labels,
        )

    def connect_to_ui_signals(self) -> None:
        self.viewer.label_changed.connect(self.on_label_change)
        self.viewer.point_data_changed.connect(self.on_point_data_change)
        self.viewer.image_change.connect(self.on_image_change_clicked)

    def on_label_change(self, new_label: str, updated_cells: list[Cell]) -> None:
        self.data.edit_cells(updated_cells)

    def on_point_data_change(self, action: str, changed_points: list[Cell]) -> None:
        match action:
            case "added":
                self.data.add_cells(changed_points)
            case "changed":
                self.data.edit_cells(changed_points)
            case "removed":
                self.data.remove_cells(changed_points)
            case _:
                self.logger.error("Unsupported switch case in controller (point data changed)")

    def on_image_change_clicked(self, direction: str) -> None:
        self._export_data()
        match direction:
            case "next":
                if self.image_idx + 1 < self.loader.data_count:
                    self.image_idx += 1
                    self.load_new_image(self.image_idx)
            case "prev":
                if self.image_idx - 1 >= 0:
                    self.image_idx -= 1
                    self.load_new_image(self.image_idx)
            case _:
                self.logger.error("Unsupported switch case in controller (image switch)")

    def _nav_label_text(self) -> str:
        return f"{self.image_idx + 1}/{self.loader.data_count} images"

    def run(self) -> None:
        self.init_ui()
        self.viewer.run()
        self._export_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Honeycomb Annotation Tool")
    parser.add_argument("data_dir", type=Path, help="Path to the input data directory")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        parser.error(f"Provided data_dir {data_dir} does not exist.")
    controller = AnnotationController(data_dir)
    controller.run()
