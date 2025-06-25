from annotation_viewer import AnnotationViewer
from data_loader import DataLoader
from annotation_model import HoneyCombAnnotationData
from pathlib import Path
import json
from utils import setup_logger


class AnnotationController:
    def __init__(self, data_dir: Path) -> None:
        self.logger = setup_logger(data_dir)
        self.label_map = self._load_label_map()
        self.loader = DataLoader(data_dir, self.logger)
        self.data = HoneyCombAnnotationData(list(self.label_map.keys()))
        self.viewer = AnnotationViewer(self.label_map, self.logger)

        self.image_idx = 0

        self.init_ui()

        self.viewer.run()

    def _load_label_map(self) -> dict[str, str]:
        curr_path = Path(__file__).parent  # TODO: currently hardcoded
        with open(curr_path / "label_map.json") as f:
            return json.load(f)

    def init_ui(self) -> None:
        assert self.image_idx == 0, "image index not at 0"
        self.data.update_data(*self.loader.load_data(image_idx=0))
        self.viewer.register_layers(self.data.full_data)
        self.viewer.connect_to_events()
        self.connect_to_ui_signals()

    def load_new_image(self, image_idx: int) -> None:
        self.data.update_data(*self.loader.load_data(image_idx))
        self.viewer.update_view(self.data.full_data)

    def connect_to_ui_signals(self) -> None:
        self.viewer.label_changed.connect(self.on_label_change)

    def on_label_change(self, new_label: str, selected_indices: list[int]) -> None:
        print("new label", new_label)
        print("selected indices", selected_indices)


if __name__ == "__main__":
    data_dir = Path(
        r"C:\Users\sturmd\Documents\Development\Privates\honeybee_cells_segmentation_inference\segmentation_restruct\annotator\data\input"
    )
    AnnotationController(data_dir)
