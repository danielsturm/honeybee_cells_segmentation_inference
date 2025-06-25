from dataclasses import dataclass
import numpy as np


@dataclass
class Cell:
    id: str
    x: float
    y: float
    radius: float
    label: str


@dataclass
class AnnotationDTO:
    image: np.ndarray
    image_name: str
    ids: list[str]
    points: np.ndarray
    radii: np.ndarray
    labels: list[str]


class HoneyCombAnnotationData:
    def __init__(self, label_categories: list[str]) -> None:
        self._image: np.ndarray | None = None
        self._image_name: str | None = None
        self.label_categories = label_categories
        self.cells: list[Cell] = []

    def update_data(
        self,
        image: np.ndarray,
        image_name: str,
        ids: list[str],
        points: np.ndarray,
        radii: np.ndarray,
        labels: list[str],
    ) -> None:
        self._image = image
        self._image_name = image_name
        self.cells = []
        for id, (y, x), r, label in zip(ids, points, radii, labels):
            self.cells.append(
                Cell(id=id, x=float(x), y=float(y), radius=float(r / 2), label=label)
            )

    @property
    def cell_data(self) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
        ids = [cell.id for cell in self.cells]
        points = np.array([[cell.y, cell.x] for cell in self.cells])
        radii = np.array([cell.radius * 2 for cell in self.cells])
        labels = [cell.label for cell in self.cells]
        assert (
            len(ids) == len(points) == len(radii) == len(labels)
        ), "cell data is inconsistent"
        return ids, points, radii, labels

    @property
    def full_data(self) -> AnnotationDTO:
        return AnnotationDTO(self.image, self.image_name, *self.cell_data)

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            raise RuntimeError("Image data not yet loaded")
        return self._image

    @property
    def image_name(self) -> str:
        if self._image_name is None:
            raise RuntimeError("Image name not yet set")
        return self._image_name
