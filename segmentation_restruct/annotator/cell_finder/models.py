from dataclasses import dataclass, field
import numpy as np
from typing import Literal


@dataclass
class CellNode:
    id: str
    position: np.ndarray  # shape (2,)
    neighbors: dict = field(default_factory=lambda: {i: None for i in range(6)})


@dataclass
class HexGraphConfig:
    neighbour_pos_tolerance: float = 17.0
    bidrectional_assignment: bool = False
    image_size: tuple[int, int] | None = None
    max_iterations: int = 3  # number of max iterations until the graph building stops
    curve_aware_candidate_pred: bool = True
    cluster_vote_threshold: int = 2
    cluster_pred_eps: float = 16.0  # eps param for dbscan to cluster temp predictions
    pred_merge_dist: float = 25.0
    cluster_conflict_pred_eps: float = (
        22.0  # eps param for dbscan to cluster final predictions when resolving conflicts
    )
    min_dist_nodes: float = 40.0  # min distance that new predictions need to have to existing points
    prefer_method: Literal["curve", "lattice_vector"] | None = "curve"
    debug: bool = False
