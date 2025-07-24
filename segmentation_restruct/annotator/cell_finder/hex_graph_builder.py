from dataclasses import dataclass, field
import numpy as np
import uuid
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN, KMeans
from collections import defaultdict
from typing import Literal

from segmentation_restruct.annotator.cell_finder.utils import visualize_hex_lattice_graph


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
    max_iterations: int = 3
    curve_aware_candidate_pred: bool = True
    cluster_vote_threshold: int = 2
    dbscan_eps: float = 16.0
    prefer_method: Literal["curve", "lattice_vector"] = "curve"


class HexLatticeGraph:
    def __init__(self, seed_points, lattice_vectors, tolerance=15.0, image_size=None, bidirectional=True):
        self.seed_points = seed_points
        self.nodes = {}
        self.pos_index = []  # list of np.ndarray positions
        self.id_index = []  # list of UUIDs (same index as pos_index)
        self.vecs = [np.array(v) for v in lattice_vectors]  # v1, v2, v3
        self.tolerance = tolerance
        self.image_size = image_size  # (width, height), optional
        self.bidirectional = bidirectional

    @property
    def cell_positions(self) -> list[tuple[int, int, int, float]]:
        return [(*cell, 24, 1.0) for cell in self.pos_index]

    def grow_graph_iteratively(self, n_iterations=3, verbose=False):
        for pt in self.seed_points:
            self.add_node(pt)

        self.build_edges()

        # visualize_hex_lattice_graph(self)

        for i in range(n_iterations):
            if verbose:
                print(f"\nIteration {i + 1}")

            # --- 1. Predict missing cells ---
            raw_candidates = self.collect_missing_neighbor_candidates()

            # Early exit if nothing missing
            if not raw_candidates:
                if verbose:
                    print("No candidate predictions found. Stopping.")
                break

            # --- 2. Cluster and validate ---
            validated = self.cluster_and_filter_candidates(raw_candidates)

            # visualize_hex_lattice_graph(self, candidates=raw_candidates, validated=validated)

            if verbose:
                print(f"Validated {len(validated)} new cells")

            if not validated:
                if verbose:
                    print("No validated predictions passed filtering. Stopping.")
                break

            # --- 3. Add new predicted nodes ---
            new_nodes = 0
            for pos, support in validated:
                self.add_node(position=pos)
                new_nodes += 1

            if verbose:
                print(f"Added {new_nodes} new nodes")

            # --- 4. Rebuild graph edges ---
            self.build_edges()

            # visualize_hex_lattice_graph(self)

            # If nothing was added, break early
            if new_nodes == 0:
                if verbose:
                    print("No new nodes added this round. Stopping.")
                break

    def add_node(self, position):
        rounded_pos = np.round(position).astype(int)
        node_id = str(uuid.uuid4())
        # node = CellNode(id=node_id, position=np.array(position))
        node = CellNode(id=node_id, position=rounded_pos)
        self.nodes[node_id] = node
        self.pos_index.append(node.position)
        self.id_index.append(node_id)
        return node

    def _in_bounds(self, pos):
        if self.image_size is None:
            return True
        x, y = pos
        width, height = self.image_size
        return 0 <= x < width and 0 <= y < height

    def _vector_direction_index(self, direction, angle_tol_deg=20):
        """
        Match a direction vector to one of the 6 canonical lattice directions.
        Returns index 0–5 or None.
        """
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None
        direction_unit = direction / norm

        lattice_dirs = self.vecs + [-v for v in self.vecs]  # directions 0–5
        lattice_dirs = [v / np.linalg.norm(v) for v in lattice_dirs]

        cos_angles = [np.dot(direction_unit, d) for d in lattice_dirs]
        best_idx = np.argmax(cos_angles)
        angle_diff = np.degrees(np.arccos(np.clip(cos_angles[best_idx], -1.0, 1.0)))

        if angle_diff <= angle_tol_deg:
            return best_idx
        return None

    def build_edges(self):

        positions = np.array(self.pos_index)
        search_radius = max(np.linalg.norm(v) for v in self.vecs) + self.tolerance
        nbrs = NearestNeighbors(radius=search_radius)
        nbrs.fit(positions)

        for i, node_id in enumerate(self.id_index):
            origin = self.nodes[node_id]
            origin_pos = origin.position

            for dir_index, vec in enumerate(self.vecs + [-v for v in self.vecs]):
                predicted_pos = origin_pos + vec

                # if np.array_equal(origin_pos, np.array([1540, 626])):
                #     print("predicted pos", predicted_pos, "direction idx:", dir_index)

                if not self._in_bounds(predicted_pos):
                    origin.neighbors[dir_index] = "OUT_OF_BOUNDS"
                    continue

                # Find actual neighbor close to predicted position
                distances = np.linalg.norm(positions - predicted_pos, axis=1)
                close_indices = np.where(distances <= self.tolerance)[0]

                # if np.array_equal(origin_pos, np.array([1540, 626])):
                #     print(f"\n→ Looking for match for direction {dir_index}")
                #     print(f"Predicted pos: {predicted_pos}")
                #     for idx, pos in enumerate(positions):
                #         dist = np.linalg.norm(pos - predicted_pos)
                #         if dist <= self.tolerance + 5:  # slightly loosen threshold to check
                #             print(f"  ✓ Candidate at {pos}, dist = {dist:.2f}, ID = {self.id_index[idx]}")

                if len(close_indices) > 0:
                    # if np.array_equal(origin_pos, np.array([1540, 626])):
                    #     print(f"Predicted position: {predicted_pos}")
                    #     print("Close candidates (within tolerance):")
                    #     for idx in close_indices:
                    #         coord = positions[idx]
                    #         dist = distances[idx]
                    #         print(f"  → ID: {self.id_index[idx]}, Position: {coord}, Distance: {dist:.2f}")

                    neighbor_id = self.id_index[close_indices[0]]
                    origin.neighbors[dir_index] = neighbor_id

                    # Optional: bidirectional assignment
                    if self.bidirectional:
                        reverse_dir = (dir_index + 3) % 6
                        self.nodes[neighbor_id].neighbors[reverse_dir] = node_id
                else:
                    origin.neighbors[dir_index] = None

    def collect_missing_neighbor_candidates(self, curve_aware=True):

        candidates = []
        directions = self.vecs + [-v for v in self.vecs]

        for node_id, node in self.nodes.items():
            pos = node.position

            for dir_idx in range(6):
                method = "curve"
                neighbor_id = node.neighbors[dir_idx]

                if neighbor_id is not None:
                    continue  # already connected or marked out of bounds

                # Predict position of missing neighbor in this direction
                pred_pos = None

                if curve_aware:
                    # Step back along the opposite direction
                    back_dir = (dir_idx + 3) % 6
                    prev1_id = node.neighbors[back_dir]

                    if prev1_id is not None and prev1_id != "OUT_OF_BOUNDS":
                        prev1 = self.nodes[prev1_id]
                        prev2_id = prev1.neighbors[back_dir]

                        if prev2_id is not None and prev2_id != "OUT_OF_BOUNDS":
                            prev2 = self.nodes[prev2_id]

                            # Collect positions: [prev2, prev1, current]
                            pts = np.array([prev2.position, prev1.position, pos])

                            # Fit linear regression to x and y separately
                            X = np.arange(3).reshape(-1, 1)
                            model_x = LinearRegression().fit(X, pts[:, 0])
                            model_y = LinearRegression().fit(X, pts[:, 1])

                            # Predict next point at step = 3
                            next_x = model_x.predict([[3]])[0]
                            next_y = model_y.predict([[3]])[0]
                            pred_pos = np.array([next_x, next_y])

                if pred_pos is None:
                    # Fallback to lattice vector-based prediction
                    vec = directions[dir_idx]
                    pred_pos = pos + vec
                    method = "lattice_vector"

                # Final bounds check
                if not self._in_bounds(pred_pos):
                    continue

                candidates.append((pred_pos, node_id, dir_idx, method))

        return candidates

    def cluster_and_filter_candidates(self, candidates, vote_threshold=2, merge_radius=16.0, prefer_method="curve"):

        if not candidates:
            return []

        # Extract just the position vectors for clustering
        pos_array = np.array([pos for (pos, *_) in candidates])

        # Cluster nearby predictions using DBSCAN
        clustering = DBSCAN(eps=merge_radius, min_samples=1).fit(pos_array)
        labels = clustering.labels_

        clustered = defaultdict(list)

        # Group candidates by cluster label
        for idx, label in enumerate(labels):
            clustered[label].append(candidates[idx])

        filtered = []

        for cluster in clustered.values():
            if len(cluster) < vote_threshold:
                continue  # not enough support

            # Split by method if needed
            if prefer_method:
                method_groups = defaultdict(list)  # why is this needed
                for item in cluster:
                    method = item[3]
                    method_groups[method].append(item)

                # Use preferred method group if enough votes
                if prefer_method in method_groups and len(method_groups[prefer_method]) >= vote_threshold:
                    used_cluster = method_groups[prefer_method]
                else:
                    used_cluster = cluster
            else:
                used_cluster = cluster

            # Compute average predicted position
            positions = np.array([item[0] for item in used_cluster])
            mean_pos = positions.mean(axis=0)

            support = [{"source": item[1], "dir": item[2], "method": item[3], "pos": item[0]} for item in used_cluster]

            filtered.append((mean_pos, support))

        return filtered

    @classmethod
    def estimate_lattice_vectors_by_angle_clustering(
        cls,
        seed_points_xy: np.ndarray,
        k: int = 6,
        n_axes: int = 3,
        expected_spacing: float = 48.0,
        tolerance: float = 15.0,
    ) -> np.ndarray:
        """
        Estimate dominant lattice directions by clustering angles of neighbor vectors.

        Returns:
            lattice_vectors: n_axes x 2 array of unit lattice vectors scaled by mean spacing
        """
        min_dist = expected_spacing - tolerance
        max_dist = expected_spacing + tolerance

        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(seed_points_xy)
        distances, indices = nbrs.kneighbors(seed_points_xy)

        raw_vectors = []
        for i, idxs in enumerate(indices):
            origin = seed_points_xy[i]
            for j in idxs[1:]:  # skip self
                target = seed_points_xy[j]
                vec = target - origin
                dist = np.linalg.norm(vec)
                if min_dist <= dist <= max_dist:
                    raw_vectors.append(vec)

        if not raw_vectors:
            raise ValueError("No valid neighbor vectors found in the given distance range.")

        raw_vectors = np.array(raw_vectors)

        # Step 1: convert to angles in degrees and collapse opposite directions
        angles = np.arctan2(raw_vectors[:, 1], raw_vectors[:, 0])
        angles_deg = np.degrees(angles) % 180  # collapse opposite directions

        # Step 2: KMeans in angle space
        angle_kmeans = KMeans(n_clusters=n_axes, n_init=10, random_state=42)
        angle_kmeans.fit(angles_deg.reshape(-1, 1))
        centroid_angles = np.sort(angle_kmeans.cluster_centers_.flatten())  # degrees

        # Step 3: convert angle centroids back to 2D unit vectors
        unit_vectors = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in centroid_angles])

        # Step 4: scale lattice vectors by average neighbor distance
        avg_dist = np.mean([np.linalg.norm(v) for v in raw_vectors])
        lattice_vectors = unit_vectors * avg_dist

        return lattice_vectors
