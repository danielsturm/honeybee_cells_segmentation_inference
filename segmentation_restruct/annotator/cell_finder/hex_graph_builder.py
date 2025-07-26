from dataclasses import dataclass, field
import numpy as np
import uuid
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN, KMeans
from collections import defaultdict
from typing import Literal
from scipy.spatial import KDTree

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
    prefer_method: Literal["curve", "lattice_vector"] | None = "curve"
    debug: bool = False


class HexLatticeGraph:
    def __init__(self, seed_points, lattice_vectors, config: HexGraphConfig):
        self.seed_points = seed_points
        self.nodes = {}
        self.pos_index = []  # list of np.ndarray positions
        self.id_index = []  # list of UUIDs (same index as pos_index)
        self.vecs = [np.array(v) for v in lattice_vectors]  # v1, v2, v3
        self.config = config

    @property
    def cell_positions(self) -> list[tuple[int, int, int, float]]:
        return [(*cell, 24, 1.0) for cell in self.pos_index]

    def grow_graph_iteratively(self, verbose=False):
        for pt in self.seed_points:
            self.add_node(pt)

        self.build_edges()

        # visualize_hex_lattice_graph(self)

        for i in range(self.config.max_iterations):
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

            if self.config.debug:
                visualize_hex_lattice_graph(self, candidates=raw_candidates, validated=validated)

            if verbose:
                print(f"Validated {len(validated)} new cells")

            if not validated:
                if verbose:
                    print("No validated predictions passed filtering. Stopping.")
                break

            conflict_resolved = self.resolve_prediction_conflicts(validated=validated)
            if self.config.debug:
                visualize_hex_lattice_graph(self, candidates=raw_candidates, validated=validated)

            # --- 3. Add new predicted nodes ---
            new_nodes = 0
            for pos, support in conflict_resolved:
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

    def resolve_prediction_conflicts(self, validated: list[dict], merge_radius: float = 22.0):
        if not validated:
            return []

        positions = np.array([pos for pos, _ in validated])
        clustering = DBSCAN(eps=merge_radius, min_samples=1).fit(positions)
        labels = clustering.labels_

        grouped = defaultdict(list)
        for idx, label in enumerate(labels):
            grouped[label].append(validated[idx])

        final_validated = []

        occupied = KDTree(np.array(self.pos_index)) if self.pos_index else None

        for group in grouped.values():

            # like a normal prediction
            if len(group) == 1:
                final_validated.append(group[0])
                continue

            # Too many predictions clustered — discard all, mark their predictor edges
            if len(group) > 3:
                for _, support in group:
                    for s in support:
                        pid = s["source"]
                        dir_idx = s["dir"]
                        if pid in self.nodes:
                            self.nodes[pid].neighbors[dir_idx] = "CONFLICT"
                continue

            # Merge support and check for conflicting predictor directions
            seen = set()
            conflict = False
            all_support = []
            for _, support in group:
                for s in support:
                    pair = (s["source"], s["dir"])
                    if pair in seen:
                        conflict = True
                        break
                    seen.add(pair)
                    all_support.append(s)
                if conflict:
                    break

            if conflict:
                for s in all_support:
                    pid = s["source"]
                    dir_idx = s["dir"]
                    if pid in self.nodes:
                        self.nodes[pid].neighbors[dir_idx] = "CONFLICT"
                continue

            # No conflict — merge and assign directly
            merged_pos = np.mean([pos for pos, _ in group], axis=0)
            self._add_node_with_edge_assignments(merged_pos, all_support)
            # final_validated.append((node.position, all_support)) # not needed because already done

        return final_validated

    def _add_node_with_edge_assignments(self, position: np.ndarray, support: list[dict]):
        rounded_pos = np.round(position).astype(int)
        node = self.add_node(rounded_pos)  # Uses standard UUID assignment

        for s in support:
            pred_id = s["source"]
            dir_idx = s["dir"]

            if pred_id not in self.nodes:
                continue

            # Link predictor → new node
            self.nodes[pred_id].neighbors[dir_idx] = node.id

            if self.config.bidrectional_assignment:
                reverse_dir = (dir_idx + 3) % 6
                self.nodes[node.id].neighbors[reverse_dir] = pred_id

        return node

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
        if self.config.image_size is None:
            return True
        x, y = pos
        width, height = self.config.image_size
        return 0 <= x < width and 0 <= y < height

    def build_edges(self):

        positions = np.array(self.pos_index)
        search_radius = max(np.linalg.norm(v) for v in self.vecs) + self.config.neighbour_pos_tolerance
        nbrs = NearestNeighbors(radius=search_radius)
        nbrs.fit(positions)

        for i, node_id in enumerate(self.id_index):
            origin = self.nodes[node_id]
            origin_pos = origin.position

            for dir_index, vec in enumerate(self.vecs + [-v for v in self.vecs]):
                # Skip if own edge is marked as NOT_SURE
                if origin.neighbors.get(dir_index) == "NOT_SURE":
                    continue

                predicted_pos = origin_pos + vec

                if not self._in_bounds(predicted_pos):
                    origin.neighbors[dir_index] = "OUT_OF_BOUNDS"
                    continue

                # Find actual neighbor close to predicted position
                distances = np.linalg.norm(positions - predicted_pos, axis=1)
                close_indices = np.where(distances <= self.config.neighbour_pos_tolerance)[0]

                if len(close_indices) > 0:

                    # TODO: all these neighbors have to have the same id.
                    neighbor_id = self.id_index[close_indices[0]]

                    # skip if reverse neighbor has direction marked as not sure
                    neighbor_node = self.nodes[neighbor_id]
                    reverse_dir = (dir_index + 3) % 6
                    if neighbor_node.neighbors.get(reverse_dir) == "NOT_SURE":
                        continue

                    # assign forward edge
                    origin.neighbors[dir_index] = neighbor_id

                    # Optional: bidirectional assignment
                    if self.config.bidrectional_assignment:
                        reverse_dir = (dir_index + 3) % 6
                        self.nodes[neighbor_id].neighbors[reverse_dir] = node_id
                else:
                    origin.neighbors[dir_index] = None

    def collect_missing_neighbor_candidates(self):

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

                if self.config.curve_aware_candidate_pred:
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

    def cluster_and_filter_candidates(self, candidates):

        if not candidates:
            return []

        # Extract just the position vectors for clustering
        pos_array = np.array([pos for (pos, *_) in candidates])

        # Cluster nearby predictions using DBSCAN
        clustering = DBSCAN(eps=self.config.dbscan_eps, min_samples=1).fit(pos_array)
        labels = clustering.labels_

        clustered = defaultdict(list)

        # Group candidates by cluster label
        for idx, label in enumerate(labels):
            clustered[label].append(candidates[idx])

        filtered = []

        for cluster in clustered.values():
            if len(cluster) < self.config.cluster_vote_threshold:
                continue  # not enough support

            # Split by method if needed
            if self.config.prefer_method:
                method = self.config.prefer_method
                method_groups = defaultdict(list)  # why is this needed
                for item in cluster:
                    method = item[3]
                    method_groups[method].append(item)

                # Use preferred method group if enough votes
                if method in method_groups and len(method_groups[method]) >= self.config.cluster_vote_threshold:
                    used_cluster = method_groups[method]
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


# =========================================================================================================

# TODO: Function to remove two cells that are too close to each other, when building edges
# TODO: can it happen that A assigns B as neighbor but B does not do with A? relates to bidirectional assignment
# TODO: lattice vector fallback in position prediction is applied immediately. could also be applied later, if curve gets stuck and still missing neighbors
# TODO: prefer method in cluster_and_filter_candidates only filters out the prefered method and ignores the other. if the prefered are too few, it skips.
#   should be rather taking all candidates and just weight the ones that are prefered when calculating new point

# =========================================================================================================

# - if two final predictions are too close they are both removed and all valid cells around that have an open edge in this direction are marked at the edge as not_sure
# - or at least find the cells that predicted that predicton and assign the edges as not_sure
# - or two final predictions that are within a distance (eg 20) will form their mean (a super prediction) and all predictors will conntect to this guy. (will that work with build_edges?)
# - if the predicted point is too close to a real point, both could be removed. the real point just removed and the predicted also but also assign its predictors as not_sure
# - alternative: if predicted point too close to existing node, just assign the edges of the predictors to that existing node.
# - related to the previous. could also first check if the real point has a missing edges in that direction of the predictors. if not just mark as not suere
# - what can you do about the iterations. i dont know how long it takes until all cells are found (could wait until only few things are changing) but if there is an error

# possiblities:
# - two or three final_predicted fall together
#   - compute the average position and assign the edges to the predictors of each final_predicted
#   - check if all predictors predict their final_predicted to different directions.
#   - if two or more predictors predict to the same direction discard all final_predicted and mark the related missing edges of predictors as not_sure
# - more than three final_predicted fall together (each node has six neighbors and min two build a final_predicted)
#   - this should not happen. discard all and mark the edges of all predictors of all final_predicted as not_sure
# - one final_predicted is next/too close to an existing_node
#   - does this existing_node have missing edges in the direction of the predictors?
#   - do the predictors of the final_predicted have missing edges in that direction? (this is not relevant. should not happen, otherwise the would not predict)
#   - if yes merge.
#   - if no, discard and mark posssible open edges as not_sure
# - a final_predicted is next to another final_predicted and existing_node
#   - in this case discard all final_predicted and mark all edges of the responsible predictors as not_sure
# - two existing_nodes should be neighbors but are not connected, because the angle is too much off or they are too distant
#   - could increase the tolerance but don't know if this affects the rest of the procedure.
#   - leve it for now. maybe these missing edges can be ignored

#  =========================================================================================================
# Stopping Problem
# - create a flag that is set to true each time a node is added to the graph per loop. if the flag is true the loop continues.
# - still have a max_iterations number to prevent infinite loop in an unforseen case
