import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN, KMeans
from segmentation_restruct.annotator.cell_finder.utils import show_cells_on_image
from sklearn.neighbors import NearestNeighbors


def test_hough_lines(image_path):
    # 1. Read image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return

    # 2. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (17, 17), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # 3. Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 4. Apply Hough Line Transform
    # lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=150)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=200, minLineLength=150, maxLineGap=20)

    # 5. Draw lines on the original image if lines are found
    # if lines is not None:
    #     for line in lines:
    #         rho, theta = line[0]
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         # Calculate start and end points of the line segment
    #         x1 = int(x0 + 1000 * (-b))
    #         y1 = int(y0 + 1000 * (a))
    #         x2 = int(x0 - 1000 * (-b))
    #         y2 = int(y0 - 1000 * (a))

    #         # Draw the line on the image
    #         cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 6. Display results using matplotlib
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Hough Lines")
    plt.axis("off")
    plt.show()


def detect_frame(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to suppress internal comb edges
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Strong Canny thresholds to get only frame edges
    edges = cv2.Canny(blurred, 100, 200, apertureSize=3)

    # Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=300, maxLineGap=20)

    # Draw detected lines
    img_lines = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display edges and lines for debugging
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(edges, cmap="gray")
    plt.title("Edges")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    plt.title("Detected Lines")
    plt.axis("off")

    plt.show()

    return img_lines


def detect_frame_contour(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur to suppress noise
    blurred = cv2.GaussianBlur(gray, (17, 17), 0)

    # Strong thresholding to isolate the frame
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = img.copy()

    max_area = 0
    frame_contour = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 4:  # Looking for quadrilateral
                max_area = area
                frame_contour = approx

    # Draw detected frame
    if frame_contour is not None:
        cv2.drawContours(img_contours, [frame_contour], -1, (0, 0, 255), 3)
    else:
        print("No quadrilateral frame contour detected.")

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(thresh, cmap="gray")
    plt.title("Thresholded Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    plt.title("Detected Frame Contour")
    plt.axis("off")

    plt.show()

    # return frame_contour


def detect_frame_downscaled(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return

    # Downscale
    scale = 0.25
    small = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Strong blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # High Canny thresholds
    edges = cv2.Canny(blurred, 150, 250)

    # Probabilistic Hough Transform with high minLineLength
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20)

    img_lines = small.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Show results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(edges, cmap="gray")
    plt.title("Edges")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    plt.title("Detected Lines (Downscaled)")
    plt.axis("off")

    plt.show()


def my_test(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found.")
        return

    scale = 0.1
    small = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 150, 250)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=20)

    # Draw detected lines
    img_lines = small.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(edges, cmap="gray")
    plt.title("Edges")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    plt.title("Detected Lines")
    plt.axis("off")

    plt.show()


# my_test(
#     r"C:\Users\sturmd\Documents\Development\Privates\honeybee_cells_segmentation_inference\segmentation_restruct\annotator\data\input\20240603_cam-2.png"
# )


def generate_hexagon_template(radius: int = 24, margin: int = 20) -> None:

    size = 2 * (radius + margin)
    center = (size // 2, size // 2)

    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    points = np.array(
        [(int(center[0] + radius * np.cos(angle)), int(center[1] + radius * np.sin(angle))) for angle in angles],
        dtype=np.int32,
    )
    points = points.reshape((-1, 1, 2))
    points_list = [points]

    img = np.zeros((size, size), dtype=np.uint8)
    cv2.polylines(img, points_list, isClosed=True, color=255, thickness=2)

    plt.figure(figsize=(15, 15))
    plt.imshow(img)
    plt.title("Template")
    plt.axis("off")
    plt.show()


# generate_hexagon_template()


def test_cell_templates():
    templates_path = Path(
        r"C:\Users\sturmd\Documents\Development\Privates\honeybee_cells_segmentation_inference\segmentation_restruct\annotator\cell_finder"
    )
    templates = list(templates_path.glob("*.png"))
    for template in templates:
        img = cv2.imread(str(template), cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, 100, 250, apertureSize=3)
        plt.figure(figsize=(15, 15))
        plt.imshow(edges)
        plt.title(template.stem)
        plt.axis("off")
        plt.show()


# test_cell_templates()


def test_cell_estimation_pipeline():
    img_path = Path(
        r"C:\Users\sturmd\Documents\Development\Privates\honeybee_cells_segmentation_inference\segmentation_restruct\annotator\cell_finder\performance_validation\ground_truth\20240603_cam-2.png"
    )
    template_folder = Path(__file__).parent / "pattern_matching"
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    clahe_img = _apply_clahe(gray)
    temp_match_res = _template_matching(gray_image=clahe_img, template_folder=template_folder)
    filtered_matches = _non_max_suppression(temp_match_res)

    seed_points = np.array([(x, y) for x, y, _, _ in filtered_matches])

    # lattice_vectors = estimate_lattice_vectors(seed_points)
    lattice_vectors = estimate_lattice_vectors_by_angle_clustering(seed_points)

    # color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # show_cells_on_image(color, filtered_matches)
    graph = HexLatticeGraph(
        seed_points=seed_points,
        lattice_vectors=lattice_vectors,
        tolerance=17.0,
        image_size=tuple(reversed(gray.shape)),
        bidirectional=False,
    )
    graph.grow_graph_iteratively(n_iterations=5)

    # # 5. Add all seed points to the graph
    # for pt in seed_points:
    #     graph.add_node(pt)

    # # 6. Build edges between nodes using lattice directions
    # graph.build_edges()
    # # print(graph.nodes)

    # candidates = graph.collect_missing_neighbor_candidates(curve_aware=True)
    # validated = graph.cluster_and_filter_candidates(candidates)
    # visualize_hex_lattice_graph(graph, candidates=candidates, validated=validated)

    # # Done: now graph.nodes contains the full structure
    # print(f"Graph contains {len(graph.nodes)} nodes.")
    # print(gray.shape)
    show_lattice_vectors_on_image(clahe_img, seed_points, lattice_vectors)


def _template_matching(
    gray_image,
    template_folder: Path,
    threshold: float = 0.725,
    scale_factor: float = 0.425,
) -> list[tuple]:
    assert template_folder.is_dir(), f"Template folder does not exist: {template_folder}"

    results = []
    for template_path in template_folder.glob("*.png"):
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue

        w, h = template.shape[::-1]
        result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1]):
            score = result[pt[1], pt[0]]
            center_x = pt[0] + w // 2
            center_y = pt[1] + h // 2
            # radius = int(min(w, h) * scale_factor)
            radius = 24
            results.append((center_x, center_y, radius, score))

    return results


def _apply_clahe(img, clipLimit=2.0, tileGridSize=(8, 8)):

    if img.dtype in [np.float32, np.float64]:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(img)


def _non_max_suppression(matches, overlap_thresh=0.3):
    if len(matches) == 0:
        return []

    boxes = np.array([[x - r, y - r, x + r, y + r] for x, y, r, _ in matches])
    scores = np.array([score for *_, score in matches])

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]

    return [matches[i] for i in keep]


def estimate_lattice_vectors(seed_points_xy: np.ndarray, k: int = 6, n_vectors: int = 3):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(seed_points_xy)
    distances, indices = nbrs.kneighbors(seed_points_xy)

    vectors = []
    for i, idxs in enumerate(indices):
        origin = seed_points_xy[i]
        for j in idxs[1:]:  # skip self
            target = seed_points_xy[j]
            vec = target - origin
            vectors.append(vec)

    vectors = np.array(vectors)
    if len(vectors) < n_vectors:
        raise ValueError("Not enough vectors to estimate lattice directions.")

    kmeans = KMeans(n_clusters=n_vectors, n_init=10, random_state=42).fit(vectors)
    lattice_vectors = kmeans.cluster_centers_

    return lattice_vectors


def estimate_lattice_vectors_by_angle_clustering(
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


# def show_lattice_vectors_on_image(image, seed_points, lattice_vectors):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image, cmap="gray")
#     seed_points = np.array(seed_points)
#     plt.scatter(seed_points[:, 0], seed_points[:, 1], color="red", s=5)

#     center = seed_points.mean(axis=0)
#     for vec in lattice_vectors:
#         plt.arrow(center[0], center[1], vec[0], vec[1], head_width=20, head_length=30, color="cyan", linewidth=2)


#     plt.title("Estimated Lattice Vectors from Seeds")
#     plt.axis("equal")
#     plt.show()


def show_lattice_vectors_on_image(image, seed_points, lattice_vectors, scale=2.5):
    import matplotlib.pyplot as plt

    seed_points = np.array(seed_points)
    center = seed_points.mean(axis=0)

    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray")
    plt.scatter(seed_points[:, 0], seed_points[:, 1], color="red", s=5, label="Detected cells")

    # Normalize and scale vectors for visibility
    for i, vec in enumerate(lattice_vectors):
        vec_scaled = vec * scale  # scale vector length for visualization
        plt.arrow(
            center[0], center[1], vec_scaled[0], vec_scaled[1], head_width=20, head_length=30, color="cyan", linewidth=2
        )
        end_x = center[0] + vec_scaled[0]
        end_y = center[1] + vec_scaled[1]
        plt.text(end_x, end_y, f"v{i + 1}", color="cyan", fontsize=12)

    print("Center of vectors (origin):", center)
    print("Estimated lattice vectors:")
    for i, vec in enumerate(lattice_vectors):
        print(f"v{i + 1}: {vec}")

    plt.title("Estimated Lattice Vectors from Detected Cells")
    plt.axis("equal")
    plt.legend()
    plt.show()


@dataclass
class CellNode:
    id: str
    position: np.ndarray  # shape (2,)
    neighbors: dict = field(default_factory=lambda: {i: None for i in range(6)})


class HexLatticeGraph:
    def __init__(self, seed_points, lattice_vectors, tolerance=15.0, image_size=None, bidirectional=True):
        self.seed_points = seed_points
        self.nodes = {}  # UUID → CellNode
        self.pos_index = []  # list of np.ndarray positions
        self.id_index = []  # list of UUIDs (same index as pos_index)
        self.vecs = [np.array(v) for v in lattice_vectors]  # v1, v2, v3
        self.tolerance = tolerance
        self.image_size = image_size  # (width, height), optional
        self.bidirectional = bidirectional

    def grow_graph_iteratively(self, n_iterations=3, verbose=False):
        for pt in self.seed_points:
            self.add_node(pt)

        self.build_edges()

        print("only seed points", self.nodes)
        print("as many node", len(self.nodes))
        visualize_hex_lattice_graph(self)

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

            visualize_hex_lattice_graph(self, candidates=raw_candidates, validated=validated)

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

            print("only after rebuilding", self.nodes)
            print("as many node", len(self.nodes))
            visualize_hex_lattice_graph(self)

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
        assert (1528, 572) in [tuple(p) for p in self.pos_index]

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
                    method = "fallback"

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
                method_groups = defaultdict(list)
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


def visualize_hex_lattice_graph(graph, candidates=None, validated=None, max_nodes=300, figsize=(12, 12)):
    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes
    node_ids = list(graph.nodes.keys())  # [:max_nodes]
    for node_id in node_ids:
        node = graph.nodes[node_id]
        x, y = node.position
        ax.plot(x, y, "o", color="black", markersize=4)

        for dir_idx, neighbor in node.neighbors.items():
            if neighbor is None:
                # draw missing neighbor indicator
                vec = (graph.vecs + [-v for v in graph.vecs])[dir_idx]
                end = node.position + vec
                ax.plot([x, end[0]], [y, end[1]], linestyle="dotted", color="gray", alpha=0.3)
            elif neighbor == "OUT_OF_BOUNDS":
                vec = (graph.vecs + [-v for v in graph.vecs])[dir_idx]
                end = node.position + vec
                ax.plot([x, end[0]], [y, end[1]], linestyle="dashed", color="red", alpha=0.5)
            else:
                # draw real connection
                if neighbor in graph.nodes:
                    neighbor_node = graph.nodes[neighbor]
                    nx, ny = neighbor_node.position
                    ax.plot([x, nx], [y, ny], color="blue", linewidth=1, alpha=0.7)

    # --- Visualize predicted candidate cells ---
    # if candidates is not None:
    #     for pos, _, _ in candidates:
    #         ax.plot(pos[0], pos[1], "rx", markersize=5)  # Red 'x' for missing neighbor
    if candidates is not None:
        for pos, _, _, method in candidates:
            if method == "curve":
                ax.plot(pos[0], pos[1], marker="x", color="green", markersize=5)
            else:
                ax.plot(pos[0], pos[1], marker="x", color="red", markersize=5)

    # --- Visualize validated aggregated predictions ---
    if validated is not None:
        for final_pos, support in validated:
            ax.plot(final_pos[0], final_pos[1], marker="*", color="orange", markersize=5)

    ax.set_aspect("equal")
    ax.set_title("Hex Lattice Graph Visualization")
    ax.invert_yaxis()  # for image-like top-down view

    # Add legend
    legend_items = [
        mpatches.Patch(color="blue", label="Connected neighbors"),
        mpatches.Patch(color="red", label="Out of bounds (dashed)"),
        mpatches.Patch(color="gray", label="Missing (dotted)"),
        mpatches.Patch(color="green", label="Predicted (curve)"),
        mpatches.Patch(color="red", label="Predicted (lattice)"),
        mpatches.Patch(color="orange", label="Validated (clustered)"),
    ]
    ax.legend(handles=legend_items)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # def build_edges(self):
    #     positions = np.array(self.pos_index)
    #     search_radius = max(np.linalg.norm(v) for v in self.vecs) + self.tolerance
    #     nbrs = NearestNeighbors(radius=search_radius)
    #     nbrs.fit(positions)

    #     for i, node_id in enumerate(self.id_index):
    #         origin = self.nodes[node_id]

    #         neighbors_idx = nbrs.radius_neighbors([origin.position], return_distance=False)[0]

    #         for j in neighbors_idx:
    #             if i == j:
    #                 continue
    #             target_pos = positions[j]
    #             vec = target_pos - origin.position

    #             dir_index = self._vector_direction_index(vec)
    #             if dir_index is None:
    #                 continue

    #             # Validate spatial consistency
    #             predicted_pos = origin.position + (self.vecs + [-v for v in self.vecs])[dir_index]
    #             if not self._in_bounds(predicted_pos):
    #                 origin.neighbors[dir_index] = "OUT_OF_BOUNDS"
    #                 continue

    #             neighbor_id = self.id_index[j]
    #             origin.neighbors[dir_index] = neighbor_id

    #             if self.bidirectional:
    #                 reverse_index = (dir_index + 3) % 6
    #                 self.nodes[neighbor_id].neighbors[reverse_index] = node_id


# class HexLatticeGraph:
#     def __init__(self, lattice_vectors, tolerance=10.0):
#         self.nodes = {}  # id → CellNode
#         self.pos_index = []  # flat list of positions
#         self.id_index = []  # matching list of ids
#         self.vecs = [np.array(v) for v in lattice_vectors]
#         self.tolerance = tolerance

#     def add_node(self, position):
#         node_id = str(uuid.uuid4())
#         node = CellNode(id=node_id, position=np.array(position))
#         self.nodes[node_id] = node
#         self.pos_index.append(node.position)
#         self.id_index.append(node_id)
#         return node

#     def build_edges(self):

#         positions = np.array(self.pos_index)
#         nbrs = NearestNeighbors(radius=np.linalg.norm(self.vecs[0]) + self.tolerance)
#         nbrs.fit(positions)

#         for i, node_id in enumerate(self.id_index):
#             origin = self.nodes[node_id]
#             neighbors_idx = nbrs.radius_neighbors([origin.position], return_distance=False)[0]

#             for j in neighbors_idx:
#                 if i == j:
#                     continue
#                 neighbor_pos = positions[j]
#                 direction = neighbor_pos - origin.position

#                 # Match direction to one of the lattice vectors
#                 for d, vec in enumerate(self.vecs + [-v for v in self.vecs]):
#                     if np.linalg.norm(direction - vec) <= self.tolerance:
#                         dir_index = d if d < 3 else d - 3  # map to 0–5
#                         origin.neighbors[dir_index] = self.id_index[j]
#                         break


test_cell_estimation_pipeline()
