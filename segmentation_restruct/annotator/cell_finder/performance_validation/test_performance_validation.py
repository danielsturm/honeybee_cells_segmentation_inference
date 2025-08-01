import itertools
from pathlib import Path
from dataclasses import asdict

from segmentation_restruct.annotator.cell_finder.models import HexGraphConfig
from segmentation_restruct.annotator.cell_finder.cell_finder import CellFinder
from segmentation_restruct.annotator.cell_finder.performance_validation.performance_validation import (
    CellFindPerformanceValidator,
)


input_path = Path(
    r"C:\Users\sturmd\Documents\Development\Privates\honeybee_cells_segmentation_inference\segmentation_restruct\annotator\data\cell_finder_test_imgs"
)

threshold = 0.725
scale_factor = 0.425

cluster_pred_eps_values = [14.0, 15.0, 16.0, 17.0, 18.0]
bidirectional_values = [True, False]

results_summary = []

for cluster_pred_eps, bidirectional in itertools.product(cluster_pred_eps_values, bidirectional_values):

    graph_config_base = HexGraphConfig(
        curve_aware_candidate_pred=False,
        prefer_method="lattice_vector",
        neighbour_pos_tolerance=22,
        max_iterations=15,
        cluster_pred_eps=cluster_pred_eps,
        bidrectional_assignment=bidirectional,
    )

    cell_finder = CellFinder(input_dir=input_path, output_path=input_path)
    results = cell_finder.run_with_graph_building(threshold, scale_factor, graph_config_base)

    parameters = {"threshold": threshold, "scale_factor": scale_factor, "graph_config": asdict(graph_config_base)}
    cell_finder.save_artifacts(method="pattern matching", results=results, parameters=parameters)

    validator = CellFindPerformanceValidator(prediction_path=input_path)
    _, _, _, precision, recall, f1 = validator.run_performance_validation(visualize=True)

    results_summary.append(
        {
            "cluster_pred_eps": cluster_pred_eps,
            "bidirectional": bidirectional,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

sorted_results = sorted(results_summary, key=lambda r: r["f1"], reverse=True)

for i, r in enumerate(sorted_results, 1):
    print(
        f"{i:2d}. eps={r['cluster_pred_eps']:<4} | bidir={r['bidirectional']} "
        f"| F1={r['f1']:.4f} | Precision={r['precision']:.4f} | Recall={r['recall']:.4f}"
    )
