import itertools
from pathlib import Path
from dataclasses import asdict
from typing import cast, Literal
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

from segmentation_restruct.annotator.cell_finder.models import HexGraphConfig
from segmentation_restruct.annotator.cell_finder.cell_finder import CellFinder
from segmentation_restruct.annotator.cell_finder.performance_validation.performance_validation import (
    CellFindPerformanceValidator,
)


def analyze_result_file(result_file_path: Path):
    with open(result_file_path, "r") as f:
        data = json.load(f)

    df = pd.json_normalize(data)

    config_cols = [col for col in df.columns if col.startswith("config.")]
    X = df[config_cols].select_dtypes(include=["number"])
    y = df["f1"]

    rf = RandomForestRegressor(random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)

    correlation = X.corrwith(y)
    summary = pd.DataFrame({"correlation_with_f1": correlation, "rf_importance": importances}).sort_values(
        "rf_importance", ascending=False
    )

    print("\n Parameter Influence Analysis:")
    print(summary)

    print("\n Best-performing values per parameter (based on mean F1):")
    best_values = {}
    for param in X.columns:
        grouped = df.groupby(param)["f1"].mean()
        best_val = grouped.idxmax()
        best_score = grouped.max()
        best_values[param.replace("config.", "")] = best_val
        print(f"  {param.replace('config.', '')}: {best_val} (mean F1 = {best_score:.4f})")

    summary["rf_importance"].sort_values().plot(kind="barh", title="Feature Importance (F1)", figsize=(8, 5))
    plt.tight_layout()
    plt.show()


input_path = Path(
    r"C:\Users\sturmd\Documents\Development\Privates\honeybee_cells_segmentation_inference\segmentation_restruct\annotator\data\cell_finder_test_imgs"
)

threshold = 0.725
scale_factor = 0.425

neighbour_pos_tolerance_values = [20.2, 20.5, 20.7]
bidirectional_values = [True, False]
cluster_pred_eps_values = [16.0, 16.5, 17.0, 17.5, 18.0, 18.5]
pred_merge_dist_values = [27.0, 27.5]
cluster_conflict_pred_eps_values = [28.5, 29.0]
min_dist_nodes_values = [33.5]

valid_curve_pairs_values = [
    (False, "lattice_vector"),
    (True, "curve"),
]

results_summary = []

for (
    neighbour_pos_tolerance,
    cluster_pred_eps,
    pred_merge_dist,
    cluster_conflict_pred_eps,
    min_dist_nodes,
    # (curve_aware_candidate_pred, prefer_method),
) in itertools.product(
    neighbour_pos_tolerance_values,
    cluster_pred_eps_values,
    pred_merge_dist_values,
    cluster_conflict_pred_eps_values,
    min_dist_nodes_values,
    # valid_curve_pairs_values,
):

    graph_config_base = HexGraphConfig(
        curve_aware_candidate_pred=True,
        prefer_method="curve",
        neighbour_pos_tolerance=neighbour_pos_tolerance,
        max_iterations=15,
        cluster_pred_eps=cluster_pred_eps,
        pred_merge_dist=pred_merge_dist,
        cluster_conflict_pred_eps=cluster_conflict_pred_eps,
        bidrectional_assignment=False,
        min_dist_nodes=min_dist_nodes,
    )

    cell_finder = CellFinder(input_dir=input_path, output_path=input_path)
    results = cell_finder.run_with_graph_building(threshold, scale_factor, graph_config_base)

    parameters = {"threshold": threshold, "scale_factor": scale_factor, "graph_config": asdict(graph_config_base)}
    cell_finder.save_artifacts(method="pattern matching", results=results, parameters=parameters)

    validator = CellFindPerformanceValidator(prediction_path=input_path)
    total_tp, total_fp, total_fn, precision, recall, f1, report_name = validator.run_performance_validation(
        visualize=True
    )

    results_summary.append(
        {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "config": asdict(graph_config_base),
            "name": report_name,
        }
    )

sorted_results = sorted(results_summary, key=lambda r: r["f1"], reverse=True)

results_path: Path = Path(__file__).parent / "results"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
result_file = results_path / f"{timestamp}_results.json"

with open(result_file, "w") as f:
    json.dump(sorted_results, f, indent=2)

print(json.dumps(sorted_results, indent=2))
analyze_result_file(result_file)
