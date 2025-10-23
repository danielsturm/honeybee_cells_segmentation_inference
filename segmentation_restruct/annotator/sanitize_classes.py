from pathlib import Path
import json

segmentation_json_dir = Path(r"E:\Bachelorarbeit\final_dataset\sanitized_labels")


def find_json_files(path: Path):
    return path.glob("*.json")


def load_json_data_from_path(path: Path):
    assert path.is_file
    if path.stem.startswith("cell_finder_config"):
        return
    with open(path, "r") as f:
        cells = json.load(f)
        return cells


def unify_labels(input_dir: Path):
    to_be_unified_labels = ["pollen", "open_honey", "unclear_cell_class"]
    destination_class = "other_cell"
    files = find_json_files(input_dir)
    for file in files:
        data = load_json_data_from_path(file)
        if data is None:
            print("file has no data:", file)
            continue
        changed = False
        for obj in data:
            if obj.get("label") in to_be_unified_labels:
                obj["label"] = destination_class
                changed = True
        if changed:
            write_json_data_to_path(file, data)

    print("finished sanitizing")


def write_json_data_to_path(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


unify_labels(segmentation_json_dir)
