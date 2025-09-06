from pathlib import Path
import json
import matplotlib.pyplot as plt
from collections import Counter

label_map_path = Path(
    r"C:\Users\sturmd\Documents\Development\Privates\honeybee_cell_segmentation_pipeline\annotator\data\label_classes.json"
)
segmentation_json_dir = Path(
    r"C:\Users\sturmd\Documents\Development\Privates\honeybee_cell_segmentation_pipeline\annotator\data\input"
)
segmentation_masks_dir = Path(
    r"C:\Users\sturmd\Documents\Development\Privates\honeybee_cell_segmentation_pipeline\annotator\data\output"
)


def load_label_map():
    with open(label_map_path, "r") as f:
        map = json.load(f)
        print(map)


def load_json_data():
    data = []

    file_paths = segmentation_json_dir.glob("*.json")
    for path in file_paths:
        assert path.is_file
        if path.stem.startswith("cell_finder_config"):
            continue
        with open(path, "r") as f:
            cells = json.load(f)
            data.extend(cells)
    return data


def plot_label_distribution_old(data):
    labels = [item["label"] for item in data]
    label_counts = Counter(labels)
    classes = list(label_counts.keys())
    counts = list(label_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color="skyblue")
    plt.xlabel("Cell Class")
    plt.ylabel("Count")
    plt.title("Distribution of Cell Classes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_label_distribution(data):
    labels = [item["label"] for item in data]
    label_counts = Counter(labels)
    sorted_items = sorted(label_counts.items(), key=lambda x: x[1])
    classes = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    total = sum(counts)
    percentages = [count / total * 100 for count in counts]

    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=(6, 0.35 * len(classes)))
    bar_height = 0.4
    bars = plt.barh(classes, counts, color="mediumaquamarine", height=bar_height)
    plt.xlabel("Absolute count per class / relative distribution in %")
    # plt.ylabel("Cell Class")
    plt.title("Distribution of Cell Classes")

    # Annotate percentage at the tip of each bar
    for bar, pct in zip(bars, percentages):
        plt.text(
            bar.get_width() + total * 0.01,  # small offset
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
        )

    plt.tight_layout()
    plt.show()


all_data = load_json_data()
print(len(all_data))
print(all_data[-1])

plot_label_distribution(all_data)

load_label_map()
