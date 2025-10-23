from pathlib import Path
import argparse


class MaskWriter:
    def __init__(self, input_img_path: Path, input_json_path: Path, mask_out_path: Path) -> None:
        self.input_img_path = input_img_path
        assert self.input_img_path.is_dir(), f"images path {self.input_img_path} does not exist"
        self.json_in_path = input_json_path
        assert self.json_in_path.is_dir(), f"json path {self.json_in_path} does not exist"
        self.output_path = mask_out_path
        assert self.output_path.is_dir(), f"output path {self.output_path} does not exist"
        self.data_paths = self._find_data()
        self.masks_out_path = self._create_out_dir()

    def _load_label_map(self) -> dict[str, str]:
        # curr_path = Path(__file__).parent  # TODO: currently hardcoded
        with open(LABELS_PATH) as f:
            label_data = json.load(f)

        label_map = {"unlabeled": "#053aebff"}
        label_map.update({item["name"]: item["color"] for item in label_data})
        return label_map

    def _create_out_dir(self) -> Path:
        out_path = self.output_path / "ground_truth_masks"
        Path.mkdir(out_path, parents=True, exist_ok=True)
        return out_path

    def _find_data(self) -> dict[str, tuple[Path, Path]]:
        result = {}
        png_files = list(self.input_img_path.glob("*.png"))
        for png_file in png_files:
            json_file = self.json_in_path / f"{png_file.stem}.json"
            if json_file.exists():
                result[png_file.stem] = (png_file, json_file)
        return result

    def run(self) -> None:
        for data_tuple in self.data_paths.values():


def main():
    parser = argparse.ArgumentParser(description="Run to create ground truth masks from json files")
    parser.add_argument("input_path", type=str, help="Path to input data directory.")
    parser.add_argument(
        "--input_json_path", type=str, default=None, help="Optional path to output directory. Defaults to input path."
    )
    parser.add_argument(
        "--masks_output_path", type=str, default=None, help="Optional path to output directory. Defaults to input path."
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    input_json_path = Path(args.input_json_path) if args.input_json_path else input_path
    masks_output_path = Path(args.masks_output_path) if args.masks_output_path else input_path

    mask_creator = MaskWriter(input_path, input_json_path, masks_output_path)
    mask_creator.run()


if __name__ == "__main__":
    main()
