from pathlib import Path


def print_file_stems(directory: str) -> None:
    path = Path(directory)
    if not path.is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        return

    for file in path.iterdir():
        if file.is_file():
            print(file.stem)


print_file_stems(
    r"C:\Users\sturmd\OneDrive - Televic Group NV\Dokumente\Development\Privates\semantic-segmentation-template\data\val\images"
)
