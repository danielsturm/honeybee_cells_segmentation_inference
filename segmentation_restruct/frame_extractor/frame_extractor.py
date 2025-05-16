import subprocess
import platform
from pathlib import Path
import shutil
import tempfile


class FrameExtractor:
    def __init__(
        self,
        output_root_dir: Path,
        interval_sec: int = 1,
        fps: int = 6,
        file_format: str = "png",
    ):
        self.interval_sec = interval_sec
        self.video_fps = fps
        self.file_format = file_format
        self.ffmpeg_bin_path = self.determine_ffmpeg_bin_path()
        self.output_dir = self.create_output_dir(output_root_dir)

    def create_output_dir(self, output_root_dir: Path) -> Path:
        output_dir = output_root_dir / "extracted_frames"
        Path.mkdir(output_dir, exist_ok=True)
        return output_dir

    def determine_ffmpeg_bin_path(self) -> Path:
        bin_path = Path(__file__).parent.resolve() / "bin"
        system = platform.system()
        if system == "Windows":
            return bin_path / "ffmpeg.exe"
        elif system == "Linux":
            return bin_path / "ffmpeg_linux"
        else:
            raise OSError(f"Unsupported platform: {system}")

    def read_timestamps(self, txt_file: Path) -> list[str]:
        with txt_file.open("r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        return lines

    def extract_from(self, video_file_path: Path) -> None:
        txt_file = video_file_path.with_suffix(".txt")

        if not video_file_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_file_path}")
        if not txt_file.exists():
            raise FileNotFoundError(f"Timestamp file not found: {txt_file}")

        all_timestamps = self.read_timestamps(txt_file)
        step = self.interval_sec * self.video_fps
        selected_timestamps = all_timestamps[::step]
        frame_count = len(selected_timestamps)

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"tmp_{video_file_path.stem}_"))

        cmd = [
            str(self.ffmpeg_bin_path),
            "-y",
            "-c:v",
            "hevc_cuvid",
            "-i",
            str(video_file_path),
            "-vf",
            f"select='not(mod(n\\,{step}))'",
            # -vsync is deprecated, use "-fps_mode" instead
            "-vsync",
            "vfr",
            # "-q:v",
            # "2",  # good quality JPEG
            str(tmp_dir / f"frame_%05d.{self.file_format}"),
        ]

        # TODO: implement proper logging to file for errors
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        extracted_frames = sorted(tmp_dir.glob(f"frame_*.{self.file_format}"))
        if len(extracted_frames) != frame_count:
            raise RuntimeError(
                f"Mismatch: {len(extracted_frames)} frames vs {frame_count} timestamps"
            )

        for img_path, timestamp in zip(extracted_frames, selected_timestamps):
            new_path = self.output_dir / f"{timestamp}.{self.file_format}"
            shutil.move(str(img_path), str(new_path))

        shutil.rmtree(tmp_dir)

        print(
            f"Extracted {len(selected_timestamps)} frames from {video_file_path.name} to {self.output_dir}"
        )
