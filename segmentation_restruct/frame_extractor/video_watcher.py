from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileSystemEvent
import threading
import time
from pathlib import Path
from frame_extractor import FrameExtractor


class VideoWatcher(FileSystemEventHandler):
    def __init__(
        self,
        watch_dir: Path,
        out_dir: Path,
        file_format: str = "png",
        interval_sec=5,
        max_workers=2,
    ):
        self.watch_dir = watch_dir
        self.file_format = file_format
        self.frame_extractor = FrameExtractor(
            output_root_dir=out_dir, interval_sec=interval_sec
        )
        self.processing_files = set()
        self.semaphore = threading.Semaphore(max_workers)

    def on_created(self, event: FileSystemEvent) -> None:
        if not isinstance(event, FileCreatedEvent):
            return
        path = Path(str(event.src_path))
        if path.suffix.lower() == ".mp4":
            if path in self.processing_files:
                return
            threading.Thread(
                target=self.wait_for_and_process_video, args=(path,), daemon=True
            ).start()

    def wait_until_file_ready(self, path: Path, timeout: int = 60) -> None:
        last_size = -1
        for _ in range(timeout):
            size = path.stat().st_size
            if size == last_size:
                return
            last_size = size
            time.sleep(1)
        raise TimeoutError(f"File {path} is not stable.")

    def wait_for_and_process_video(self, path: Path) -> None:
        with self.semaphore:
            try:
                self.wait_until_file_ready(path)

                txt_path = path.with_suffix(".txt")
                for _ in range(10):
                    if txt_path.exists():
                        break
                    time.sleep(1)
                else:
                    print(f"Warning: No .txt file for {path.name}, skipping.")
                    return

                if self.is_already_processed(path):
                    print(f"Info: Video already processed: {path.name}")
                    return

                self.processing_files.add(path)
                self.process_video(path)
            finally:
                self.processing_files.discard(path)

    def is_already_processed(self, video_path: Path) -> bool:
        txt_file = video_path.with_suffix(".txt")
        if not txt_file.exists():
            return False
        try:
            timestamps = self.frame_extractor.read_timestamps(txt_file)
            step = self.frame_extractor.interval_sec * self.frame_extractor.video_fps
            selected_timestamps = timestamps[::step]
            if not selected_timestamps:
                return False
            first_frame = (
                self.frame_extractor.output_dir
                / f"{selected_timestamps[0]}.{self.file_format}"
            )
            return first_frame.exists()
        except Exception:
            return False

    def process_video(self, path: Path):
        try:
            self.wait_until_file_ready(path)
            self.frame_extractor.extract_from(path)
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.processing_files.discard(path)

    def start(self):

        for path in self.watch_dir.glob("*.mp4"):
            if path not in self.processing_files:
                threading.Thread(
                    target=self.wait_for_and_process_video, args=(path,), daemon=True
                ).start()

        observer = Observer()
        observer.schedule(self, str(self.watch_dir), recursive=False)
        observer.start()
        print(f"Watching {self.watch_dir}")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()


if __name__ == "__main__":
    # path = Path(__file__).parent / "playground"
    out_dir = Path(r"D:\Bachelorarbeit\bee_videos")
    in_dir = Path(r"D:\Bachelorarbeit\bee_videos\cam-0")
    watcher = VideoWatcher(
        watch_dir=in_dir, out_dir=out_dir, interval_sec=3, max_workers=2
    )
    watcher.start()
