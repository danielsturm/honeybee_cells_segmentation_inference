import time
from functools import wraps
from pydantic import BaseModel
from typing import Literal


def timed(name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            label = name or func.__name__
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            print(f"[TIMER] {label} took {end - start:.2f} seconds")
            return result

        return wrapper

    return decorator


class BgImageGenConfig(BaseModel):
    window_size: int = 10
    num_median_images: int | None = 48
    apply_clahe: Literal["intermediate", "post"] = "post"
    mask_dilation: Literal[0, 9, 15, 25] = 0
    # frame_interval_in_sec: int = 3
    median_computation: Literal["cupy", "cuda_support", "masked_array"] = "cupy"
    segmentation_model: Literal["unet_effnetb0"] = "unet_effnetb0"
    device: Literal["cuda", "cpu"] = "cuda"
