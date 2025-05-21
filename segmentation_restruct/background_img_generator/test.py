import segmentation_restruct.background_img_generator.background_img_generator as bg
from pathlib import Path
from segmentation_restruct.utils import BgImageGenConfig

config_1 = BgImageGenConfig(
    window_size=10,
    num_median_images=48,
    apply_clahe="post",
    mask_dilation=0,
    median_computation="cupy",
)
config_2 = BgImageGenConfig(
    window_size=10,
    num_median_images=100,
    apply_clahe="post",
    mask_dilation=0,
    median_computation="cupy",
)
config_3 = BgImageGenConfig(
    window_size=10,
    num_median_images=300,
    apply_clahe="post",
    mask_dilation=0,
    median_computation="cupy",
)
config_3a = BgImageGenConfig(
    window_size=10,
    num_median_images=300,
    apply_clahe="intermediate",
    mask_dilation=0,
    median_computation="cupy",
)
config_3b = BgImageGenConfig(
    # like jacobs configuration
    window_size=10,
    num_median_images=300,
    apply_clahe="post",
    mask_dilation=0,
    median_computation="masked_array",
)
config_4 = BgImageGenConfig(
    window_size=10,
    num_median_images=500,
    apply_clahe="post",
    mask_dilation=0,
    median_computation="cupy",
)
config_5 = BgImageGenConfig(
    window_size=10,
    num_median_images=800,
    apply_clahe="post",
    mask_dilation=0,
    median_computation="cupy",
)
config_6 = BgImageGenConfig(
    window_size=10,
    num_median_images=100,
    apply_clahe="intermediate",
    mask_dilation=0,
    median_computation="cupy",
)
config_6b = BgImageGenConfig(
    window_size=10,
    num_median_images=100,
    apply_clahe="intermediate",
    mask_dilation=15,
    median_computation="cupy",
)
config_7a = BgImageGenConfig(
    window_size=10,
    num_median_images=500,
    apply_clahe="intermediate",
    mask_dilation=0,
    median_computation="cupy",
)
config_5a = BgImageGenConfig(
    window_size=10,
    num_median_images=360,
    apply_clahe="intermediate",
    mask_dilation=15,
    median_computation="cupy",
)
config_5b = BgImageGenConfig(
    window_size=10,
    num_median_images=700,
    apply_clahe="intermediate",
    mask_dilation=15,
    median_computation="cupy",
)
config_8 = BgImageGenConfig(
    window_size=10,
    num_median_images=100,
    apply_clahe="intermediate",
    mask_dilation=0,
    median_computation="masked_array",
)
config_9 = BgImageGenConfig(
    window_size=10,
    num_median_images=500,
    apply_clahe="intermediate",
    mask_dilation=15,
    median_computation="cupy",
)


def test_bg_img_gen(config):

    out_dir = Path("C:/Users/sturmd/Desktop/Bachelorarbeit/bee_data/20240529/cam-2")
    big = bg.BackgroundImageGenerator(
        source_path=out_dir, output_path=out_dir, config=config
    )

    big.mask_out_bees()

    # big.create_background_masked_median_mode()
    big.process_rolling_backgrounds(sampling_rate=1)


# test_bg_img_gen(config_1)


def test_different_configs(config):
    out_dir = Path(r"D:\Bachelorarbeit\bee_videos\extracted_frames_ival_30sec")

    # in_dir = Path(r"D:\Bachelorarbeit\bee_videos\cam-0")

    big = bg.BackgroundImageGenerator(
        source_path=out_dir, output_path=out_dir, config=config
    )

    big.mask_out_bees()
    big.process_rolling_backgrounds(sampling_rate=1)


test_different_configs(config_6b)
