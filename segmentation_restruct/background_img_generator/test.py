import segmentation_restruct.background_img_generator.background_img_generator as bg
from pathlib import Path
from segmentation_restruct.utils import BgImageGenConfig


def test_bg_img_gen():

    config = BgImageGenConfig(num_median_images=20)
    out_dir = Path("C:/Users/sturmd/Desktop/Bachelorarbeit/bee_data/20240529/cam-2")
    big = bg.BackgroundImageGenerator(
        source_path=out_dir, output_path=out_dir, config=config
    )

    big.mask_out_bees()

    # big.create_background_masked_median_mode()
    big.process_rolling_backgrounds(sampling_rate=5)


test_bg_img_gen()
