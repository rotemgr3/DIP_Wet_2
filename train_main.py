import logging
import os
import sys
import warnings

from train_and_test import test_model, train_model


def create_config():
    config = {
        "batch_size": 32,
        # "architecture": "UNet",
        "architecture": "FC",
        # Input and output dims are 128*128*3 = 49152 so need to create some bottlenech
        "hidden_dims": [1000, 800, 1000],
        # "hidden_dims": [64, 128, 256, 512],
        "epochs": 100,
        "learning_rate": 0.001,
        # the dir that contains the data dir
        "dataset_dir": "/Users/rotem.green/.cache/kagglehub/datasets/rajat95gupta/smartphone-image-denoising-dataset/versions/1/SIDD_Small_sRGB_Only",
        "crop_size": 128,
        "output_dir": "./q2_models/exp2",
    }
    return config


def setup_logging(output_dir):
    log_file = os.path.join(output_dir, "train_main.log")
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(
            log_file), logging.StreamHandler(sys.stdout)],
    )
    # If file exists, flush it and start over
    open(log_file, "w").close()

    # If want to write to file instead of stdout
    # sys.stdout = open(log_file, "a")
    # sys.stderr = open(log_file, "a")
    warnings.simplefilter("always", Warning)
    logging.captureWarnings(True)


def main():
    config = create_config()
    # setup_logging(config["output_dir"])
    train_model(config)
    test_model(config)


if __name__ == "__main__":
    main()
    print("main end")
