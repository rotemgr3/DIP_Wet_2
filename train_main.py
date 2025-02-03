import logging
import os
import sys
import warnings

from train_and_test import test_model, train_model


def create_config():
    config = {
        "batch_size": 128,
        "gpu": 4,
        # "architecture": "FC",
        "architecture": "UNet",
        "hidden_dims": [64, 128, 256, 512],
        # "hidden_dims": [1000, 800, 1000],
        "epochs": 100,
        "learning_rate": 0.001,
        "dataset_dir": "/home/priel.hazan/.cache/kagglehub/datasets/rajat95gupta/smartphone-image-denoising-dataset/versions/1/SIDD_Small_sRGB_Only",
        "crop_size": 128,
        # "output_dir": "./q2_models/exp2",
        "output_dir": "./q3_models/exp2",
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
    sys.stdout = open(log_file, "a")
    sys.stderr = open(log_file, "a")
    warnings.simplefilter("always", Warning)
    logging.captureWarnings(True)


def main():
    config = create_config()
    setup_logging(config["output_dir"])
    train_model(config)
    test_model(config)


if __name__ == "__main__":
    main()
    print("main end")
