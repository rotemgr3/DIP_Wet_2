import os
import random

import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SIDD_Dataset(Dataset):
    def __init__(self, dataset_dir: str, mode: str, crop_size: int = 128):
        self.dataset_dir = dataset_dir
        # make dataset dir abs path cause may contain ../ and things like this:
        self.dataset_dir = os.path.abspath(self.dataset_dir)
        self.crop_size = crop_size
        self.data_dir = os.path.join(dataset_dir, "Data")
        self.data_dir = os.path.abspath(self.data_dir)
        self.mode = mode
        print(f"data_dir: {self.data_dir}")
        # Load scene instances from Scene_Instances.txt
        with open(os.path.join(dataset_dir, "Scene_Instances.txt"), "r") as f:
            self.scene_instances = [line.strip() for line in f]

        # Split the dataset into train, validation, and test sets
        train_scenes, val_scenes, test_scenes = self.split_dataset()

        # Use the appropriate subset based on the mode
        if mode == "train":
            self.scene_instances = train_scenes
        elif mode == "val":
            self.scene_instances = val_scenes
        elif mode == "test":
            self.scene_instances = test_scenes
        else:
            raise ValueError(
                "Invalid mode. Must be one of 'train', 'val', or 'test'.")

        # Precompute index mapping
        self.index_map = self._precompute_index_map()
        # Cache images to avoid reading from disk
        self.image_cache = {}
        # Prefetch images
        self._prefetch()

    def __len__(self):
        return len(self.index_map)

    def _prefetch(self):
        self.scence_cache = {}
        for instance in tqdm(self.scene_instances, desc="Prefetching images"):
            instance_dir = os.path.join(self.data_dir, instance)
            noisy_path = next(
                (
                    os.path.join(instance_dir, f)
                    for f in os.listdir(instance_dir)
                    if f.startswith("NOISY_")
                ),
                None,
            )
            if noisy_path:
                noisy = cv2.imread(noisy_path)
            clean_path = next(
                (
                    os.path.join(instance_dir, f)
                    for f in os.listdir(instance_dir)
                    if f.startswith("GT_")
                ),
                None,
            )
            if clean_path:
                clean = cv2.imread(clean_path)
            self.scence_cache[instance] = (noisy, clean)

    def __getitem__(self, idx):
        if idx in self.image_cache:
            return self.image_cache[idx]
        instance_idx, block_idx = self.index_map[idx]
        instance_dir = self.scene_instances[instance_idx]
        if self.mode == "test" or self.mode == "val":
            imgs = self._get_fixed_block(instance_dir, block_idx)
        else:
            imgs = self._get_block(instance_dir, block_idx)
        self.image_cache[idx] = imgs
        return imgs

    def _get_block(self, instance_dir, block_idx):
        noisy, clean = self.scence_cache[instance_dir]

        # Calculate block position
        h, w, _ = noisy.shape
        crop_size = self.crop_size
        num_blocks_x = w // crop_size
        x = (block_idx % num_blocks_x) * crop_size
        y = (block_idx // num_blocks_x) * crop_size

        # Extract block
        noisy_block = noisy[y: y + crop_size, x: x + crop_size, :]
        clean_block = clean[y: y + crop_size, x: x + crop_size, :]

        # Convert to tensor
        return self._to_tensor(
            noisy_block
        ), self._to_tensor(clean_block)

    def _to_tensor(self, image):
        return torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1)

    def _get_fixed_block(self, instance_dir, block_idx):
        noisy, clean = self.scence_cache[instance_dir]

        # Determine fixed block positions
        h, w, _ = noisy.shape
        crop_size = self.crop_size
        positions = [
            (0, 0),  # Top-left
            (0, w - crop_size),  # Top-right
            (h - crop_size, 0),  # Bottom-left
            (h - crop_size, w - crop_size),  # Bottom-right
        ]
        y, x = positions[block_idx]

        # Extract block
        noisy_block = noisy[y: y + crop_size, x: x + crop_size, :]
        clean_block = clean[y: y + crop_size, x: x + crop_size, :]

        # Convert to tensor
        return self._to_tensor(
            noisy_block
        ), self._to_tensor(clean_block)

    def _precompute_index_map(self):
        index_map = []
        for instance_idx, instance in enumerate(self.scene_instances):
            instance_dir = os.path.join(self.data_dir, instance)
            noisy_path = next(
                (
                    os.path.join(instance_dir, f)
                    for f in os.listdir(instance_dir)
                    if f.startswith("NOISY_")
                ),
                None,
            )
            if noisy_path:
                noisy = cv2.imread(noisy_path, cv2.IMREAD_COLOR)
                h, w, _ = noisy.shape
                num_blocks = (h // self.crop_size) * (w // self.crop_size)
                if self.mode == "test" or self.mode == "val":
                    # Only 4 blocks per image in test mode
                    index_map.extend(
                        [(instance_idx, block_idx) for block_idx in range(4)]
                    )
                else:
                    index_map.extend(
                        [(instance_idx, block_idx)
                         for block_idx in range(num_blocks)]
                    )
        return index_map

    def split_dataset(self, num_images_test=25):
        # Split dataset into train, validation, and test sets
        random.seed(42)
        random.shuffle(self.scene_instances)
        total_images = len(self.scene_instances)

        # Split data (control train/val/test split)
        num_test = num_images_test
        num_val = num_images_test
        num_train = total_images - num_test - num_val

        train_scenes = self.scene_instances[:num_train]
        val_scenes = self.scene_instances[num_train: num_train + num_val]
        test_scenes = self.scene_instances[num_train + num_val:]

        return train_scenes, val_scenes, test_scenes
