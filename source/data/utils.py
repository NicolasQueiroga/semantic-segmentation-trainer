import os
import cv2
import torch
import numpy as np
import albumentations as A
from typing import List, Tuple
from torch.utils.data import DataLoader
from source.configs.logger import setup_logger

data_logger = setup_logger("data.utils", "data.log", topic="DATA")


def load_image(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def validate_dataset_config(dataset_config: dict) -> None:
    data_logger.info("Validating dataset configuration...")
    required_keys = ["format", "class_ids", "train_dir"]
    for key in required_keys:
        if key not in dataset_config:
            data_logger.error(f"Missing required key '{key}' in dataset configuration.")
            raise ValueError(f"Missing required key '{key}' in dataset configuration.")
    if not os.path.exists(dataset_config["train_dir"]):
        data_logger.error(
            f"Training directory '{dataset_config['train_dir']}' does not exist."
        )
        raise FileNotFoundError(
            f"Training directory '{dataset_config['train_dir']}' does not exist."
        )
    if "val_dir" in dataset_config and not os.path.exists(dataset_config["val_dir"]):
        data_logger.error(
            f"Validation directory '{dataset_config['val_dir']}' does not exist."
        )
        raise FileNotFoundError(
            f"Validation directory '{dataset_config['val_dir']}' does not exist."
        )
    if "test_dir" in dataset_config and not os.path.exists(dataset_config["test_dir"]):
        data_logger.error(
            f"Test directory '{dataset_config['test_dir']}' does not exist."
        )
        raise FileNotFoundError(
            f"Test directory '{dataset_config['test_dir']}' does not exist."
        )
    data_logger.info("Dataset configuration validated successfully.")


def calculate_dataset_statistics(
    dataset: List,
    image_size: Tuple[int, int],
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    data_logger.info("Calculating dataset statistics...")
    transform = A.Compose([A.Resize(image_size[0], image_size[1])])
    dataset.transform = transform

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    n_pixels = 0
    channel_sum = torch.zeros(3, device=device)
    channel_sum_squared = torch.zeros(3, device=device)

    for batch in dataloader:
        images, _ = batch
        images = images.to(device).float() / 255.0
        images = images.permute(0, 3, 1, 2)

        n_pixels += images.numel() // images.shape[1]
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sum_squared += (images**2).sum(dim=[0, 2, 3])

    mean = channel_sum / n_pixels
    std = torch.sqrt(channel_sum_squared / n_pixels - mean**2)

    mean = mean.cpu().numpy()
    std = std.cpu().numpy()

    data_logger.info(f"Dataset statistics calculated: mean={mean}, std={std}")
    return mean, std
