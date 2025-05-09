import yaml

from source.data.transforms import get_transforms
from source.configs.logger import setup_logger

from .dataset import COCODataset
from .utils import calculate_dataset_statistics, validate_dataset_config

data_logger = setup_logger("data.dataset_factory", "data.log", topic="DATA")


def create_datasets(config_path: str) -> dict[str, COCODataset]:
    data_logger.info(f"Loading dataset configuration from {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    dataset_config = config["dataset"]
    validate_dataset_config(dataset_config)

    dataset_format = dataset_config["format"]
    image_size = dataset_config.get("image_size", None)

    if dataset_format == "coco":
        datasets = {}
        if "train_dir" in dataset_config:
            data_logger.info("Creating training dataset...")
            datasets["train"] = COCODataset(
                data_dir=dataset_config["train_dir"],
                class_ids=dataset_config["class_ids"],
            )
            mean, std = calculate_dataset_statistics(
                datasets["train"], image_size=image_size
            )
            datasets["train"].transform = get_transforms(
                image_size=image_size, mean=mean, std=std, train=True
            )
            data_logger.info("Training dataset created successfully.")

        if "val_dir" in dataset_config:
            data_logger.info("Creating validation dataset...")
            datasets["val"] = COCODataset(
                data_dir=dataset_config["val_dir"],
                class_ids=dataset_config["class_ids"],
            )
            mean, std = calculate_dataset_statistics(
                datasets["val"], image_size=image_size
            )
            datasets["val"].transform = get_transforms(
                image_size=image_size, mean=mean, std=std, train=False
            )
            data_logger.info("Validation dataset created successfully.")

        if "test_dir" in dataset_config:
            data_logger.info("Creating test dataset...")
            datasets["test"] = COCODataset(
                data_dir=dataset_config["test_dir"],
                class_ids=dataset_config["class_ids"],
            )
            mean, std = calculate_dataset_statistics(
                datasets["test"], image_size=image_size
            )
            datasets["test"].transform = get_transforms(
                image_size=image_size, mean=mean, std=std, train=False
            )
            data_logger.info("Test dataset created successfully.")

        return datasets
    else:
        data_logger.error(f"Unsupported dataset format: {dataset_format}")
        raise ValueError(
            f"Unsupported dataset format: {dataset_format}. Supported formats: ['coco']"
        )
