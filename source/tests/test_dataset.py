import unittest
import yaml
import numpy as np
import tempfile
from data.dataset_factory import create_datasets
from data.dataset import COCODataset
from source.configs.logger import setup_logger

test_logger = setup_logger("tests.test_dataset", "test.log", topic="TEST")

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_logger.info("Setting up datasets for testing...")
        cls.datasets = create_datasets("source/configs/dataset_config.yaml")
        test_logger.info("Datasets setup completed.")

    # Test for train dataset
    def test_train_dataset_exists(self):
        train_dataset = self.datasets.get("train")
        self.assertIsNotNone(train_dataset, "Train dataset should not be None")
        self.assertGreater(len(train_dataset), 0, "Train dataset should not be empty")

    def test_train_dataset_image(self):
        train_dataset = self.datasets.get("train")
        image, _ = train_dataset[0]
        self.assertIsNotNone(image, "Image in train dataset should not be None")
        self.assertIsInstance(image, np.ndarray, "Image should be a NumPy array")
        self.assertEqual(image.ndim, 3, "Image should be 3-dimensional (HWC)")

    def test_train_dataset_mask(self):
        train_dataset = self.datasets.get("train")
        _, masks = train_dataset[0]
        self.assertIsInstance(
            masks, np.ndarray, "Masks in train dataset should be a NumPy array"
        )
        self.assertTrue(masks.ndim >= 2, "Masks should have at least 2 dimensions")
        self.assertTrue(
            np.issubdtype(masks.dtype, np.integer), "Masks should be of integer type"
        )

    # Test for validation dataset
    def test_val_dataset_exists(self):
        val_dataset = self.datasets.get("val")
        if val_dataset:
            self.assertGreater(
                len(val_dataset), 0, "Validation dataset should not be empty"
            )

    def test_val_dataset_image(self):
        val_dataset = self.datasets.get("val")
        if val_dataset:
            image, _ = val_dataset[0]
            self.assertIsNotNone(
                image, "Image in validation dataset should not be None"
            )
            self.assertIsInstance(image, np.ndarray, "Image should be a NumPy array")
            self.assertEqual(image.ndim, 3, "Image should be 3-dimensional (HWC)")

    def test_val_dataset_mask(self):
        val_dataset = self.datasets.get("val")
        if val_dataset:
            _, masks = val_dataset[0]
            self.assertIsInstance(
                masks, np.ndarray, "Masks in validation dataset should be a NumPy array"
            )
            self.assertTrue(masks.ndim >= 2, "Masks should have at least 2 dimensions")
            self.assertTrue(
                np.issubdtype(masks.dtype, np.integer),
                "Masks should be of integer type",
            )

    # Test for test dataset
    def test_test_dataset_exists(self):
        test_dataset = self.datasets.get("test")
        if test_dataset:
            self.assertGreater(len(test_dataset), 0, "Test dataset should not be empty")

    def test_test_dataset_image(self):
        test_dataset = self.datasets.get("test")
        if test_dataset:
            image, _ = test_dataset[0]
            self.assertIsNotNone(image, "Image in test dataset should not be None")
            self.assertIsInstance(image, np.ndarray, "Image should be a NumPy array")
            self.assertEqual(image.ndim, 3, "Image should be 3-dimensional (HWC)")

    def test_test_dataset_mask(self):
        test_dataset = self.datasets.get("test")
        if test_dataset:
            _, masks = test_dataset[0]
            self.assertIsInstance(
                masks, np.ndarray, "Masks in test dataset should be a NumPy array"
            )
            self.assertTrue(masks.ndim >= 2, "Masks should have at least 2 dimensions")
            self.assertTrue(
                np.issubdtype(masks.dtype, np.integer),
                "Masks should be of integer type",
            )

    # Edge case tests
    def test_empty_dataset(self):
        empty_dataset = self.datasets.get("empty")
        if empty_dataset:
            self.assertEqual(
                len(empty_dataset), 0, "Empty dataset should have zero length"
            )

    def test_invalid_dataset_config(self):
        invalid_config = {"dataset": {"missing_required_key": "value"}}
        with tempfile.NamedTemporaryFile(
            "w", delete=False, suffix=".yaml"
        ) as temp_file:
            yaml.dump(invalid_config, temp_file)
            temp_file_path = temp_file.name

        with self.assertRaises(ValueError):
            create_datasets(temp_file_path)

    def test_missing_annotation_file(self):
        with self.assertRaises(FileNotFoundError):
            _ = COCODataset(data_dir="non_existent_directory", class_ids=[1, 2, 3])

    def test_image_without_annotations(self):
        train_dataset = self.datasets.get("train")
        if train_dataset:
            for idx in range(len(train_dataset)):
                _, masks = train_dataset[idx]
                self.assertIsInstance(
                    masks, np.ndarray, "Masks should be a NumPy array"
                )
                self.assertTrue(
                    masks.ndim >= 2, "Masks should have at least 2 dimensions"
                )

    def test_overlapping_annotations(self):
        train_dataset = self.datasets.get("train")
        if train_dataset:
            for idx in range(len(train_dataset)):
                _, masks = train_dataset[idx]
                self.assertTrue(
                    np.max(masks) <= len(train_dataset.class_ids),
                    "Mask values should not exceed the number of class IDs",
                )


if __name__ == "__main__":
    unittest.main()
