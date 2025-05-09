import os
import numpy as np
import torch
import warnings
from torch.utils.data import Dataset
from typing import List
from pycocotools.coco import COCO
from functools import lru_cache

from .utils import load_image


class COCODataset(Dataset):
    def __init__(self, data_dir: str, class_ids: List[int]):
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module="pycocotools"
        )
        self.coco = COCO(os.path.join(data_dir, "_annotations.coco.json"))
        self.image_ids = self.coco.getImgIds()
        self.img_dir = data_dir
        self.class_ids = class_ids
        self.transform = None

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_id = self.image_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        image = load_image(img_path)
        mask = self._load_masks(img_id)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]

        return image, mask

    @lru_cache(maxsize=1000)
    def _load_masks(self, img_id: int) -> np.ndarray:
        annotation_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(annotation_ids)

        img_info = self.coco.imgs[img_id]
        height, width = img_info["height"], img_info["width"]
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        for annotation in annotations:
            category_id = annotation["category_id"]
            if category_id in self.class_ids:
                mask = self.coco.annToMask(annotation)
                class_index = self.class_ids.index(category_id) + 1
                combined_mask[mask > 0] = class_index

        return combined_mask
