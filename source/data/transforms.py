import albumentations as A
from typing import List, Tuple


def get_transforms(
    image_size: Tuple[int, int],
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    train: bool = True,
):
    if train:
        return A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=mean, std=std),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=mean, std=std),
            ]
        )
