import os
import torch

import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from reusable_generator import ReusableGenerator
from typing import Dict, Set, List, Tuple, Optional, Callable, Iterator

from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.datasets import ImageFolder

plt.switch_backend("tkagg")

__all__ = ["batch_visualization", "get_image_folder_dataset", "get_pair_dataloader","get_imgs_save_path", "save_imgs"]


################################################################################################
def _get_path_by_depth(path: str, depth: int, end_depth: int = None) -> str:
    reversed_path: List[str] = list(reversed(path.split(os.sep)))

    if end_depth is None:
        depth_by_path = f"{os.sep}".join(reversed_path[:depth][::-1])
    else:
        depth_by_path = f"{os.sep}".join(reversed_path[end_depth: depth][::-1])
    return depth_by_path


#################################################################################################


def batch_visualization(images: np.ndarray,
                        labels: np.ndarray | None = None,
                        cmap: str = None,
                        title: str = None,
                        upscale: bool = True
                        ) -> None:
    batch_size = images.shape[0]
    assert batch_size == pow(np.sqrt(batch_size), 2), "Batch size must be divisible by its square root"

    if upscale: images *= 255  # Rescale from [0,1] to [0,255]
    if not isinstance(images, np.ndarray): images = images.numpy()
    if not isinstance(labels, np.ndarray) and labels is not None:
        labels = labels.numpy()
        labels = labels.astype("uint8")

    images = images.astype("uint8")
    plt.figure(figsize=(40, 40))
    if title is not None:
        plt.suptitle(t=title)

    for i in range(images.shape[0]):
        ax = plt.subplot(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)), i + 1)
        ax.imshow(images[i], cmap=cmap)
        if labels is not None: ax.set_title(f"""Class: {labels[i].squeeze()}""")
        ax.axis("off")

    plt.show()
    return None


def get_image_folder_dataset(root: str,
                             transform: Optional[Callable] = None,
                             target_transform: Optional[Callable] = None
                             ) -> Iterator[ImageFolder]:
    depth = 0
    flag = True
    min_depth = 0
    num_classes: int = 0
    for dirpath, dirname, filename in sorted(os.walk(root)):
        if len(filename) > 0 and filename[0].endswith("jpg") and flag is True:
            if flag:
                min_depth = depth
                flag = False
            dataset: ImageFolder = ImageFolder(os.path.join(root, _get_path_by_depth(dirpath, depth, depth - 1)),
                                               transform, target_transform)
            yield dataset, depth
        else:
            depth += 1

        # Reset when meet root
        if _get_path_by_depth(dirpath, depth + (2 * num_classes), 1) == root:
            flag = True
            depth = min_depth
            num_classes = 1

        # Retrieve num_class for reset condition
        try:
            if os.listdir(os.path.join(dirpath, dirname[0]))[0].endswith(".jpg"):
                num_classes = len(dirname)
        except:
            continue



@ReusableGenerator
def get_pair_dataloader(roots: List[str],
                        save_path_root: str,
                        batch_size: int,
                        transform: Compose = None,
                        target_transform: Compose = None
                        ) -> Tuple[List[DataLoader], List[str]]:
    pair_dataloaders: Dict[int: Dict[str: List[DataLoader], str: List[str]]] = {}
    for root in roots:
        i = 0
        for dataset, depth in get_image_folder_dataset(root, transform, target_transform):
            dataloader: DataLoader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=False)
            if i not in pair_dataloaders.keys():
                pair_dataloaders[i] = {"dataloader": [dataloader], "save_path": get_imgs_save_path(dataset, depth, save_path_root)}
            else:
                pair_dataloaders[i]["dataloader"].append(dataloader)
            i += 1

    for i in range(len(pair_dataloaders)):
        yield pair_dataloaders[i]["dataloader"], pair_dataloaders[i]["save_path"]


def get_imgs_save_path(dataset: ImageFolder,
                       depth: int,
                       save_path_root: str = None
                       ) -> List[str]:
    imgs_save_path: List[str] = []
    unique_save_paths: Set[str] = set()  # used for making save dir

    for img_path, _ in dataset.imgs:
        no_root_path: str = _get_path_by_depth(img_path, depth + 1)
        imgs_save_path.append(no_root_path)
        unique_save_paths.add(f"{os.sep}".join(no_root_path.split(os.sep)[:-1]))

    if save_path_root is not None:
        for unique_save_path in unique_save_paths:
            os.makedirs(os.path.join(save_path_root, unique_save_path), exist_ok=True)
    return imgs_save_path


def save_imgs(preds: np.ndarray,
              save_path_root: str,
              imgs_save_path: List[str],
              origins: np.ndarray = None
              ) -> None:
    if origins is not None:
        origins = np.transpose(origins, axes=(0, 2, 3, 1))  # B,C,H,W -> B,H,W,C
        zipped_obj = zip(preds, imgs_save_path, origins)

        for pred, fp, origin in tqdm(zipped_obj, total=len(imgs_save_path), desc="Saving imgs", colour="cyan"):
            fp = os.path.join(save_path_root, fp)
            origin_fp = fp.split(".")[0] + "_origin.jpg"

            Image.fromarray(pred).save(fp=fp)
            Image.fromarray(origin).save(fp=origin_fp)
    else:
        zipped_obj = zip(preds, imgs_save_path)

        for pred, fp in tqdm(zipped_obj, total=len(imgs_save_path), desc="Saving imgs", colour="cyan"):
            fp = os.path.join(save_path_root, fp)
            Image.fromarray(pred).save(fp=fp)
    return None
