import os
import gc
import shutil
import itertools
import numpy as np
from tqdm import tqdm
from HED import get_HED
from typing import List, Tuple
from matplotlib import pyplot as plt
from PIL.ImageOps import exif_transpose


import torch
import torchvision
from torch.utils.data import DataLoader, StackDataset
from torchvision.transforms import v2, Compose, InterpolationMode
from utils import img_folder_datasets, get_imgs_save_path, get_pair_dataloader, save_imgs


plt.switch_backend("tkagg")
torch.manual_seed(12345)
torch.set_grad_enabled(False)
torch.backends.cudnn.enabled = True
####################################################################################################


def detect_edge(root: str ,
                save_path_root: str,
                algorithm: str = "hed",
                device: str = "cpu",
                batch_size: int = 1,
                crop_rate: float = .1,
                invert_color: bool = False,
                save_origin_along: bool = False,
                input_shape: Tuple[int] = (480, 480)
                ) -> None:
    def _get_preprocessing_transform(input_shape: Tuple[int] = (480, 480),
                                     invert_color: bool = False
                                     ) -> Compose:
        transform_lst = [
            v2.Lambda(lambda img: exif_transpose(img)),
            v2.Resize(input_shape, InterpolationMode.BICUBIC, antialias=True),
            v2.RandomAutocontrast(p=1.),
            v2.RandomAdjustSharpness(p=1., sharpness_factor=2.),
            v2.GaussianBlur(kernel_size=3, sigma=1.)
        ]

        if invert_color:
            transform_lst += [v2.RandomInvert(p=1.)]

        transform_lst += [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=False),
        ]
        return Compose(transform_lst)

    def _canny_detector():
        pass

    def _HED_detector(imgs: torch.Tensor, device: str = "cuda") -> torch.Tensor:
        # Create model if not exist in local var
        if "model" not in locals():
            model = get_HED(device=device).eval()

        preds = model(imgs.to(device))
        preds = torch.squeeze(preds, dim=1).cpu()  # Model's return: B,1,H,W --sqeeze--> B,H,W
        preds = (torch.squeeze(preds) if batch_size == 1 else preds) * 255.  # H,W if N==1
        preds = v2.CenterCrop(size=[int(size * (1 - crop_rate)) for size in input_shape])(preds)  # crop 20% border to reduce noise edges
        return preds


    available_algorithms = ["hed", "canny"]
    assert algorithm in available_algorithms, f"Your selected algorithm is unavailable"

    for dataset, depth in img_folder_datasets(root, _get_preprocessing_transform(input_shape, invert_color)):
        imgs_save_path: List[str] = get_imgs_save_path(dataset, depth, save_path_root)
        dataloader: DataLoader = DataLoader(dataset, batch_size, False, num_workers=4, drop_last=False)

        # Used for saving image
        cached_preds: np.ndarray = None  # BHW
        cached_origins: np.ndarray = None  # B3HW

        for (imgs, labels) in tqdm(dataloader, total=len(dataloader), desc="Detecting"):
            if algorithm == "hed":
                preds = _HED_detector(imgs, device).type(torch.uint8).numpy()
                preds = np.where(preds > 0, 255, preds)
            elif algorithm == "canny":
                pass

            cached_preds = preds if cached_preds is None else np.vstack((cached_preds, preds))

            # cache original imgs along
            if save_origin_along:
                imgs = imgs.type(torch.uint8).numpy()
                cached_origins = imgs if cached_origins is None else np.vstack((cached_origins, imgs))

        # Save cached imgs
        if save_origin_along:
            save_imgs(cached_preds, save_path_root, imgs_save_path, cached_origins, pred_mode="L")
        else:
            save_imgs(cached_preds, save_path_root, imgs_save_path, pred_mode="L")

        # Reset cuda
        torch.cuda.empty_cache()
        gc.collect()
    return None


def merge_detected_edge(merge_roots: List[str],
                        save_path_root: str,
                        batch_size: int = 1,
                        delete_merge_roots: bool = True,
                        ) -> None:
    def _get_preprocessing() -> Compose:
        return Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, False)
        ])

    for dataloaders, save_path in get_pair_dataloader(merge_roots, save_path_root, batch_size, _get_preprocessing()):
        aggregated_imgs: torch.Tensor = None

        for dataloader in dataloaders:
            cached_imgs: torch.Tensor = None

            for imgs, labels in tqdm(dataloader, total=len(dataloader), desc="Processing"):
                cached_imgs = imgs if cached_imgs is None else torch.vstack((cached_imgs, imgs))

            # BCHW -> BHWC
            cached_imgs = cached_imgs.type(torch.uint8).movedim(1, -1)
            aggregated_imgs = cached_imgs if aggregated_imgs is None else torch.add(aggregated_imgs, cached_imgs)
        save_imgs(aggregated_imgs.numpy(), save_path_root, save_path)

    if delete_merge_roots:
        for merge_root in merge_roots:
            shutil.rmtree(merge_root)
    return None


def crop_image_by_edge(orig_img_path: str,
                       edge_img_path: str,
                       save_path_root: str,
                       batch_size: int = 1,
                       delete_merge_roots: bool = True
                       ) -> None:
    def _orig_img_preprocessing() -> Compose:
        return Compose([
            v2.Lambda(lambda img: exif_transpose(img)),
            v2.Resize((1200, 1200), InterpolationMode.BICUBIC, antialias=True),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, False),
        ])

    def _edge_img_preprocessing() -> Compose:
        return Compose([
            v2.Grayscale(),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, False)
        ])

    def _compute_rotation_angle(edge_img: torch.Tensor) -> int:
        ROI = torch.argwhere(edge_img > 0)

        try:
            v = torch.max(ROI, axis=0)[0] - torch.min(ROI, axis=0)[0]
            v =v.roll(1)  # (y, x) -> (x, y)
            i = torch.tensor([1, 0])

            angle = (v @ i) / (torch.sqrt(v@v) * torch.sqrt(i@i))
            angle = torch.arccos(angle)
            angle = (angle * 180 / torch.pi).type(torch.int)
            angle = torch.tensor(90)
            # print(angle)
        except:
            angle = torch.tensor(0)
        return angle.item()

    def _rotate_img(img, angle, interpolation=InterpolationMode.BILINEAR) -> np.ndarray:
        img = torchvision.transforms.v2.functional.rotate(img, angle, interpolation)
        return img.detach().cpu().numpy()

    for (orig_dataset, depth), (edge_dataset, _) in zip(img_folder_datasets(orig_img_path, _orig_img_preprocessing()),
                                                        img_folder_datasets(edge_img_path, _edge_img_preprocessing())):
        save_paths = get_imgs_save_path(orig_dataset, depth, save_path_root)
        print(orig_dataset, edge_dataset)
        dataloader = DataLoader(StackDataset(orig_dataset, edge_dataset), batch_size=batch_size, shuffle=False, drop_last=False)

        aggregated_orig_imgs: np.ndarray = None
        aggregated_edge_imgs: np.ndarray = None

        for (orig_imgs, _), (edge_imgs, _) in tqdm(dataloader, total=len(dataloader)):
            angles: List[int] = list(map(_compute_rotation_angle, edge_imgs.squeeze(dim=1)))  # (B,H,W)
            # print(angles)
            rotated_orig_imgs = np.array(list(map(_rotate_img, orig_imgs.to("cuda"), angles)))  # B,3,H,W
            rotated_edge_imgs = np.array(list(map(_rotate_img, edge_imgs.to("cuda"), angles)))  # B,1,H,W

            # aggregated_orig_imgs = rotated_orig_imgs if aggregated_orig_imgs is None else np.vstack((aggregated_orig_imgs, rotated_orig_imgs))
            # aggregated_edge_imgs = aggregated_edge_imgs if aggregated_edge_imgs is None else np.vstack((aggregated_edge_imgs, rotated_edge_imgs))

            aggregated_orig_imgs = orig_imgs.numpy() if aggregated_orig_imgs is None else np.vstack((aggregated_orig_imgs, orig_imgs.numpy()))
        aggregated_orig_imgs = np.transpose(aggregated_orig_imgs.astype(np.uint8), (0, 2, 3, 1))
        save_imgs(aggregated_orig_imgs, save_path_root, save_paths)
    return None


def main() -> None:
    root = os.path.join(os.getenv("HOME"), "Downloads/Dataset/seam_puckering")
    # for root, save_path_root, invert_color in zip((root, root),
    #                                               (f"{os.getenv('HOME')}/Downloads/hed_ouput", f"{os.getenv('HOME')}/Downloads/invert_hed_ouput"),
    #                                               (False, True)):
    #     detect_edge(root, save_path_root,
    #                 invert_color=invert_color,
    #                 save_origin_along=False,
    #                 batch_size=104, crop_rate=.2,
    #                 device="cuda")
    #####################################################################
    edge_img_root = os.path.join(os.getenv("HOME"), "Downloads/merge_output")
    merge_roots = [
        os.path.join(os.getenv("HOME"), "Downloads", "hed_ouput"),
        os.path.join(os.getenv("HOME"), "Downloads", "invert_hed_ouput")
    ]
    # merge_detected_edge(merge_roots, edge_img_root, batch_size=9999, delete_merge_roots=False)
    #####################################################################
    cropped_img_root = os.path.join(os.getenv("HOME"), "Downloads/crop_output")
    # crop_image_by_edge(root, edge_img_root, cropped_img_root, batch_size=999)
    # TODO: CHeck rotation angle
    return None


if __name__ == '__main__':
    main()