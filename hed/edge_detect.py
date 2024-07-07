import os
import gc
import shutil
import kornia
import cv2 as cv
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
def fft_transform(root: str,
                  save_path_root: str,
                  batch_size: int = 1,
                  window_size: int = 20,
                  device: str = "cpu",
                  # input_shape: Tuple[int] = (512, 512),
                  input_shape: Tuple[int] = (1024, 1024),
                  ) -> None:
    def _get_preprocessing_transform(input_shape: Tuple[int]) -> Compose:
        transform_lst = [
            v2.Lambda(lambda img: exif_transpose(img)),
            v2.RandomAutocontrast(p=1.),
            # Cropping four sides
            v2.Lambda(lambda img: img.crop(box=(120, 250, img.size[0] - 100, img.size[1] - 80))),  # box = (left, top, right, bottom)
            v2.Resize(input_shape, InterpolationMode.NEAREST_EXACT, antialias=True),
            # Convert grayscale
            v2.Lambda(lambda img: cv.cvtColor(np.array(img.convert('RGB')), cv.COLOR_RGB2GRAY)),
            # Filter 3x3
            v2.Lambda(lambda img: cv.boxFilter(img, -1, ksize=(3, 3))),
            # HW -> 1HW
            v2.Lambda(lambda img: np.expand_dims(img, axis=0)),
            # Conver to tensor
            v2.Lambda(lambda img: torch.from_numpy(img)),
            v2.ToDtype(dtype=torch.float32, scale=False),
        ]
        return Compose(transform_lst)

    y_center, x_center = input_shape[0] // 2, input_shape[1] // 2

    for dataset, depth in img_folder_datasets(root, _get_preprocessing_transform(input_shape)):
        imgs_save_path: List[str] = get_imgs_save_path(dataset, depth, save_path_root)
        dataloader: DataLoader = DataLoader(dataset, batch_size, False, num_workers=4, drop_last=False)

        for (imgs, labels) in tqdm(dataloader, total=len(dataloader), desc="Processing"):
            imgs: torch.Tensor = imgs.to(device)

            # B1HW -> BHW
            if len(imgs.shape) == 4:
                imgs = imgs.squeeze(dim=1)
            # FFT
            imgs = torch.fft.fftshift(torch.fft.fft2(imgs))
            # Zero out center
            imgs[:, y_center - window_size: y_center + window_size + 1, x_center - window_size: x_center + window_size + 1] = 0 + 0j
            # IFFT -> Image back
            imgs: np.ndarray = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(imgs))).type(torch.uint8).numpy()
        save_imgs(imgs, save_path_root, imgs_save_path, pred_mode="L")
    return None


def detect_edge(root: str,
                save_path_root: str,
                algorithm: str = "hed",
                device: str = "cpu",
                batch_size: int = 1,
                crop_rate: float = .1,
                invert_color: bool = False,
                save_origin_along: bool = False,
                input_shape: Tuple[int] = (600, 600)
                ) -> None:
    def _get_preprocessing(input_shape: Tuple[int], invert_color: bool = False) -> Compose:
        transform_lst = [
            v2.Lambda(lambda img: exif_transpose(img)),
            # Ignore img has size of (512, 512) because fft has the same size
            v2.Lambda(lambda img: img.crop(box=(120, 250, img.size[0]-110, img.size[1]-90)) if img.size != (512, 512) else img),  # box = (left, top, right, bottom)
            v2.Resize(input_shape, InterpolationMode.NEAREST, antialias=True),
            v2.RandomAutocontrast(p=1.),
            v2.GaussianBlur(kernel_size=3, sigma=3.),
            v2.RandomAdjustSharpness(p=1., sharpness_factor=7.)
        ]

        if invert_color:
            transform_lst += [v2.RandomInvert(p=1.)]

        transform_lst += [
            v2.ToImage(),
            v2.ToDtype(dtype=torch.float32, scale=False),
        ]
        return Compose(transform_lst)

    def post_processing(imgs: np.ndarray) -> np.ndarray:
        # Binarizing
        imgs = np.where(imgs > 0, 255, imgs)

        # Remove noise with morphological transformation
        # imgs = np.apply_along_axis(cv.morphologyEx, axis=0, arr=imgs, **{"op": cv.MORPH_DILATE, "kernel": kernel, "iterations": 5}).squeeze()
        kernel = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(3, 3), anchor=(-1, 1))

        imgs = list(map(lambda img: cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=5), imgs))
        imgs = list(map(lambda img: cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=3), imgs))
        imgs = list(map(lambda img: cv.morphologyEx(img, cv.MORPH_DILATE, kernel, iterations=2), imgs))

        imgs = np.array(imgs).squeeze()  # B1HW -> BHW
        return imgs

    def _HED_detector(imgs: torch.Tensor, crop_rate: float = 0., device: str = "cuda") -> torch.Tensor:
        # Create model if not exist in local var
        if "model" not in locals():
            model = get_HED(device=device).eval()

        preds = model(imgs)
        preds = torch.squeeze(preds)  # Model's return: B,1,H,W --squeeze--> B,H,W
        preds = torch.squeeze(preds) * 255.  # H,W if N==1
        preds = v2.CenterCrop(size=[int(size * (1 - crop_rate)) for size in input_shape])(preds)  # crop 20% border to reduce noise edges
        return preds.cpu()

    available_algorithms = ["canny", "hed"]
    assert algorithm in available_algorithms, f"Your selected algorithm is unavailable"

    counter = 0
    for dataset, depth in img_folder_datasets(root, _get_preprocessing(input_shape, invert_color)):
        imgs_save_path: List[str] = get_imgs_save_path(dataset, depth, save_path_root)
        dataloader: DataLoader = DataLoader(dataset, batch_size, False, num_workers=4, drop_last=False)

        # Used for saving image
        cached_preds: np.ndarray = None  # BHW
        cached_origins: np.ndarray = None  # B3HW

        for (imgs, labels) in tqdm(dataloader, total=len(dataloader), desc="Detecting"):
            imgs = imgs.to(device)

            if algorithm == "hed":
                preds: torch.Tensor = _HED_detector(imgs, crop_rate, device)
                preds: np.ndarray = preds.byte().numpy()
                preds: np.ndarray = post_processing(preds)
            elif algorithm == "canny":
                # canny return Tuple(magnitude, edges) in the shape of B1HW
                mag, preds = kornia.filters.canny(imgs / 255.)  # BCHW -> B1HW
                preds = kornia.tensor_to_image(preds.byte())  # B1HW -> BHW
                preds *= 255

            cached_preds = preds if cached_preds is None else np.vstack((cached_preds, preds))

            # cache original imgs along
            if save_origin_along:
                imgs = imgs.byte().cpu().numpy()
                cached_origins = imgs if cached_origins is None else np.vstack((cached_origins, imgs))

        # Save cached imgs
        if save_origin_along:
            save_imgs(cached_preds, save_path_root, imgs_save_path, cached_origins, pred_mode="L")
        else:
            save_imgs(cached_preds, save_path_root, imgs_save_path, pred_mode="L")

        # Reset cuda
        torch.cuda.empty_cache()
        gc.collect()

        # counter += 1
        # if counter == 5: break
    return None


def merge_detected_edge(roots: List[str],
                        save_path_root: str,
                        batch_size: int = 1,
                        delete_merge_roots: bool = True,
                        ) -> None:
    transform: Compose = Compose([v2.PILToTensor(),
                                  v2.ToDtype(torch.float32, False)
                                  ])

    for dataloaders, save_path in get_pair_dataloader(roots, save_path_root, batch_size, transform):
        aggregated_imgs: torch.Tensor = None

        for dataloader in dataloaders:
            cached_imgs: torch.Tensor = None

            for imgs, labels in tqdm(dataloader, total=len(dataloader), desc="Merging"):
                cached_imgs = imgs if cached_imgs is None else torch.vstack((cached_imgs, imgs))

            cached_imgs = cached_imgs.type(torch.uint8).movedim(1, -1)  # BCHW -> BHWC
            aggregated_imgs = cached_imgs if aggregated_imgs is None else torch.add(aggregated_imgs, cached_imgs)
        save_imgs(aggregated_imgs.numpy(), save_path_root, save_path)

    if delete_merge_roots:
        for merge_root in roots:
            shutil.rmtree(merge_root)
    return None


def detect_line(orig_img_path: str,
                       edge_img_path: str,
                       save_path_root: str,
                       batch_size: int = 1,
                       delete_merge_roots: bool = True
                       ) -> None:
    def _orig_img_preprocessing() -> Compose:
        return Compose([
            v2.Lambda(lambda img: exif_transpose(img)),
            v2.Resize((800, 800), InterpolationMode.BICUBIC, antialias=True),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, False),
        ])

    def _edge_img_preprocessing() -> Compose:
        return Compose([
            v2.Grayscale(),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, False),
        ])

    # def _compute_rotation_angle(edge_img: torch.Tensor) -> int:
    #     ROI = torch.argwhere(edge_img > 0)
    #
    #     try:
    #         v = torch.max(ROI, axis=0)[0] - torch.min(ROI, axis=0)[0]
    #         v = v.roll(1)  # (y, x) -> (x, y)
    #         oy = torch.tensor([0, 1])
    #
    #         angle = (v @ oy) / (torch.sqrt(v @ v) * torch.sqrt(oy @ oy))
    #         angle = torch.arccos(angle)
    #         angle = torch.round(angle / torch.pi * 180)
    #         angle = (-angle).type(torch.int)
    #     except:
    #         angle = torch.tensor(0)
    #     return angle.item()

    # def _rotate_img(img, angle, interpolation=InterpolationMode.BILINEAR) -> np.ndarray:
    #     img = torchvision.transforms.v2.functional.rotate(img, angle, interpolation)
    #     return img.detach().cpu().numpy()

    hough_transform = Compose([
        v2.Lambda(lambda img: img.numpy())
    ])

    for (orig_dataset, depth), (edge_dataset, _) in zip(img_folder_datasets(orig_img_path, _orig_img_preprocessing()),
                                                        img_folder_datasets(edge_img_path, _edge_img_preprocessing())):
        save_paths = get_imgs_save_path(orig_dataset, depth, save_path_root)
        dataloader = DataLoader(StackDataset(orig_dataset, edge_dataset), batch_size=batch_size, shuffle=False, drop_last=False)

        cached_orig_imgs: np.ndarray = None
        cached_edge_imgs: np.ndarray = None

        for (orig_imgs, _), (edge_imgs, _) in tqdm(dataloader, total=len(dataloader)):
            edge_imgs = hough_transform(edge_imgs)
            orig_imgs = orig_imgs.byte().numpy()


            # angles: List[int] = list(map(_compute_rotation_angle, edge_imgs.squeeze(dim=1)))  # (B,H,W)
            # print(angles)
            # rotated_orig_imgs = np.array(list(map(_rotate_img, orig_imgs.to("cuda"), angles)))  # B,3,H,W
            # rotated_edge_imgs = np.array(list(map(_rotate_img, edge_imgs.to("cuda"), angles)))  # B,1,H,W
            #
            cached_orig_imgs = orig_imgs if cached_orig_imgs is None else np.vstack((cached_orig_imgs, orig_imgs))
            cached_edge_imgs = edge_imgs if cached_edge_imgs is None else np.vstack((cached_edge_imgs, edge_imgs))

        # Save rotated imgs
        cached_orig_imgs = np.transpose(cached_orig_imgs, (0, 2, 3, 1))  # BCHW -> BHWC
        save_imgs(cached_orig_imgs, save_path_root, save_paths)

        # Reset cuda
        torch.cuda.empty_cache()
        gc.collect()
    return None


def main() -> None:
    home = os.getenv("HOME")
    root = os.path.join(home, "Downloads/Dataset/seam_puckering")
    fft_root = os.path.join(home, "Downloads", "fft")
    merge_root = os.path.join(home, "Downloads", "merge")
    rotated_root = os.path.join(home, "Downloads", "crop")
    edge_detected_roots = [
        os.path.join(os.getenv("HOME"), "Downloads", "hed"),
        os.path.join(os.getenv("HOME"), "Downloads", "invert_hed"),
        # os.path.join(os.getenv("HOME"), "Downloads", "fft_hed"),
    ]

    # for root, save_path_root, invert_color in zip((root, root), edge_detected_roots, (False, True)):
    #     detect_edge(root, save_path_root,
    #                 algorithm="hed",
    #                 invert_color=invert_color,
    #                 save_origin_along=False,
    #                 batch_size=62,
    #                 crop_rate=0.,
    #                 device="cuda")
    # merge_detected_edge(edge_detected_roots, merge_root, batch_size=9999, delete_merge_roots=True)
    detect_line(root, merge_root, rotated_root, batch_size=9999)


    # fft_transform(root=root, save_path_root=fft_root, batch_size=9999)
    # TODO
    # https://stackoverflow.com/questions/72061208/how-to-detect-an-object-that-blends-with-the-background

    return None


if __name__ == '__main__':
    main()