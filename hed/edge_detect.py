import os
import gc
import shutil
import itertools
import numpy as np

from tqdm import tqdm
from HED import get_HED
from matplotlib import pyplot as plt
from typing import List, Tuple, Dict


import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2, Compose, InterpolationMode
from utils import get_image_folder_dataset, get_imgs_save_path, get_pair_dataloader, save_imgs


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
            v2.Resize(size=input_shape, interpolation=InterpolationMode.BICUBIC, antialias=True),
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

    def _HED_detector(imgs: torch.Tensor) -> torch.Tensor:
        # Create model if not exist in local var
        if "model" not in locals():
            model = get_HED(device=device).eval()

        preds = model(imgs.to(device))
        preds = torch.squeeze(preds, dim=1).cpu()  # Model's return: N,1,H,W --sqeeze--> N,H,W
        preds = (torch.squeeze(preds) if batch_size == 1 else preds) * 255.  # H,W if N==1
        preds = v2.CenterCrop(size=[int(size * (1 - crop_rate)) for size in input_shape])(preds)  # crop 20% border to reduce noise edges
        return preds


    available_algorithms = ["hed", "canny"]
    assert algorithm in available_algorithms, f"Your selected algorithm is unavailable"

    for dataset, depth in get_image_folder_dataset(root, _get_preprocessing_transform(input_shape, invert_color)):
        imgs_save_path: List[str] = get_imgs_save_path(dataset, depth, save_path_root)
        dataloader: DataLoader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, shuffle=False, drop_last=False)

        # Used for saving image
        cached_preds: np.ndarray = None
        cached_origins: np.ndarray = None
        for (imgs, labels) in tqdm(dataloader, total=len(dataloader), desc="Detecting"):
            if algorithm == "hed":
                preds = _HED_detector(imgs).type(torch.uint8).numpy()
                preds = np.where(preds > 0, 255, preds)
            elif algorithm == "canny":
                pass

            cached_preds = preds if cached_preds is None else np.vstack((cached_preds, preds))

            # cache original imgs along
            if save_origin_along:
                origins = imgs.type(torch.uint8).numpy()
                cached_origins = origins if cached_origins is None else np.vstack((cached_origins, origins))

        # Save cached imgs
        if save_origin_along:
            save_imgs(cached_preds, save_path_root, imgs_save_path, cached_origins)
        else:
            save_imgs(cached_preds, save_path_root, imgs_save_path)

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
            v2.ToDtype(torch.float32),

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
                       edge_img_path,
                       save_path_root: str,
                       batch_size: int = 1,
                       delete_merge_roots: bool = True
                       ) -> None:
    def _get_preprocessing() -> Compose:
        return Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32),

        ])

    pair_dataloaders: Tuple[List[DataLoader], List[str]] = get_pair_dataloader(
        [edge_img_path, orig_img_path],
        save_path_root,
        batch_size,
        _get_preprocessing()
    )

    for (dataloaders, save_path) in pair_dataloaders:
        print(dataloaders)
        # continue
        # for dataloader in dataloaders:
        #     for imgs, labels in tqdm(dataloader, total=len(dataloader), desc=phase):
        #         continue
    # orig_img_dataloader = DataLoader(dataset=get_image_folder_dataset(orig_img_path),
    #                                  batch_size=batch_size,
    #                                  num_workers=4,
    #                                  shuffle=False,
    #                                  drop_last=False
    #                                  )
    # edge_img_dataloader = DataLoader(dataset=get_image_folder_dataset(edge_img_path),
    #                                  batch_size=batch_size,
    #                                  num_workers=4,
    #                                  shuffle=False,
    #                                  drop_last=False
    #                                  )

    # for (orig_imgs, _), (edge_imgs, _) in tqdm(zip(orig_img_dataloader, edge_img_dataloader), total=len(orig_img_dataloader)):
    #     continue




    # # argwhere, max, min
    # img = "/home/trong/Downloads/Dataset/seam_puckering/black_s_black (31)/level_1/black_s_black_l1_17.jpg"
    # # img = "/home/trong/Downloads/Dataset/seam_puckering/black_s_black (31)/level_1/black_s_black_l1_15.jpg"
    #
    # edge_img = "/home/trong/Downloads/merge_output/black_s_black (31)/level_1/black_s_black_l1_17.jpg"
    # # edge_img = "/home/trong/Downloads/merge_output/black_s_black (31)/level_1/black_s_black_l1_15.jpg"
    # ####
    #
    # img = torchvision.io.read_image(img, mode=torchvision.io.ImageReadMode.GRAY).squeeze()
    # edge_img = torchvision.io.read_image(edge_img, mode=torchvision.io.ImageReadMode.GRAY).squeeze()
    #
    #
    # ## find the non-zero min-max coords of canny
    # ROI = torch.argwhere(edge_img > 0)
    # y1, x1 = torch.min(ROI, axis=0)[0].tolist()
    # y2, x2 = torch.max(ROI, axis=0)[0].tolist()
    #
    # # img = edge_img[y1:y2, x1:x2].numpy()
    # # PIL.Image.fromarray(img).save(fp="/home/trong/Downloads/cropped.jpg")
    #
    # v = np.array([x2-x1, y2-y1])
    # i = np.array([1, 0])
    # j = np.array([0, 1])
    #
    # #####
    # angle = 90 - round(np.arccos((v@i)/(np.linalg.norm(v)*np.linalg.norm(i))) * 180 / np.pi)
    # edge_img = torchvision.transforms.v2.functional.rotate(edge_img.unsqueeze(dim=0), angle=angle, interpolation=InterpolationMode.BILINEAR)
    #
    # affine_matrix = np.array([[np.cos(5), -np.sin(5), 0],
    #                             [np.sin(5), np.cos(5), 0],
    #                             [0, 0, 1]])
    # matrix = np.array([[x1, x2], [y1, y2], [0,0]])
    # print(matrix)
    # print(np.matmul(affine_matrix, matrix))
    #
    #
    # ######
    # PIL.Image.fromarray(edge_img.squeeze().numpy()).save(fp="/home/trong/Downloads/cropped.jpg")




    ## find the non-zero min-max coords of canny
    # img = img.numpy()
    # pts = np.argwhere(img > 0)
    # y1, x1 = pts.min(axis=0)
    # y2, x2 = pts.max(axis=0)
    # print(y1, x1)
    # print(y2, x2)
    #
    # import cv2 as cv
    # cropped = img[y1:y2, x1:x2]
    # cv.imwrite("/home/trong/Downloads/cropped.png", cropped)

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
    crop_image_by_edge(root, edge_img_root, cropped_img_root)
    return None


if __name__ == '__main__':
    main()
