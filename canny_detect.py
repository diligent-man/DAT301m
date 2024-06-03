import os
import cv2 as cv
import numpy as np
import pandas as pd

from tqdm import tqdm
from matplotlib import pyplot as plt

plt.switch_backend("tkagg")


def detect_edge(image: np.ndarray,
                fname: str = None,
                convert_grayscale: bool = True,
                visualize: bool = True,
                save: bool = True
               ) -> np.ndarray:
    if convert_grayscale:
        img = cv.cvtColor(image, code=cv.COLOR_BAYER_RG2GRAY)

    canny = cv.Canny(image, threshold1=255, threshold2=255, apertureSize=3, L2gradient=False)

    if save or visualize:
        assert isinstance(fname, str), "File name must be provided for saving img"

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))
        for obj, name, ax in zip((image, canny), ("original", "canny"), axes):
            ax.imshow(obj, cmap="gray") if obj.ndim == 2 else ax.imshow(obj)
            ax.set_title(name)

        if save: plt.savefig(fname=fname)
        if visualize: plt.show()

        plt.close()  # close window
        plt.clf()  # clear fig
    return canny


def sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]]
                      )
    image = cv.filter2D(image, -1, kernel)
    return image


def get_img_hist(image, path) -> None:
    hist = np.histogram(image.ravel(), bins=256, range=[0, 256])
    plt.plot(hist[0])
    plt.title(path.split(os.sep)[-1].split(".")[0])
    plt.savefig(fname=path)
    plt.close()  # close window
    plt.clf()  # clear fig
    return None


def preprocess(image: np.ndarray) -> np.ndarray:
    image = cv.cvtColor(image, code=cv.COLOR_RGB2GRAY)
    image = cv.createCLAHE(clipLimit=1, tileGridSize=(8, 8)).apply(image)
    # img = cv.GaussianBlur(img, ksize=(3, 3), sigmaY=2, sigmaX=2)
    image = cv.blur(image, ksize=(3, 3))
    image = 255 - image
    return image


def center_crop(image: np.ndarray, new_width=512, new_height=512) -> np.ndarray:
    width, height = image.shape[0], image.shape[1]

    crop_width = new_width if new_width < height else height
    crop_height = new_height if new_height < width else width
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    image = image[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
    return image


def main() -> None:
    # df = pd.read_csv(r"D:\Dataset\seam_puckering\annotation.csv")
    # root_dataset = r"D:\Dataset\seam_puckering"
    #
    # counter = 0
    # for i, row in tqdm(enumerate(df.values), total=len(df.values)) :
    #     img_path = os.path.join(root_dataset, row[0])
    #     img = cv.imread(filename=img_path, flags=cv.IMREAD_COLOR)
    #     img = cv.resize(src=img, dsize=(640, 640), interpolation=cv.INTER_NEAREST_EXACT)
    #     edges = detect_edge(image=preprocess(img),
    #                         fname=os.path.join(os.getcwd(), "canny", f"{counter}.jpg"),
    #                         convert_grayscale=True,
    #                         visualize=False,
    #                         save=True)
    #     edges = center_crop(edges, new_width=224, new_height=224)
    #
    #     plt.imshow(edges, cmap="gray")
    #     plt.show()
        # lines_list = []
        # lines = cv.HoughLinesP(edges,
        #                        rho=2,
        #                        theta=np.pi / 180,
        #                        threshold=10,
        #                        minLineLength=50,
        #                        maxLineGap=10)
        # if lines is not None:
        #     for points in lines:
        #         x1, y1, x2, y2 = points[0]
        #         img = cv.line(img, (x1, y1), (x2, y2), (255, 255, 0), 10)
        #         # Maintain a simples lookup list for points
        #         # lines_list.append([(x1, y1), (x2, y2)])
        #
        #     cv.imwrite(os.path.join(os.getcwd(), "canny", f"{counter}_line.jpg"), img)
        # counter += 1

    # from scipy.io import loadmat
    # from matplotlib import pyplot as plt
    # plt.switch_backend("tkagg")
    #
    # root = "/home/trong/Downloads/Dataset/BSDS500/groundTruth/train"
    # for i in os.listdir("/home/trong/Downloads/Dataset/BSDS500/groundTruth/train"):
    #     print(i)
    #     img = loadmat(os.path.join(root, i))
    #     img = np.array(img["groundTruth"][0][0][0][0][0])
    #     plt.imshow(img, cmap="gray")
    #     plt.title(i)
    #     plt.show()
    return None


if __name__ == '__main__':
    main()