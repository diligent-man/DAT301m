import os

import torch
from PIL import Image
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
plt.switch_backend("tkagg")


def main1() -> None:
    img = "/home/trong/Downloads/fft_hed/black_s_black (31)/level_1/black_s_black_l1_02_origin.jpg"
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)

    img = cv.boxFilter(img, -1, (7, 7))
    img = torch.tensor(img)
    print(img.shape)
    fft = np.fft.fftshift(np.fft.fft2(img))
    # fft = torch.fft.fftshift(torch.fft.fft2(img))

    window_size = 30
    y_center, x_center = img.shape[0] // 2, img.shape[1] // 2

    fft[y_center - window_size: y_center + window_size + 1, x_center - window_size: x_center + window_size + 1] = 0 + 0j
    # plt.imshow(torch.log(torch.abs(fft)) + 1, cmap="gray")
    plt.imshow(np.log(np.abs(fft) + 1), cmap="gray")
    plt.show()

    # img_back = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft))).type(torch.uint8).numpy()
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fft))).astype(np.uint8).copy()
    plt.imshow(img_back, cmap="gray")
    plt.show()

    lines = cv.HoughLines(img_back, 1, np.pi / 180, 10, None, 0, 0)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        img_back = cv.line(img_back, pt1, pt2, 175, 5, cv.LINE_AA)

    plt.imshow(img_back, cmap="gray")
    plt.show()
    return None


def main2() -> None:
    import io

    import requests

    def download_image(url: str, filename: str = "") -> str:
        filename = url.split("/")[-1] if len(filename) == 0 else filename
        # Download
        bytesio = io.BytesIO(requests.get(url).content)
        # Save file
        with open(filename, "wb") as outfile:
            outfile.write(bytesio.getbuffer())

        return filename

    url = "https://ih1.redbubble.net/image.675644909.6235/flat,800x800,075,f.u3.jpg"
    download_image(url, "paranoia_agent.jpg")



    import kornia
    import matplotlib.pyplot as plt
    import torch

    # read the image with Kornia
    img_tensor = kornia.io.load_image("paranoia_agent.jpg", kornia.io.ImageLoadType.RGB32)[None, ...]  # BxCxHxW
    print(img_tensor.shape)
    img_array = kornia.tensor_to_image(img_tensor)
    print(type(img_array))

    plt.axis("off")
    plt.imshow(img_array)
    plt.show()

    # create the operator
    canny = kornia.filters.Canny()
    # blur the image
    # tensor B1HW
    x_magnitude, x_canny = canny(img_tensor)


    # convert back to numpy
    #
    img_magnitude = kornia.tensor_to_image(x_magnitude.byte())
    img_canny = kornia.tensor_to_image(x_canny.byte()) # BHWC
    print(img_magnitude.shape, img_canny.shape)

    # Create the plot
    fig, axs = plt.subplots(1, 3, figsize=(16, 16))
    axs = axs.ravel()

    axs[0].axis("off")
    axs[0].set_title("image source")
    axs[0].imshow(img_array)

    axs[1].axis("off")
    axs[1].set_title("canny magnitude")
    axs[1].imshow(img_magnitude, cmap="Greys")

    axs[2].axis("off")
    axs[2].set_title("canny edges")
    axs[2].imshow(img_canny, cmap="Greys")

    plt.show()

    # plt.imsave("/home/trong/Downloads/demo.png", img_canny, cmap="gray")
    Image.fromarray(img_canny * 255, "L").save("/home/trong/Downloads/demo.jpg")
    return None


if __name__ == '__main__':
    main1()
    # main2()




