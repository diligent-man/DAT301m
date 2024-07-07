import os

import torch
from PIL import Image
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
plt.switch_backend("tkagg")


def main1() -> None:
    # img = "/home/trong/Downloads/merge/black_s_black (31)/level_1/black_s_black_l1_15.jpg"
    img = "/home/trong/Downloads/merge/black_s_black (31)/level_1/black_s_black_l1_17.jpg"
    img = cv.imread(img, cv.IMREAD_GRAYSCALE)

    # img = cv.boxFilter(img, -1, (7, 7))
    # img = torch.tensor(img)
    # fft = np.fft.fftshift(np.fft.fft2(img))
    # fft = torch.fft.fftshift(torch.fft.fft2(img))

    # window_size = 30
    # y_center, x_center = img.shape[0] // 2, img.shape[1] // 2

    # fft[y_center - window_size: y_center + window_size + 1, x_center - window_size: x_center + window_size + 1] = 0 + 0j
    # plt.imshow(torch.log(torch.abs(fft)) + 1, cmap="gray")
    # plt.imshow(np.log(np.abs(fft) + 1), cmap="gray")
    # plt.show()

    # img_back = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(fft))).type(torch.uint8).numpy()
    # img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(fft))).astype(np.uint8).copy()
    # plt.imshow(img_back, cmap="gray")
    # plt.show()

    lines = cv.HoughLines(img, 1, np.pi / 180, 70, None, 0, 0)
    print(len(lines))
    if lines is not None:
        for i in range(len(lines)):
            rho, theta = lines[i][0]

            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a * rho
            y0 = b * rho

            # rho * cos - 1000 * sin
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))

            # print(rho, theta, a, b, x0, y0, pt1, pt2)
            img = cv.line(img, pt1, pt2, 177, 1, cv.LINE_AA)

    plt.imshow(img, cmap="gray")
    plt.show()

    oy = np.array([0, 1])
    v = np.array(pt2) - np.array(pt1)
    cos_theta = (v@oy) / (np.sqrt(v@v) + 1)
    theta = np.arccos(cos_theta) / np.pi * 180

    def rotation(image, angleInDegrees):
        h, w = image.shape[:2]
        img_c = (w / 2, h / 2)

        rot = cv.getRotationMatrix2D(img_c, angleInDegrees, 1)

        rad = np.radians(angleInDegrees)
        sin = np.sin(rad)
        cos = np.cos(rad)
        b_w = int((h * abs(sin)) + (w * abs(cos)))
        b_h = int((h * abs(cos)) + (w * abs(sin)))

        rot[0, 2] += ((b_w / 2) - img_c[0])
        rot[1, 2] += ((b_h / 2) - img_c[1])

        outImg = cv.warpAffine(image, rot, (b_w, b_h), flags=cv.INTER_LINEAR)
        return outImg

    img = rotation(img, -theta)
    plt.imshow(img, cmap="gray")
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




