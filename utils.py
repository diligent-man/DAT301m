from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import numpy as np
import tensorflow as tf

from typing import List
from matplotlib import pyplot as plt
from keras.src.engine.keras_tensor import KerasTensor

plt.switch_backend("tkagg")


__all__ = ["batch_visualization"]


def multiple_batch_visualization(dataloader: tf.data.Dataset | np.ndarray | tf.keras.preprocessing.image.Iterator,
                                 batch_size: int,
                                 upscale: bool = True) -> None:
    assert batch_size == pow(np.sqrt(batch_size), 2), "Batch size must be divisible by its square root"

    for images, labels in dataloader:
        if upscale: images *= 255  # Rescale from [0,1] to [0,255]
        if not isinstance(images, np.ndarray): images = images.numpy()
        if not isinstance(labels, np.ndarray): labels = labels.numpy()

        images = images.astype(np.uint8)
        labels = labels.astype(np.uint8)

        plt.figure(figsize=(40, 40))
        for i in range(batch_size):
            ax = plt.subplot(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)), i + 1)
            ax.imshow(images[i])
            ax.set_title(f"""Class: {labels[i].squeeze()}""")
            ax.axis("off")
        plt.show()
    return None


def batch_visualization(images: tf.data.Dataset | np.ndarray,
                        labels: tf.data.Dataset | np.ndarray | None = None,
                        upscale: bool = True) -> None:
    batch_size = images.shape[0]
    assert batch_size == pow(np.sqrt(batch_size), 2), "Batch size must be divisible by its square root"

    if upscale: images *= 255  # Rescale from [0,1] to [0,255]
    if not isinstance(images, np.ndarray): images = images.numpy()
    if not isinstance(labels, np.ndarray) and labels is not None:
        labels = labels.numpy()
        labels = labels.astype("uint8")

    images = images.astype("uint8")

    plt.figure(figsize=(40, 40))
    for i in range(images.shape[0]):
        ax = plt.subplot(int(np.sqrt(batch_size)), int(np.sqrt(batch_size)), i + 1)
        ax.imshow(images[i], cmap="gray")
        if labels is not None: ax.set_title(f"""Class: {labels[i].squeeze()}""")
        ax.axis("off")
    plt.show()
    return None


def feature_map_visualization(model: tf.keras.Model, train_loader: tf.data.Dataset) -> None:
    # train_loader formater: imgs, labels
    layer_outputs: List[KerasTensor] = [layer.output for layer in model.layers]
    layer_names: List[str] = [layer.name for layer in model.layers]

    # Define model including conv/ maxpool layers
    model_of_layers = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
    feature_maps = model_of_layers.predict(next(iter(train_loader))[0])

    for counter, (layer_name, feature_map) in enumerate(zip(layer_names, feature_maps)):
        if len(feature_map.shape) == 4:
            # -------------------------------------------
            # Just do this for the conv / maxpool layers, not the fully-connected layers
            # -------------------------------------------
            n_features = feature_map.shape[-1]  # number of features in the feature map
            size = feature_map.shape[1]  # feature map shape (1, size, size, n_features)

            # Tile the images in this matrix
            display_grid = np.zeros((size, size * n_features))

            # -------------------------------------------------
            # Post-process the feature to be visually palatable
            # -------------------------------------------------
            for i in range(n_features):
                x = feature_map[0, :, :, i]  # take 1st img in batch
                x = (x - x.mean()) / x.std()  # z norm
                x = x * 64 + 128  # scale up for visualizing feature maps in DL
                x = np.clip(x, 0, 255).astype('uint8')
                display_grid[:, i * size: (i + 1) * size] = x  # Tile each filter into a horizontal grid

            # -----------------
            # Display the grid
            # -----------------
            scale = 40. / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='Greys')

        # Just visualize first three layers
        if counter > 3:
            break
    plt.show()
    return None
