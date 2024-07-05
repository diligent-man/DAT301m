import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import pandas as pd

from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow.python.data.ops.zip_op import _ZipDataset
from tensorflow.python.framework.ops import SymbolicTensor
from tensorflow.python.data.ops.shuffle_op import _ShuffleDataset
from tensorflow.python.data.ops.prefetch_op import _PrefetchDataset
from tensorflow.python.data.ops.batch_op import _ParallelBatchDataset


__all__ = ['get_train_val_loader']
###############################################################################################################


def _read_img(root_dataset: Path, filepath: str, data_type: str, input_shape: Tuple = (224, 224, 3)) -> SymbolicTensor:
    root_dataset: tf.Tensor = tf.strings.as_string(str(root_dataset), name="root_dataset_conversion")
    byte_img: SymbolicTensor = tf.io.read_file(tf.strings.join([root_dataset, filepath], separator="/"))
    img: SymbolicTensor = tf.image.decode_jpeg(byte_img, channels=input_shape[-1])  # convert from bytes to np arr

    if data_type == "train":
        # Min resolution in the seam puckering dataset is (8xx, 8xx)
        # First resize to 512 for batching
        # Model's input shape will be performed after agmentation layers
        img: SymbolicTensor = tf.image.resize(img, size=(800, 800), method="nearest")  # resize to model's input shape
    elif data_type == "val":
        img: SymbolicTensor = tf.image.resize(img, size=input_shape[:2], method="nearest")  # resize to model's input shape
        img: SymbolicTensor = tf.cast(img, tf.float32)
        # img: SymbolicTensor = tf.subtract(tf.divide(img, 127.5), 1)
        img: SymbolicTensor = tf.divide(img, 255.)
    return img


def _convert_label_to_float(label: str) -> SymbolicTensor:
    label: SymbolicTensor = tf.strings.split(label, sep="_")[1]  # "class_1" -> "1"
    label: SymbolicTensor = tf.expand_dims(label, axis=-1)  # ["1"] -> [["1"]]
    label: SymbolicTensor = tf.strings.to_number(input=label, out_type=tf.float32)  # [["1"]] -> [[1.]]
    label: SymbolicTensor = tf.subtract(label, 1)  # [[1.]] -> [[0.]]
    return label
###############################################################################################################


def get_train_val_loader(root_dataset: str,
                         train_annotation: str,
                         val_annotation: str,
                         workers: int,
                         batch_size: int,
                         prefetch_size: int,
                         input_shape: Tuple[int],
                         seed: int
                         ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    dataset_loader = []
    for data_type, annotation in zip(("train", "val"), (train_annotation, val_annotation)):
        df: pd.DataFrame = pd.read_csv(Path(annotation))

        dataset: _ZipDataset = tf.data.Dataset.zip(datasets=(
            tf.data.Dataset.from_tensor_slices(df["img"]).map(
                lambda filepath: _read_img(root_dataset, filepath, data_type, input_shape)
            ),  # imgs
            tf.data.Dataset.from_tensor_slices(df["class"]).map(_convert_label_to_float))  # labels
        )
        dataset: _ShuffleDataset = dataset.shuffle(buffer_size=dataset.cardinality(), seed=seed, reshuffle_each_iteration=True)
        dataset: _ParallelBatchDataset = dataset.batch(batch_size=batch_size, drop_remainder=True, num_parallel_calls=workers, deterministic=True)
        dataset: _PrefetchDataset = dataset.prefetch(buffer_size=prefetch_size)
        dataset_loader.append(dataset)
    return dataset_loader[0], dataset_loader[1]
