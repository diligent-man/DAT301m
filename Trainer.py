import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import keras_cv
import tensorflow as tf
from typing import Tuple, List
from tensorflow.keras.applications import (
    VGG16, ResNet50, EfficientNetB0, MobileNetV2
)

__all__ = ['Trainer']


class Trainer:
    __available_models = {
        "VGG16": VGG16,
        "ResNet50": ResNet50,
        "EfficientNetB0": EfficientNetB0,
        "MobileNetV2": MobileNetV2,
    }

    def __init__(self,
                 model_name: str,
                 num_classes: int,
                 train_loader: tf.data.Dataset,
                 input_shape: Tuple[int, int, int],
                 lr: float,
                 t_mult: int,
                 epoch_to_warm_restart: int,
                 seed: int,
                 pretrained: bool = False,
                 get_summary: bool = True
                 ) -> None:
        self.__seed: int = seed
        self.__model_name: str = model_name
        self.__model = self.__create_model(model_name,
                                           num_classes,
                                           train_loader,
                                           input_shape,
                                           lr, t_mult,
                                           epoch_to_warm_restart,
                                           seed,
                                           pretrained,
                                           get_summary
                                           )

    @property
    def model(self):
        return self.__model

    def __create_model(self,
                       model_name: str,
                       num_classes: int,
                       train_loader: tf.data.Dataset,
                       input_shape: Tuple[int],
                       lr: float, t_mult: int,
                       epoch_to_warm_restart: int,
                       seed: int = 12345,
                       pretrained: bool = False,
                       get_summary: bool = True,
                       ) -> tf.keras.Sequential:
        def _get_augmentations(input_shape: Tuple[int], seed) -> tf.keras.Sequential:
            # Augmentations for seam puckering dataset
            return tf.keras.Sequential([
                tf.keras.layers.RandomZoom(height_factor=(-.3, .3), width_factor=(-.3, .3), fill_mode='reflect', interpolation='nearest', seed=seed),
                tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed),
                tf.keras.layers.RandomRotation(factor=(-1, 1), fill_mode="reflect", interpolation="nearest",seed=seed),
                keras_cv.layers.RandomSharpness(factor=.01, value_range=(0, 255), seed=seed),

                tf.keras.layers.CenterCrop(*input_shape[:2]),
                keras_cv.layers.AutoContrast(value_range=(0, 255)),
                tf.keras.layers.Rescaling(scale=1./255)], name="preprocessing_and_augmentation")
                # tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)], name="preprocessing_and_augmentation")

        def _adapt_classifier(model_name: str, num_classes: int) -> tf.keras.Sequential:
            classifier: tf.keras.Sequential = None

            if model_name == "VGG16":
                classifier = tf.keras.Sequential(layers=[
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(4096, activation="relu"), tf.keras.layers.Dropout(rate=.5),
                    tf.keras.layers.Dense(4096, activation="relu"), tf.keras.layers.Dropout(rate=.5),
                    tf.keras.layers.Dense(units=num_classes, activation="sigmoid" if num_classes == 1 else "softmax")],
                    name="classifier"
                )

            elif model_name == "ResNet50":
                classifier = tf.keras.Sequential(layers=[
                    tf.keras.layers.GlobalAvgPool2D(),
                    tf.keras.layers.Dense(units=1024, activation="relu"), tf.keras.layers.Dropout(rate=.5),
                    tf.keras.layers.Dense(units=num_classes, activation="sigmoid" if num_classes == 1 else "softmax")],
                    name="classifier"
                )

            elif model_name == "EfficientNetB0":
                classifier = tf.keras.Sequential(layers=[
                    tf.keras.layers.GlobalAvgPool2D(),
                    tf.keras.layers.Dropout(rate=.5),
                    tf.keras.layers.Dense(units=1000, activation="relu"), tf.keras.layers.Dropout(rate=.5),
                    tf.keras.layers.Dense(units=num_classes, activation="sigmoid" if num_classes == 1 else "softmax")],
                    name="classifier"
                )

            elif model_name == "MobileNetV2":
                classifier = tf.keras.Sequential(layers=[
                    tf.keras.layers.GlobalAvgPool2D(),
                    tf.keras.layers.Dropout(rate=.3),
                    tf.keras.layers.Dense(units=1024, activation="relu"), tf.keras.layers.Dropout(rate=.5),
                    tf.keras.layers.Dense(units=512, activation="relu"), tf.keras.layers.Dropout(rate=.5),
                    tf.keras.layers.Dense(units=num_classes, activation="sigmoid" if num_classes == 1 else "softmax")],
                    name="classifier"
                )
            return classifier

        def _compile_model(model: tf.keras.models.Sequential,
                           train_loader: tf.data.Dataset,
                           lr: float, t_mult: int,
                           epoch_to_warm_restart: int
                           ) -> None:
            # Use Cosine Annealing With Warm Restart as a lr scheduler
            T_0 = len(train_loader) * epoch_to_warm_restart
            scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=lr,
                                                                          first_decay_steps=T_0,
                                                                          t_mul=t_mult
                                                                          )

            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                          optimizer=tf.keras.optimizers.Adam(learning_rate=scheduler, use_ema=True, amsgrad=True),
                          metrics=["acc",
                                   tf.keras.metrics.F1Score(average="micro", name="f1")]
                          )
            return None

        if pretrained:
            # Created model will not include head/ classifier
            # Define head/ classifier separately
            model: tf.keras.Model = self.__available_models[model_name](input_shape=input_shape, include_top=False)
            # Freeze backbone
            for layer in model.layers:
                layer.trainable = False

            # Add predefined head
            model = tf.keras.Sequential([model, _adapt_classifier(model_name, num_classes)])
        else:
            model: tf.keras.Model = self.__available_models[model_name](input_shape=input_shape, include_top=True,
                                                                        classes=num_classes, weights=None)

        model = tf.keras.Sequential([_get_augmentations(input_shape, seed), model])
        _compile_model(model, train_loader, lr, t_mult, epoch_to_warm_restart)

        if get_summary:
            model.build(input_shape=(None, *(800, 800, 3)))
            model.summary(expand_nested=True,
                          show_trainable=True
                          )
        return model

    ##########3########################################################################################

    def train(self,
              train_loader: tf.data.Dataset,
              val_loader: tf.data.Dataset,
              epochs: int,
              workers: int,
              max_queue_size: int,
              use_multiprocessing: bool,
              apply_checkpointing: bool,
              apply_early_stopping: bool,
              apply_tensorboard: bool
              ) -> tf.keras.callbacks.History:
        def _get_callbacks(apply_checkpointing: bool,
                           apply_early_stopping: bool,
                           apply_tensorboard: bool
                           ) -> tf.keras.callbacks.Callback:
            callbacks: List[tf.keras.callbacks.Callback] = []
            if apply_checkpointing:
                checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", self.__model_name)
                callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                                    monitor="val_loss",
                                                                    verbose=0,
                                                                    save_best_only=True,
                                                                    save_freq="epoch"
                                                                    ))

            if apply_early_stopping:
                callbacks.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                                                  patience=5,
                                                                  verbose=1,
                                                                  restore_best_weights=True)
                                 )
            if apply_tensorboard:
                log_dir = os.path.join(os.getcwd(), "tensorboard_logs", self.__model_name)
                callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
            return callbacks
        history: tf.keras.callbacks.History = self.__model.fit(x=train_loader,
                                                               validation_data=val_loader,
                                                               epochs=epochs,
                                                               shuffle=True,
                                                               workers=workers,
                                                               max_queue_size=max_queue_size,
                                                               use_multiprocessing=use_multiprocessing,
                                                               callbacks=_get_callbacks(apply_checkpointing,
                                                                                        apply_early_stopping,
                                                                                        apply_tensorboard)
                                                               )
        return history
