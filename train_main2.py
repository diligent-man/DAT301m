import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import pandas as pd
import tensorflow as tf
from global_env import GlobalENV
from Trainer import Trainer


def get_train_val_generator_from_dataframe(train_df: pd.DataFrame,
                                           val_df: pd.DataFrame,
                                           root_dataset: str,
                                           batch_size: int,
                                           seed: int
                                           ) -> tf.keras.preprocessing.image.Iterator:
    train_loader = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=180,
        fill_mode="wrap",
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255
    ).flow_from_dataframe(
        dataframe=train_df,
        directory=root_dataset,
        x_col="img", y_col="class",
        target_size=(224, 224),
        color_mode="rgb",
        classes={"level_1":0., "level_2":1., "level_3":2., "level_4":3., "level_5":4.},
        class_mode="sparse",
        batch_size=batch_size,
        validate_filenames=False,
        seed=seed
    )

    val_loader = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    ).flow_from_dataframe(
        dataframe=val_df,
        directory=root_dataset,
        x_col="img", y_col="class",
        target_size=(224, 224),
        color_mode="rgb",
        classes={"level_1": 0., "level_2": 1., "level_3": 2., "level_4": 3., "level_5": 4.},
        class_mode="sparse",
        batch_size=batch_size,
        validate_filenames=True,
        seed=seed
    )
    return train_loader, val_loader


def main() -> None:
    train_loader, val_loader = get_train_val_generator_from_dataframe(
        train_df=pd.read_csv(GlobalENV.TRAIN_ANNOTATION),
        val_df=pd.read_csv(GlobalENV.VAL_ANNOTATION),
        root_dataset=GlobalENV.ROOT_DATASET,
        batch_size=GlobalENV.BATCH_SIZE,
        seed=GlobalENV.SEED
    )

    trainer = Trainer(GlobalENV.MODEL_NAME,
                      GlobalENV.NUM_CLASSES,
                      train_loader,
                      GlobalENV.INPUT_SHAPE,
                      GlobalENV.LR,
                      GlobalENV.T_MULT,
                      GlobalENV.EPOCH_TO_WARM_RESTART,
                      GlobalENV.SEED,
                      GlobalENV.PRETRAINED,
                      True
                      )
    history = trainer.train(train_loader, val_loader,
                            GlobalENV.EPOCHS,
                            GlobalENV.WORKERS,
                            GlobalENV.MAX_QUEUE_SIZE,
                            GlobalENV.USE_MULTIPROCESSING,
                            GlobalENV.APPLY_CHECKPOINTING,
                            GlobalENV.APPLY_EARLY_STOPPING,
                            GlobalENV.APPLY_TENSORBOARD_CALLBACK
                            )
    return None

if __name__ == '__main__':
    main()