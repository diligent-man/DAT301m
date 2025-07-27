import os
from pathlib import Path

__all__ = ["GlobalENV"]


class GlobalENV:
    # Dataset Path
    ENV = False
    ROOT_DATASET = Path(r"/home/trong/Downloads/Dataset/seam_puckering/filtered_crop_seam_puckering") if ENV else\
                   Path(r"/tf/Dataset/seam_puckering/filtered_crop_seam_puckering")

    TRAIN_ANNOTATION = os.path.join(ROOT_DATASET, "train_annotation.csv")
    VAL_ANNOTATION = os.path.join(ROOT_DATASET, "val_annotation.csv")

    # Model
    NUM_CLASSES = 5
    PRETRAINED = True
    # MObilenetV2, ResNet50, VGG16, EfficientNetB0
    MODEL_NAME = "EfficientNetB0"

    # Training setups
    WORKERS = 8
    EPOCHS = 100
    BATCH_SIZE = 16
    PREFETCH_SIZE = 32
    MAX_QUEUE_SIZE = 20
    USE_MULTIPROCESSING = True
    INPUT_SHAPE = (224, 224, 3)

    # Scheduler
    LR = 1e-4
    T_MULT = 2
    EPOCH_TO_WARM_RESTART = 20

    # Tensorboard
    APPLY_CHECKPOINTING = True
    APPLY_EARLY_STOPPING = True
    APPLY_TENSORBOARD = True

    # Misc
    SEED = 12345
