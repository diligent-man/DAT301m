import os
from pathlib import Path

__all__ = ["GlobalENV"]


class GlobalENV:
    # Dataset Path
    ENV = True
    ROOT_DATASET = Path(r"/home/trong/Downloads/Dataset/seam_puckering/") if ENV else\
                   Path(r"/tf/Dataset/seam_puckering/")
    TRAIN_ANNOTATION = os.path.join(ROOT_DATASET, "train_annotation.csv")
    VAL_ANNOTATION = os.path.join(ROOT_DATASET, "val_annotation.csv")

    # Model
    NUM_CLASSES = 5
    PRETRAINED = True
    # VGG16, ResNet50, EfficientNetB0, MobileNetV2
    MODEL_NAME = "MobileNetV2"

    # Training setups
    WORKERS = 8
    EPOCHS = 50
    BATCH_SIZE = 16
    PREFETCH_SIZE = 32
    MAX_QUEUE_SIZE = 20
    USE_MULTIPROCESSING = True
    INPUT_SHAPE = (224, 224, 3)

    # Scheduler
    LR = 1e-3
    T_MULT = 1
    EPOCH_TO_WARM_RESTART = 10

    # Tensorboard
    APPLY_CHECKPOINTING = True
    APPLY_EARLY_STOPPING = False
    APPLY_TENSORBOARD = True

    # Misc
    SEED = 12345
