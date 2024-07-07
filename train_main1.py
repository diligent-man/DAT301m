from Trainer import Trainer
from global_env import GlobalENV
from data_preprocessing import get_train_val_loader


def main() -> None:
    train_loader, val_loader = get_train_val_loader(GlobalENV.ROOT_DATASET,
                                                    GlobalENV.TRAIN_ANNOTATION,
                                                    GlobalENV.VAL_ANNOTATION,
                                                    GlobalENV.WORKERS,
                                                    GlobalENV.BATCH_SIZE,
                                                    GlobalENV.PREFETCH_SIZE,
                                                    GlobalENV.INPUT_SHAPE,
                                                    GlobalENV.SEED
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
    history = trainer.train(train_loader,
                            val_loader,
                            GlobalENV.EPOCHS,
                            GlobalENV.WORKERS,
                            GlobalENV.MAX_QUEUE_SIZE,
                            GlobalENV.USE_MULTIPROCESSING,
                            GlobalENV.APPLY_CHECKPOINTING,
                            GlobalENV.APPLY_EARLY_STOPPING,
                            GlobalENV.APPLY_TENSORBOARD
                            )
    return None

if __name__ == '__main__':
    main()