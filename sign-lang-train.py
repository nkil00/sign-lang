from train.train_sign_lang import TrainSignLang
from models.cnn_models import ConvSignLangNN_7

from torch.nn import CrossEntropyLoss

import os

LABEL_PATH = os.path.join(".", "data", "sign_lang_train", "labels.csv") 
IMG_DIR = os.path.join(".", "data", "sign_lang_train")
"./models/state_dicts/"
MODEL_DIR = os.path.join(".", "eval", "model-prm", "grids")
INFO_DIR = os.path.join(".", "eval", "info", "grids")

EPOCHS = 15
BATCH_SIZE = 32



def grid_lr(grid_params: dict):
    lrs = grid_params["lr"]
    batch_size = grid_params["batch_size"]
    epochs = grid_params["epochs"]

    for epoch in epochs:
        for bs in batch_size:
            for lr in lrs:
                print("Epochs: ", epoch)
                print("BatchSize: ", bs)
                print("Learning Rate:", lr)
                tsm = TrainSignLang(epochs=epoch,
                                      lr=lr,
                                      train_set_size=0.8,
                                      batch_size=bs)
                tsm.init_data(image_dir=IMG_DIR,
                                labels_path=LABEL_PATH)

                model = ConvSignLangNN_7()
                lossf = CrossEntropyLoss()
                tsm.init_model(model=model,
                               loss_fn=lossf,
                               optim="Adam")
                tsm.train()
                tsm.evaluate()
                tsm.save_model(MODEL_DIR)
                tsm.save_info(INFO_DIR)
    

if __name__ == "__main__":
    lrs = [0.001, 0.005, 0.0001]
    grid_params = {
        "lr": lrs,
        "batch_size": [32, 64, 256],
        "epochs": [10, 15]
    }
    grid_lr(grid_params)
