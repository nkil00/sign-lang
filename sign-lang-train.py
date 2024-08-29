import sys
import os

from argparse import ArgumentParser

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the directory to sys.path
sys.path.append(current_dir)

from train.train_sign_lang import TrainSignLang
from models.cnn_models import ConvSignLangNN_7, ConvSignLangNN_4

from torch.nn import CrossEntropyLoss

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

LABEL_PATH = os.path.join(current_dir, "data", "sign_lang_train", "labels.csv") 
IMG_DIR = os.path.join(current_dir, "data", "sign_lang_train")
MODEL_DIR = os.path.join(current_dir, "eval", "model-prm", "grids")
INFO_DIR = os.path.join(current_dir, "eval", "info", "grids")

EPOCHS = 15
BATCH_SIZE = 32

def grid_lr(grid_params: dict, df: pd.DataFrame, device="cpu", sample_ratio=1):
    lrs = grid_params["lr"]
    batch_size = grid_params["batch_size"]
    epochs = grid_params["epochs"]
    thresholds = grid_params["thresholds"]
    augment = grid_params["augment"]
    for epoch in epochs:
        for bs in batch_size:
            for lr in lrs:
                for th in thresholds:
                    for au in augment:
                        print("Epochs: ", epoch)
                        print("BatchSize: ", bs)
                        print("Learning Rate:", lr)
                        print("Threshold: ", th)
                        print("Data Augmentation: ", au)
                        tsm = TrainSignLang(epochs=epoch,
                                              lr=lr,
                                              train_set_size=0.8,
                                              batch_size=bs,
                                              device=device)
                        tsm.init_data(image_dir=IMG_DIR, 
                                      label_df=df,
                                      sample_ratio=sample_ratio,
                                      threshold=th,
                                      augment_data=au)

                        model = ConvSignLangNN_4()
                        lossf = CrossEntropyLoss()
                        tsm.init_model(model=model,
                                       loss_fn=lossf,
                                       optim="Adam")
                        tsm.train(vocal=True)
                        tsm.evaluate(vocal=True)
                        tsm.save_model(MODEL_DIR)
                        tsm.save_info(INFO_DIR)
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", dest="device")
    parser.add_argument("-r", "--sample_ratio", dest="sample_ratio")
    args = parser.parse_args()
    # test

    device = "cpu"
    if args.device: device = args.device

    sample_ratio = 1
    if args.sample_ratio: sample_ratio = float(args.sample_ratio)

    grid_params = {
        "lr": [0.001, 0.0001],
        "batch_size": [32],
        "epochs": [1, 10, 15, 20],
        "thresholds" : [-1],
        "augment" : [True, False]
    }
    df = pd.read_csv(LABEL_PATH)
    grid_lr(grid_params, df, device, sample_ratio)
