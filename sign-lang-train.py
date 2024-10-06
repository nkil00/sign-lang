import sys
import os

from argparse import ArgumentParser

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the directory to sys.path
sys.path.append(current_dir)

from train.train_sign_lang import TrainSignLang
from models.cnn_models import ConvSignLangNN_7, ConvSignLangNN_4, ConvSignLangNN_4_, ConvSignLangNN_5_, ConvSignLangNN_4_4, ConvSignLangNN_4_KAGG

from torch.nn import CrossEntropyLoss

import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))

LABEL_PATH = os.path.join(current_dir, "data", "sign_lang_train", "labels.csv") 
IMG_DIR = os.path.join(current_dir, "data", "sign_lang_train")
MODEL_DIR = os.path.join(current_dir, "eval", "model-prm", "grids")
INFO_DIR = os.path.join(current_dir, "eval", "info", "grids")

EPOCHS = 15
BATCH_SIZE = 32
def it_info(epoch, bs, lr, th, au):
    print("Epochs           :", epoch)
    print("BatchSize        :", bs)
    print("Learning Rate    :", lr)
    print("Threshold        :", th)
    print("Data Augmentation:", au)

def grid_lr(grid_params: dict, label_df: pd.DataFrame, device = "cpu", sample_ratio: float = 1.0,
            neurons=None, vocal=False, dataset: str = "kaggle"):
    print(f"-- SAMPLE RATIO: {sample_ratio} --")


    lrs = grid_params["lr"]
    batch_size = grid_params["batch_size"]
    epochs = grid_params["epochs"]
    thresholds = grid_params["thresholds"]
    augment = grid_params["augment"]

    approx_total_models = len(lrs)*len(batch_size)*len(epochs)*len(thresholds)*len(augment) * 2 
    models_done = 0
    model_names = ["4"]
    for epoch in epochs:
        for bs in batch_size:
            for lr in lrs:
                for th in thresholds:
                    for au in augment:
                        for mo in model_names:
                            # print the current hyperparameters
                            it_info(epoch, bs, lr, th, au)

                            tsm = TrainSignLang(epochs=epoch,
                                                lr=lr,
                                                train_set_size=0.8,
                                                batch_size=bs,
                                                device=device,
                                                dataset=dataset)
                            tsm.init_data(image_dir=IMG_DIR, 
                                          label_df=label_df,
                                          sample_ratio=sample_ratio,
                                          threshold=th,
                                          augment_data=au)

                            if dataset == "kaggle":
                                model = ConvSignLangNN_4_KAGG()
                            elif neurons == None:
                                model = ConvSignLangNN_4()
                            else: # elif mo == "4":
                                model = ConvSignLangNN_4_(conv1_in=neurons["c1_in"][0],
                                                           conv2_in=neurons["c2_in"][0],
                                                           conv3_in=neurons["c3_in"][0],
                                                           #conv4_in=neurons["c4_in"][0],
                                                           first_dim=neurons["l1"][0],
                                                           second_dim=neurons["l2"][0],
                                                           third_dim=neurons["l3"][0],
                                                           )
                            lossf = CrossEntropyLoss()
                            tsm.init_model(model=model,
                                           loss_fn=lossf,
                                           optim="Adam")
                            tsm.train(vocal=vocal)
                            tsm.evaluate(vocal=vocal)
                            tsm.save_model(MODEL_DIR)
                            tsm.save_info(INFO_DIR)

                            # get overview over how many models have been trained yet
                            models_done += 1
                            print(f"> {models_done}/{approx_total_models} done Training! (approx..)")
            

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", dest="device")
    parser.add_argument("-r", "--sample_ratio", dest="sample_ratio")
    parser.add_argument("-v", "--vocal", dest="vocal", action="store_true")
    parser.add_argument("-s", "--dataset", dest="dataset", default="uibk")

    args = parser.parse_args()

    device = "cpu"
    if args.device: device = args.device

    sample_ratio = 1
    if args.sample_ratio: sample_ratio = float(args.sample_ratio)

    ds = args.dataset

    vocal = args.vocal

    grid_params = {
        "lr":         [0.001],
        "batch_size": [32],
        "epochs":     [30],
        "thresholds": [300],
        "augment" :   [True]
    }

    neurons = {"c1_in": [3],
               "c2_in": [6],
               "c3_in": [12],
               "c4_in": [18],
               "l1":    [512],
               "l2":    [256],
               "l3":    [128],
               "l4":    [64]
               }
    label_df = pd.DataFrame()
    if ds == "uibk":
        label_df = pd.read_csv(LABEL_PATH)
    elif ds == "kaggle":
        label_df = pd.read_csv("./data/kag_sign_lang/sign_mnist_train.csv")
        label_df.loc[label_df["label"] > 8, "label"] -= 1       # label_df = label_df["label"] - 1

    grid_lr(grid_params=grid_params,
            label_df=label_df, 
            device=device, 
            sample_ratio=sample_ratio, 
            neurons=neurons, 
            vocal=vocal,
            dataset=ds)
