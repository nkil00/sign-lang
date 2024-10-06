import optuna
from optuna.trial import TrialState

import torch
import torch.nn as nn

from argparse import ArgumentParser
from train.train_sign_lang import TrainSignLang
from models.dynamic_model import SignLangCNN

import pandas as pd

DEVICE = "cpu"
VOCAL = False
SAMPLE_RATIO = 1

IN_FEATS = 128
OUT_FEATS = 36
IN_CHANNEL = 3
BATCH_SIZE = 32

def objective(trial: optuna.Trial):
    print(DEVICE, VOCAL, SAMPLE_RATIO)
    suite = TrainSignLang( 
        epochs = trial.suggest_int("epochs", 5, 20),
        lr = 0.001,
        batch_size = BATCH_SIZE,
        device = DEVICE,

    )
    df = pd.read_csv("./data/sign_lang_train/labels.csv")

    suite.init_data(
        image_dir="./data/sign_lang_train",
        label_df=df,
        sample_ratio=SAMPLE_RATIO,
    )
    # define model
    model = SignLangCNN(
        n_flayers=trial.suggest_int("n_flayers", 1, 4),
        n_clayers=trial.suggest_int("n_clayers", 1, 3),
        hidden_size=trial.suggest_int("hidden_size", 32, 512),
        in_channels=IN_CHANNEL,
        in_features=IN_FEATS,
        out_features=OUT_FEATS,
    )

    # loss function
    lf = torch.nn.CrossEntropyLoss()

    suite.init_model(
        model = model,
        optim="Adam",
        loss_fn=lf
    )

    return suite.train_loop(vocal=VOCAL)

def main():
    global DEVICE, SAMPLE_RATIO, VOCAL, BATCH_SIZE
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", dest="device")
    parser.add_argument("-r", "--sample_ratio", dest="sample_ratio")
    parser.add_argument("-v", "--vocal", dest="vocal", action="store_true")
    parser.add_argument("-t", "--n_trials", dest="n_trials", default=100)
    parser.add_argument("-b", "--batch_size", dest="n_trials", default=32)

    args = parser.parse_args()

    DEVICE = "cpu"
    if args.device: DEVICE = args.device

    SAMPLE_RATIO = 1
    if args.sample_ratio: SAMPLE_RATIO = float(args.sample_ratio)

    VOCAL = args.vocal
    BATCH_SIZE = int(args.batch_size)

    n_trials = int(args.n_trials)


    study = optuna.create_study(direction="minimize") # minimize the test-loss
    study.optimize(objective, n_trials=n_trials)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

if __name__ == "__main__":
    main()
