import optuna
from optuna.trial import TrialState

import torch
import torch.nn as nn

from argparse import ArgumentParser
from train.train_sign_lang import TrainSignLang
from train.train_nn import get_class_weights
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
    suite = TrainSignLang( 
        epochs = trial.suggest_int("epochs", 13, 40),
        lr = trial.suggest_float("lr", 0.00075, 0.001),
        batch_size = BATCH_SIZE,
        device = DEVICE,

    )
    df = pd.read_csv("./data/sign_lang_train/labels.csv")

    suite.init_data(
        image_dir="./data/sign_lang_train",
        label_df=df,
        sample_ratio=SAMPLE_RATIO,
        threshold=300
    )
    # define model
    model = SignLangCNN(
        n_flayers=trial.suggest_int("n_flayers", 2, 4),
        n_clayers=trial.suggest_int("n_clayers", 2, 3),
        hidden_size=trial.suggest_int("hidden_size", 64, 512, step=32),
        in_channels=IN_CHANNEL,
        in_features=IN_FEATS,
        out_features=OUT_FEATS,
        trial=trial
    )

    # loss function
    cel_weights_dict = get_class_weights(suite._df)
    cel_weights = torch.tensor(list(cel_weights_dict.values()), dtype=torch.float32).to(DEVICE)

    lf = torch.nn.CrossEntropyLoss(weight=cel_weights)

    suite.init_model(
        model = model,
        optim="Adam",
        loss_fn=lf
    )

    return suite.train_loop(vocal=VOCAL, trial=trial)
def read_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--device", dest="device")
    parser.add_argument("-r", "--sample_ratio", dest="sample_ratio")
    parser.add_argument("-v", "--vocal", dest="vocal", action="store_true")
    parser.add_argument("-t", "--n_trials", dest="n_trials", default=100)
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=32)

    args = parser.parse_args()

    return args

def main():
    global DEVICE, SAMPLE_RATIO, VOCAL, BATCH_SIZE

    # read in commandline arguments
    args = read_args()
    DEVICE = "cpu"
    if args.device: DEVICE = args.device

    SAMPLE_RATIO = 1
    if args.sample_ratio: SAMPLE_RATIO = float(args.sample_ratio)

    VOCAL = args.vocal
    BATCH_SIZE = int(args.batch_size)

    n_trials = int(args.n_trials)

    # create and start study
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
