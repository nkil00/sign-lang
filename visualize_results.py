from argparse import ArgumentParser
from plotting.data_plt import plot_data_distribution, plot_predictions, plot_prediction_matrix
from models.cnn_models import ConvSignLangNN_7
from preprocessing.preprocessing import create_data_loaders, get_unique_labels

from eval.evaluation import prediction_matrix, predictions_actual

import os
import pandas as pd

import matplotlib.pyplot as plt

import torch


DATA_PATH = os.path.join(".", "data", "sign_lang_train", "labels.csv")
IMG_DIR = os.path.join(".", "data", "sign_lang_train")



def plt_data_distribution(path: str | os.PathLike):
    # load data
    df = pd.read_csv(path)
    # plot
    plot_data_distribution(df)

def plt_predictions(model, test_loader, index_class, unique_labels):
    plot_predictions(model, index_class, test_loader, unique_labels)



def plot_chosen():
    pass

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", dest="model_name")
    parser.add_argument("-d", "--ddist", 
                        dest="ddistr", 
                        action="store_true")
    parser.add_argument("-p", 
                        "--showpreds", 
                        dest="show_predictions",
                        action="store_true")

    parser.add_argument("-M",
                        "--prediction_matrix",
                        dest = "pred_mat",
                        action="store_true")
    args = parser.parse_args()

    model_dir = os.path.join(".", "eval", "model-prm", args.model_name)

    model = ConvSignLangNN_7()
    model.load_state_dict(torch.load(model_dir))

    _, test_loader = create_data_loaders(label_dir=DATA_PATH,
                                        img_dir=IMG_DIR, 
                                        augment_data=True)
    unique_labels = get_unique_labels(file_path=DATA_PATH)
    index_class = {idx:label for idx, label in enumerate(unique_labels)}

    if args.ddistr:
        print("Start plot data distribution..")
        plt_data_distribution(DATA_PATH)
        print("Done plot data distribution..")

    if args.show_predictions:
        print("Start plot data pred..")
        plt_predictions(model, test_loader, index_class, unique_labels)
        print("Done plot data pred..")

    if args.pred_mat:
        print("Start plot pred mat..")
        upred, uactual = predictions_actual(model, test_loader, index_class)
        pred_mat = prediction_matrix(upred, uactual, unique_labels)
        plot_prediction_matrix(pred_mat, unique_labels)
        print("End plot pred mat..")

if __name__ == "__main__":
    main()







