from argparse import ArgumentParser

from torch.serialization import MAP_LOCATION 
from plotting.data_plt import plot_data_distribution, plot_predictions, plot_prediction_matrix, plot_accuracy
from models.cnn_models import ConvSignLangNN_7, ConvSignLangNN_4_
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
    return plot_data_distribution(df)

def plt_predictions(model, model_name, test_loader, index_class, unique_labels):
    return plot_predictions(model, model_name,index_class, test_loader, unique_labels)


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
    parser.add_argument("-a",
                        "--accuracy",
                        dest = "accuracy",
                        action="store_true")
    parser.add_argument("-D",
                        "--dir", 
                        dest="dir")

    args = parser.parse_args()

    model_dir = os.path.join(".", "eval", "model-prm","grids",args.model_name)
    model_dir = os.path.join(".", "sign_lang_top_models", args.model_name)

    if args.dir: model_dir = args.dir
    print(model_dir)
    model = ConvSignLangNN_4_(first_dim=256,
                              second_dim=256,
                              third_dim=128,
                              conv1_in=3,
                              conv2_in=8,
                              conv3_in=16)
    try:
        model.load_state_dict(torch.load(model_dir, map_location=torch.device("cpu")))
        model.to("cpu")
    except FileNotFoundError as _:
        print(f"The Path: {model_dir} is invalid!")
        exit()

    df = pd.read_csv(DATA_PATH)
    _, test_loader = create_data_loaders(label_df=df,
                                        img_dir=IMG_DIR, 
                                        augment_data=True,
                                         train_size=0.2)
    unique_labels = get_unique_labels(file_path=DATA_PATH)
    index_class = {idx:label for idx, label in enumerate(unique_labels)}

    if args.ddistr:
        print("Start plot data distribution..")
        p1 = plt_data_distribution(DATA_PATH)
        p1.show()
        print("Done plot data distribution..")

    if args.show_predictions:
        print("Start plot data pred..")
        p2 = plt_predictions(model, args.model_name, test_loader, index_class, unique_labels)
        p2.show()
        print("Done plot data pred..")

    if args.pred_mat:
        print("Start plot pred mat..")
        upred, uactual = predictions_actual(model, test_loader, index_class)
        pred_mat = prediction_matrix(upred, uactual, unique_labels)
        p3 = plot_prediction_matrix(pred_mat, unique_labels, args.model_name)
        p3.show()
        print("End plot pred mat..")
    
    if args.accuracy:
        p4 = plot_accuracy(model=model, model_name=args.model_name, data_loader=test_loader, index_class=index_class,
                      labels=unique_labels)
        p4.show()
    
    plt.show()
if __name__ == "__main__":
    main()







