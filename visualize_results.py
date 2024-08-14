from argparse import ArgumentParser
from plotting.data_plt import plot_data_distribution, plot_predictions, plot_prediction_matrix
from models.cnn_models import ConvSignLangNN_7
from preprocessing.preprocessing import create_data_loaders, get_unique_labels

from eval.evaluation import prediction_matrix, predictions_actual

import os
import pandas as pd

import torch


DATA_PATH = os.path.join(".", "data", "sign_lang_train", "labels.csv")
IMG_DIR = os.path.join(".", "data", "sign_lang_train")



def plt_data_distribution(path: str | os.PathLike):
    # load data
    df = pd.read_csv(path)
    # plot
    plot_data_distribution(df)

def plt_predictions(model, test_loader, index_class, unique_labels):
    plot_predictions(model, test_loader,  index_class, unique_labels)



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
        plt_data_distribution(DATA_PATH)
    if args.show_predictions:
        plt_predictions(model, test_loader, index_class, unique_labels)
    
    upred, uactual = predictions_actual(model, test_loader, index_class)
    pred_mat = prediction_matrix(upred, uactual, unique_labels)
    plot_prediction_matrix(pred_mat, unique_labels).show()

if __name__ == "__main__":
    main()







