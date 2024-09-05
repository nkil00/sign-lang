import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from preprocessing.preprocessing import get_unique_labels

import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader


colors = mcolors.CSS4_COLORS
plt.style.use("ggplot")


## helper

def plot_data_distribution(df: pd.DataFrame):
    # get the distribution data
    unique_labels = get_unique_labels(df)
    num_labels = len(unique_labels)
    labels_amount = {l:0 for l in unique_labels}
        
    dcolors = list(colors.keys())[:num_labels]
    for _, row in df.iterrows():
        label = row[0]
        labels_amount[label] += 1

    plt.bar(range(num_labels),
            list(labels_amount.values()),
            tick_label=list(labels_amount.keys()),
            color=dcolors)
    plt.ylabel = "Number of Elements"
    plt.xlabel = "Different Classes" 
    plt.title("Distribution of Elements per Class in Whole Dataset")

    return plt


 
def conv_idx_prediction_to_class(batch_predictions: np.ndarray, idx_class_dict: dict):
    """
    
    :param batch_predictions:  of which the _rows_ are the probabilities of a sample to belong to the specific class (at idx), _cols_ are samples (should be BATCH_SIZE columns)
    :param idx_class_dict: dictionary where a numerical value is mapped to its class, e.g. {0: "l", 1: "a", ...}
    :return: a list of length BATCH_SIZE 
    """
    class_predictions = []
    for smp_predictions in batch_predictions:
        final_prediction_num = np.argmax(smp_predictions)
        final_prediction = idx_class_dict[final_prediction_num]
        class_predictions.append(final_prediction)
        
    return class_predictions

def plot_predictions(model: nn.Module, model_name: str, index_class: dict, data_loader: DataLoader, labels: list):
    """
    plots the distribution of actual labels in a data loader, such as the distribution of the
    predictions of the data loader
    """
    model.eval()

    predictions = []
    actual_labels = []
    with torch.no_grad():
        for feat, tar in data_loader: 
            out = model(feat).numpy()
            preds = conv_idx_prediction_to_class(out, index_class)
            predictions.append(preds)
            actual_labels.append(list(tar))

    # 

    upreds = [x for sub in predictions for x in sub]
    uactual = [x for sub in actual_labels for x in sub]

    ac_labels_cups = {l:0 for l in labels}
    pr_labels_cups = {l:0 for l in labels}

    for l in uactual:
        ac_labels_cups[l] += 1
    
    for l in upreds:
        pr_labels_cups[l] += 1
    
    fig, ax = plt.subplots(2, 1, figsize=(12, 7))

    num_classes = len(ac_labels_cups.keys())

    dcolors = list(colors.keys())[:num_classes]
    # plot prediction distribution
    ax[1].bar(range(num_classes), list(pr_labels_cups.values()), tick_label = labels, color=dcolors)
    ax[1].set_title(f"Distribution of Predictions ({model_name})")

    # plot data loader distribution
    ax[0].bar(range(num_classes), list(ac_labels_cups.values()), tick_label = labels, color=dcolors)
    ax[0].set_title(f"Distribution of Elements in DataLoader ({model_name})")

    return plt

def plot_accuracy(model: nn.Module, model_name: str, index_class: dict, data_loader: DataLoader, labels: list):
    """ plots the accuracy of the model considering each individual label """
    model.eval()

    predictions = []
    actual_labels = []
    with torch.no_grad():
        for feat, tar in data_loader: 
            out = model(feat).numpy()
            preds = conv_idx_prediction_to_class(out, index_class)
            predictions.append(preds)
            actual_labels.append(list(tar))

    upreds = [x for sub in predictions for x in sub]
    uactual = [x for sub in actual_labels for x in sub]
    
    uactual_total = {l: 0 for l in labels}

    for l in uactual:
        uactual_total[l] += 1
    
    correct_preds = {l: 0 for l in labels}

    for ac, pr in zip(uactual, upreds):
        if ac == pr:
            correct_preds[ac] += 1

    accuracy = {l: (correct_preds[l] / uactual_total[l])*100 for l in labels}
    num_classes = len(accuracy.keys())
    dcolors = list(colors.keys())[:num_classes]

    plt.bar(range(num_classes), list(accuracy.values()), tick_label=labels, color=dcolors)
    plt.title(f"Accuracy of model \"{model_name}\" for each class", fontsize=12)
    plt.xlabel("Different Classes")
    plt.ylabel("Accuracy in %")

    # print("Missing return value!")
    return plt 
    



def plot_prediction_matrix(pred_mat, labels, model_name):
    plt.figure(figsize=(8, 6))
    plt.imshow(pred_mat)
    plt.colorbar()

    plt.title(f"Prediction Matrix of ({model_name})")
    plt.xticks(range(len(labels)), labels, size="small")
    plt.yticks(range(len(labels)), labels, size="small")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")

    plt.grid(linewidth=.1)
    return plt


    

