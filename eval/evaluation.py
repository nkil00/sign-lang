import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

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


def predictions_actual(model: nn.Module,
                       data_loader: DataLoader,
                       index_class: dict
                       ):
    """
    Takes as input a model and a dataloader, sets the model to evaluation mode and makes predictions
    for the values stored in "data_loader".
    Returns two lists; upred the predictions and uactual the actual labels.
    :param model:
    :param data_loader:
    :param index_class:

    :return: Two lists, the first containing the predictions, the second the actual value for the
    predicted. Both have the actual "labels" e.g. "a", "b", etc.
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


    upreds = [x for sub in predictions for x in sub]
    uactual = [x for sub in actual_labels for x in sub]

    return upreds, uactual

def _count_labels(labels: list, actual: list) -> dict:
    """ Count how many of each label is in "actual" """
    cnt_lbl = {l: 0 for l in labels}
    for i in actual:
        cnt_lbl[i] += 1

    return cnt_lbl

def prediction_matrix(predictions: list, 
                      actual: list, 
                      labels: list):
    l = len(labels)
    label_idx = {l: idx for idx, l in enumerate(labels)}
    pred_mat = np.zeros((l, l))

    for p, a in zip(predictions, actual):
        # label i, was predicted to be j
        i, j = (label_idx[a], label_idx[p])
        pred_mat[i][j] += 1

    return pred_mat
   
    
