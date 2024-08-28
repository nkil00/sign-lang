from torch import nn
import torch
import numpy as np

from torch import Tensor

# conversion functions
def label_to_int_index(label: str | list, class_index_dict: dict):
    """
    Takes as input a label and a dictionary which maps the classes of the dataset to a specific
    index. Returns the index associated to that label.
        _param: label:
        _param: class_index_dict:
        return: int
    """
    if type(label) != list:
        return class_index_dict[label]

    return [class_index_dict[l] for l in label] 

# train functions
def train_batch_classification(model: nn.Module, batch, optimizer: torch.optim.Optimizer, loss_function,
                               class_index, device: str):
    # enable training
    model.train()
    img, tar = batch

    img = img.to(device)

    index_of_label = torch.tensor(label_to_int_index(list(tar), class_index), dtype=torch.long)
    index_of_label = index_of_label.to(device)

    # begin training
    optimizer.zero_grad()
    out = model(img)
    batch_loss = loss_function(out, index_of_label)
    batch_loss.backward()
    optimizer.step()
    
    return batch_loss.item()

def train_batch_bclsfic(model: nn.Module, batch, optimizer: torch.optim.Optimizer, loss_function,
                               class_index, device: str):
    # enable training
    model.train()
    img, tar = batch

    img = img.to(device)
    index_of_label = torch.tensor(label_to_int_index(list(tar), class_index), dtype=torch.float32)
    index_of_label = index_of_label.to(device)

    # begin training
    optimizer.zero_grad()
    out = torch.flatten(model(img))
    batch_loss = loss_function(out, index_of_label)
    batch_loss.backward()
    optimizer.step()
    
    return batch_loss.item()

def evaluate_batch_loss(model: torch.nn.Module, batch, loss_function, class_index, device):
    if model.training: model.eval()
    
    feat, tar = batch
    feat = feat.to(device)
    out = model(feat)
    idx_of_label = torch.tensor(label_to_int_index(list(tar), class_index), dtype=torch.long)
    idx_of_label = idx_of_label.to(device)

    loss = loss_function(out, idx_of_label)

    return loss.item()

def predict_batch(model: torch.nn.Module, batch, device) -> np.ndarray:
    """ Returns the predicted numerical labels of the batch. Sets the model to evaluation mode. """
    model.eval()
    with torch.no_grad():
        feat, _ = batch
        feat = feat.to(device)
        predictions = model(feat)
        print("no max:", predictions)
        predictions = predictions.detach().cpu()

        preds_max = np.argmax(predictions, axis=1)

    return preds_max


def predict_batch_binary(model: torch.nn.Module, batch, device) -> np.ndarray:
    """ Returns the predicted numerical labels of the batch. Sets the model to evaluation mode. """
    model.eval()
    with torch.no_grad():
        feat, _ = batch
        feat = feat.to(device)
        predictions = model(feat)
        print("no max:", predictions)
        predictions = predictions.detach().cpu()

        preds_max = np.round(predictions)

    return preds_max



