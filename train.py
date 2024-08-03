import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from models.cnn_models import ConvSignLangNN
from preprocessing import preprocessing
from preprocessing.preprocessing import get_unique_labels

import getopt
import sys

from tqdm import tqdm


TRAIN_SIZE = .8
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10

# get mapping dicts
unique_labels = get_unique_labels()
class_index = {label:idx for idx, label in enumerate(unique_labels)}
index_class = {idx:label for idx, label in enumerate(unique_labels)}

### function definitions

def label_to_int_index(label: str, class_index_dict: dict):
    return class_index[label]

def train_batch(model: nn.Module, batch, optimizer: torch.optim.Optimizer, loss_function):
    # enable training

    model.train()
    img, tar = batch
    index_of_label = torch.tensor([label_to_int_index(t, class_index) for t in tar], dtype=torch.long)

    # begin training
    optimizer.zero_grad()
    out = model(img)
    batch_loss = loss_function(out, index_of_label)
    batch_loss.backward()
    optimizer.step()
    
    return batch_loss.item()

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

### read in hyper_params
args_list = sys.argv[1:]
opts = "t:b:e:l:"
lopts = ["train_size=", "batch_size=", "epochs=", "learning_rate="]

try:
    args, vals = getopt.getopt(args_list, opts, lopts)

    for ca, cv in args:
        if ca in ("-t", "--train_size"):
            TRAIN_SIZE = float(cv)
        if ca in ("-l", "--learning_rate"):
            LEARNING_RATE = float(cv)
        if ca in ("-e", "--epochs"):
            EPOCHS = int(cv)
        if ca in ("-b", "--batch_size"):
            BATCH_SIZE = int(cv)

except getopt.error as e:
    print(e)
    exit()

print("=" * 100)
print("Hyperparamters:")
print("> EPOCHS:", EPOCHS)
print("> BATCH_SIZE:", BATCH_SIZE)
print("> LEARNING RATE:", LEARNING_RATE)
print("> TRAIN SIZE:", TRAIN_SIZE)

print("=" * 100)

# get test and data loader
train_loader, test_loader  = preprocessing.create_data_loaders()

print("Size Train-Set:", len(train_loader.dataset))
print("Size Test-Set:", len(test_loader.dataset))

print("=" * 100)


### model initialization
model_0 = ConvSignLangNN()
optim = torch.optim.Adam(params=model_0.parameters(), lr = LEARNING_RATE)
loss_fn_0 = torch.nn.CrossEntropyLoss()

print("Model:\n", model_0)
print("=" * 100)
print("Starting training...")

### start training
total_train_losses = []
total_test_losses = []

for epoch in range(EPOCHS):
    epoch_loss_train = []
    epoch_loss_test = []
    batch_nr = 0
    model_0.train()
    
    ## train and gather loss
    running_loss_train = 0
    for batch in tqdm(train_loader):
        feat, _ = batch
        loss = train_batch(model_0, batch, optim, loss_fn_0) # loss is already *.item()
        running_loss_train = running_loss_train + (loss * feat.size(0))
        
    epoch_loss_train = running_loss_train / len(train_loader.dataset)
    total_train_losses.append(epoch_loss_train)
    
    ## evaluate model on test set
    model_0.eval()
    running_loss_test = 0
    with torch.no_grad():
        for feat, tar in test_loader:
            out = model_0(feat)
            index_of_label = torch.tensor([label_to_int_index(t, class_index) for t in tar], dtype=torch.long)
            loss = F.cross_entropy(out, index_of_label)
            running_loss_test = running_loss_test + (loss * feat.size(0))
    
    epoch_loss_test = running_loss_test / len(test_loader.dataset)
    total_test_losses.append(epoch_loss_test)
    print(f"Epoch {epoch} done. | Loss = {epoch_loss_test}")

print(f"Training done. Final loss: {total_test_losses[-1]}")
