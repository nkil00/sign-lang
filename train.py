import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from models.cnn_models import ConvSignLangNN, ConvSignLangNN_7, ConvSignLangNN_6
from preprocessing import preprocessing
from preprocessing.preprocessing import get_unique_labels
from preprocessing.utils import generate_info, write_info

from datetime import datetime
import getopt
import sys
import os

from tqdm import tqdm

INFO_DIR = os.path.join(".", "eval", "info")
MODEL_DIR = os.path.join(".", "eval", "model-prm")
IMG_DIR = os.path.join(".", "data", "sign_lang_train")
LABEL_DIR = os.path.join(IMG_DIR, "labels.csv")

TRAIN_SIZE = .8
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
DATA_AUGMENTATION = True

# get mapping dicts

unique_labels = get_unique_labels(file_path = LABEL_DIR)
class_index = {label:idx for idx, label in enumerate(unique_labels)}
index_class = {idx:label for idx, label in enumerate(unique_labels)}

### function definitions
def conv_idx_prediction_to_class(predictions: np.ndarray, idx_label_dict: dict):
    """
    
    :param batch_predictions:  of which the _rows_ are the probabilities of a sample to belong to the specific class (at idx), _cols_ are samples (should be BATCH_SIZE columns)
    :param idx_class_dict: dictionary where a numerical value is mapped to its class, e.g. {0: "l", 1: "a", ...}
    :return: a list of length BATCH_SIZE 
    """
    class_predictions = []
    for smp_predictions in predictions:
        final_prediction_num = np.argmax(smp_predictions)
        final_prediction = idx_label_dict[final_prediction_num]
        class_predictions.append(final_prediction)
        
    return class_predictions

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

### read in hyper_params
args_list = sys.argv[1:]
opts = "t:b:e:l:d:"
lopts = ["train_size=", "batch_size=", "epochs=", "learning_rate=", "data_augmentation="]

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
        if ca in ("-d", "--data_augmentation"):
            if int(cv) == 1:
                DATA_AUGMENTATION = True
            else:
                DATA_AUGMENTATION = False


except getopt.error as e:
    print(e)
    exit()

# get test and data loader
train_loader, test_loader  = preprocessing.create_data_loaders(label_dir=LABEL_DIR, img_dir=IMG_DIR, augment_data=DATA_AUGMENTATION)

train_size = len(train_loader.dataset)
test_size = len(test_loader.dataset)

### model initialization
model_0 = ConvSignLangNN_7()
optim = torch.optim.Adam(params=model_0.parameters(), lr = LEARNING_RATE)
print(optim)
loss_fn_0 = torch.nn.CrossEntropyLoss()

info_str = generate_info(EPOCHS, BATCH_SIZE, train_size, test_size, LEARNING_RATE, model_0, optim)

print(info_str)

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


# evaluate the models perfmance
print("-" * 80, "\nEvaluating the model...")

model_0.eval()

batches_predictions = []
batches_labels = []

# get predictions
with torch.no_grad():
    for img, label in tqdm(test_loader):
        batch_preds = model_0(img).numpy()
        batch_preds = conv_idx_prediction_to_class(batch_preds, index_class)
        batches_predictions.append(batch_preds)
        batches_labels.append(list(label))

# evaluate predictions
batches_predictions = [x for sub in batches_predictions for x in sub]
batches_labels = [x for sub in batches_labels for x in sub]

correct_test_samples = 0
total_test_samples = 0
for bp, bl in zip(batches_predictions, batches_labels):
    if bp == bl:
        correct_test_samples += 1
    total_test_samples += 1

accuracy = (correct_test_samples / total_test_samples)

print(f"> The total accuracy of the model is: {accuracy:.2f}.\n")


### save model parameters

date = f"{datetime.now().strftime('%m-%d--%H:%M')}"
model_name = f"model_{total_test_losses[-1]:.3f}_{date}"
MODEL_DIR = os.path.join(MODEL_DIR, f"{model_name}.pth")
torch.save(model_0.state_dict(), MODEL_DIR)
print("=" * 100, f"\nSaved model: {model_name}")

### write info file
INFO_DIR = os.path.join(INFO_DIR, model_name + ".md")
info_str += f"\nAccuracy: {accuracy * 100:.3f}%"
write_info(info_str, INFO_DIR)




