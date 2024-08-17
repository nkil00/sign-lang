import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from data.sign_lang_train import 

from tqdm import tqdm

import pandas as pd
from preprocessing.dataset import SignLanguageDataset
from preprocessing.preprocessing import create_data_loaders, get_class_index_dict

from train.train_nn import train_batch_classification, predict

import os

class TrainSignLang():
    def __init__(self,
                 epochs: int = 30,
                 lr: float = 0.001,
                 batch_size: int = 32,
                 train_set_size: float = 0.8,
                 ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.train_set_size = train_set_size
    
    def _init_optim(self):
        if self.optimizer is None or self.optimizer == "":
            print("ERROR: Optimizer undefined.")
            return
        
        if self.optim_str == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    def init_model(self,
                   model: nn.Module,
                   loss_fn,
                   optim: str,
                   ):
        self.model = model
        self.optim_str = optim; self._init_optim()
        self.loss_fn = loss_fn

    def init_data(self, 
                  image_dir: os.PathLike | str,
                  labels_path: os.PathLike | str,
                  augment_data: bool = True):
        train_loader, test_loader = create_data_loaders( self.batch_size, self.train_set_size, img_dir = image_dir, label_dir = labels_path, augment_data=augment_data)

        self._df = pd.read_csv(labels_path)
        self.train_loader = train_loader
        self.len_trl = len(train_loader.dataset)
        self.test_loader = test_loader
        self.len_tel = len(test_loader.dataset)

    def train(self):
        # setup
        class_index_dict = get_class_index_dict(self._df)

        # training
        self.train_losses = []
        self.test_losses = []


        for epoch in tqdm(range(self.epochs)):
            # train on train set
            running_loss_train= 0
            for batch in self.train_loader:
                feat, _ = batch
                loss = train_batch_classification(self.model,
                                                  batch,
                                                  self.optimizer,
                                                  self.loss_fn,
                                                  class_index_dict)
                
                running_loss_train = running_loss_train + (loss * feat.size(0))
            epoch_loss_train = running_loss_train / self.len_trl 
            self.train_losses.append(epoch_loss_train)
                
            # get test set loss
            self.model.eval()
            running_loss_test = 0
            for batch in self.test_loader:
                feat, _ = batch
                loss = predict(model=self.model,
                               batch=batch, 
                               loss_function=self.loss_fn, 
                               class_index=class_index_dict)
                running_loss_test = running_loss_test + (loss * feat.size(0))
            epoch_test_loss = running_loss_test * self.len_tel
            self.test_losses.append(epoch_test_loss)

            print(f"Epoch \"{epoch}\" done. | Loss = {epoch_test_loss:.3f}")

        print(f"Training done! Final test loss = {self.test_losses[-1]}")



