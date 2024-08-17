import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.nn.modules.loss import _Loss
from data.sign_lang_train import 

from preprocessing.dataset import SignLanguageDataset
from preprocessing.preprocessing import create_data_loaders

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
        
        if self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)

    def init_model(self,
                   model: nn.Module,
                   loss_fn,
                   optimizer: str,
                   ):
        self.model = model
        self.optimizer = optimizer; self._init_optim()
        self.loss_fn = loss_fn

    def init_data(self, 
                  image_dir: os.PathLike | str,
                  labels_path: os.PathLike | str,
                  augment_data: bool = True):
       train_loader, test_loader = create_data_loaders( self.batch_size, self.train_set_size, img_dir = image_dir, label_dir = labels_path, augment_data=augment_data)

       self.train_loader = train_loader
       self.test_loader = test_loader

    def train(self):
        self.train_losses = []
        self.test_losses = []


