from abc import ABC, abstractmethod
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch
import os

class TrainSuite(ABC):
    def __init__(self,
                 epochs: int = 30,
                 lr: float = 0.001,
                 batch_size: int = 32,
                 train_set_size: float = 0.8,
                 device: str = "cpu"
                 ) -> None:
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.train_set_size = train_set_size
        self.device = device


    def _init_optim(self):
        if self.optim_str is None or self.optim_str == "":
            print("ERROR: Optimizer undefined.")
            return
        
        if self.optim_str == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
            self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=2)


    def init_model(self,
                   model: nn.Module,
                   loss_fn,
                   optim: str,
                   ):
        self.model = model
        model.to(self.device)
        self.optim_str = optim; self._init_optim()
        self.loss_fn = loss_fn


    @abstractmethod
    def init_data(self, 
                  image_dir: os.PathLike | str,
                  label_df: pd.DataFrame,
                  augment_data: bool=True,
                  sample_ratio: float = 1.0,
                  threshold: int = -1):
        pass


    @abstractmethod
    def train(self, vocal=False):
        pass


    @abstractmethod
    def evaluate(self, vocal=False) -> tuple:
        pass


    @abstractmethod
    def _gen_data_info(self) -> str:
        pass


    @abstractmethod
    def save_model(self, dir: str | os.PathLike, vocal=False):
        pass


    @abstractmethod
    def save_info(self, info_dir: str | os.PathLike):
        pass
