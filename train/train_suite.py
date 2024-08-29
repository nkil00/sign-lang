from abc import ABC, abstractmethod
import pandas as pd
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
        self._df = None
        self.model = None


    def _init_optim(self):
        if self.optim_str is None or self.optim_str == "":
            raise ValueError("ERROR: Optimizer undefined.")
            
        
        if self.optim_str == "Adam" and not self._multi_model:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        elif self.optim_str == "Adam":
            # init ALL models with optmizer
            self.optimizer = {}
            for mkey in self.model.keys():
                self.optimizer[mkey] = torch.optim.Adam(self.model[mkey].parameters(), self.lr)


    def init_model(self, model: nn.Module | dict[nn.Module], loss_fn, optim: str,):
        print("JLKJLK")
        if type(model) == dict:
            self._multi_model = True
            for mkey in model.keys():
                model[mkey].to(self.device)
                self.loss_fn = loss_fn
        else:
            self._multi_model = False
            self.model = model
            model.to(self.device)
            self.loss_fn = loss_fn

        self.model = model
        self.optim_str = optim; self._init_optim()

    @abstractmethod
    def init_data(self, 
                  image_dir: os.PathLike | str,
                  label_df: pd.DataFrame,
                  augment_data: bool = True,
                  sample_ratio: float = 1.0,
                  threshold: int = -1):
        pass


    @abstractmethod
    def train(self, vocal=False):
        pass


    @abstractmethod
    def evaluate(self, vocal=False):
        pass


    @abstractmethod
    def _gen_data_info(self):
        pass


    @abstractmethod
    def save_model(self, dir: str | os.PathLike, vocal=False):
        pass


    @abstractmethod
    def save_info(self, info_dir: str | os.PathLike):
        pass
