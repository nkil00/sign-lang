from abc import ABC, abstractmethod
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
    def train(self, vocal=False):
        pass


    @abstractmethod
    def evaluate(self, vocal=False):
        pass


    def _gen_data_info(self):
        binds = 80
        sep = "-" * binds + "\n"
        testr = f"- Size Test-Set: {self.len_tel}\n"
        trstr = f"- Size Train-Set: {self.len_trl}\n"
        ttstr = f"- Total Size: {len(self._df)}\n"
        augstr = f"- Data Augmentation: {self.augment_data}"

        return trstr + testr + ttstr + sep


    @abstractmethod
    def save_model(self, dir: str | os.PathLike, vocal=False):
        pass


    @abstractmethod
    def save_info(self, info_dir: str | os.PathLike):
        pass
