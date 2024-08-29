from train.train_suite import TrainSuite

from preprocessing.preprocessing import split_df_labels, create_data_loaders, get_unique_labels
from train.train_nn import train_batch_binary_classification, evaluate_batch_loss_binary
from torch import nn
import torch
import pandas as pd
from tqdm import tqdm
import os

class MultiModelTrainSignLang(TrainSuite):
    """ Class to enable training/predictions using a seperate model for each label"""
    def __init__(self,
                 epochs: int = 30,
                 lr: float = 0.001,
                 batch_size: int = 32,
                 train_set_size: float = 0.8,
                 device: str = "cpu"
                 ) -> None:
        super().__init__(epochs=epochs, 
                         lr=lr,
                         batch_size=batch_size,
                         train_set_size=train_set_size,
                         device=device)

    def init_data(self, 
                  image_dir: os.PathLike | str,
                  label_df: pd.DataFrame,
                  augment_data: bool = True,
                  sample_ratio: float = 1.0,
                  threshold: int = -1):
        if self._multi_model is None:
            raise ValueError("ERROR: Initialize the model(s) first!")
        if self._multi_model == False:
            raise ValueError("ERROR: Provide a dictionary of the models instead of a single one!")
        
        # create a df for each individual class
        self._df = split_df_labels(df=label_df, label_col="label", labels = [""])
        self.train_loader = {}
        self.test_loader = {}
        self.len_trl = {}
        self.len_tel = {}      
        # create data loaders for each class
        for lkey in self._df.keys():
            self.train_loader[lkey], self.test_loader[lkey] = create_data_loaders(label_df=self._df[lkey],
                                                                                  img_dir=image_dir,
                                                                                  batch_size=self.batch_size,
                                                                                  train_size=self.train_set_size,
                                                                                  augment_data=augment_data,
                                                                                  sample_ratio=sample_ratio)
            self.len_trl[lkey] = len(self.train_loader[lkey].dataset)
            self.len_tel[lkey] = len(self.test_loader[lkey].dataset)



    def train(self, vocal=False):
        """ train each individual model """ 
        if vocal: print("> Starting Training...")
        unique_labels = self._df.keys()
        # losses
        self.model_train_losses = {l: [] for l in unique_labels}
        self.model_test_losses = {l: [] for l in unique_labels}

        # train loops (train each model)
        for label in unique_labels:
            if vocal: print("> Trainig model with label:", label)

            model = self.model[label] # critical
            train_loader = self.train_loader[label]
            test_loader = self.test_loader[label]
            cls_idx = {label: 1, "_": 0}

            from tqdm import tqdm
            for ep in tqdm(range(self.epochs)):
                # train model
                model.train()
                running_train_loss = 0
                for batch in train_loader:
                    feat, _ = batch
                    loss = train_batch_binary_classification(model=model, 
                                                              batch=batch, 
                                                              optimizer=self.optimizer[label],
                                                              loss_function=self.loss_fn,
                                                              class_index=cls_idx,
                                                              device=self.device)
                    running_train_loss += loss * feat.size(0)

                epoch_train_loss = running_train_loss / self.len_trl[label]

                # evaluate model
                for batch in self.test_loss:
                    loss = 


    def evaluate(self, vocal=False):
        pass


    def _gen_data_info(self):
        pass


    def save_model(self, dir: str | os.PathLike, vocal=False):
        pass


    def save_info(self, info_dir: str | os.PathLike):
        pass
