from train.train_suite import TrainSuite

from preprocessing.preprocessing import split_df_labels, create_data_loaders, get_unique_labels
from train.train_nn import train_batch_binary_classification, evaluate_batch_loss_binary, predict_batch_mm
from sklearn.model_selection import train_test_split
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
        print("Initializing Data...")
        if self._multi_model is None:
            raise ValueError("ERROR: Initialize the model(s) first!")
        if self._multi_model == False:
            raise ValueError("ERROR: Provide a dictionary of the models instead of a single one!")
        
        # first split into train/test sets
        train_labels, test_labels = train_test_split(label_df,
                                                     train_size=self.train_set_size,
                                                     stratify=label_df["label"])

        # reset indices of df's -> for split_df_labels()
        train_labels.reset_index(inplace=True, drop=True)
        test_labels.reset_index(inplace=True, drop=True)

        # create a df for each individual class from train_labels
        self._df = split_df_labels(df=train_labels, label_col="label", labels = [""])
        self.train_loader = {}
        self.test_loader = {}
        self.len_trl = {}
        self.len_tel = {}      

        # create train data loaders for each class
        for lkey in self._df.keys():
            self.train_loader[lkey], _ = create_data_loaders(label_df=self._df[lkey],
                                                                                  img_dir=image_dir,
                                                                                  batch_size=self.batch_size,
                                                                                  train_size=0.99, # test samples are already excluded earlier
                                                                                  augment_data=augment_data,
                                                                                  sample_ratio=sample_ratio)
            # create one test loader
            _, self.test_loader = create_data_loaders(label_df=test_labels,
                                                      img_dir=image_dir,
                                                      batch_size=self.batch_size,
                                                      train_size=0.02, # train loader needs to have > num classes elements...
                                                      augment_data=augment_data,
                                                      sample_ratio = sample_ratio)

            self.len_trl[lkey] = len(self.train_loader[lkey].dataset)
            self.len_tel = len(self.test_loader.dataset)



    def train(self, vocal=False):
        """ train each individual model """ 
        if vocal: print("> Starting Training...")
        unique_labels = self._df.keys()
        # losses
        self.model_train_losses = {l: [] for l in unique_labels}
        self.model_test_losses = {l: [] for l in unique_labels}

        # train loops (train each model)
        for label in unique_labels:
            if vocal: print(f"> Trainig model with label \"{label}\"")

            model = self.model[label] # critical
            train_loader = self.train_loader[label]
            test_loader = self.test_loader
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
                # for batch in self.test_loss:
                #     loss = evaluate_batch_loss_binary(model, batch, self.loss_fn, cls_idx, self.device)

                if vocal: print(f"> Training Model with label \"{label}\" done.")


    def evaluate(self, vocal=False):
        """ Get Accuracy of the suite/model """
        if vocal: print("> Starting evaluation...")

        class_index = {key: i for i, key in enumerate(self.model.key()}
        correct = 0
        for batch in tqdm(self.test_loader:)
            _, tar = batch
            prediction = predict_batch_mm(models=self.model,
                                          batch=batch,
                                          device=self.device)
            
            # convert string targets to numerical values
            ntar = [class_index[l] for l list(tar)]
            # accumualte each correct prediction per batch
            correct += np.sum([1 for x, y in zip(ntar, prediction) if x == y])

        accuracy = correct / self.len_tel

        return accuracy

         

        


    def _gen_data_info(self):
        pass


    def save_model(self, dir: str | os.PathLike, vocal=False):
        pass


    def save_info(self, info_dir: str | os.PathLike):
        pass
