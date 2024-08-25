import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from datetime import datetime

import pandas as pd
from preprocessing.preprocessing import create_data_loaders, get_class_index_dict
from train.train_nn import train_batch_classification, evaluate_batch_loss, predict_batch, label_to_int_index
from preprocessing.preprocessing import balance_labels



import os

class TrainSignLang():
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

    def init_data(self, 
                  image_dir: os.PathLike | str,
                  label_df: pd.DataFrame,
                  augment_data: bool = True,
                  sample_ratio: float = 1.0,
                  threshold: int = -1):
        if threshold > 0: label_df = balance_labels(label_df,
                                                    threshold=threshold)
        train_loader, test_loader = create_data_loaders(img_dir=image_dir,
                                                        label_df=label_df, 
                                                        batch_size=self.batch_size, 
                                                        train_size=self.train_set_size, 
                                                        augment_data=augment_data, 
                                                        sample_ratio=sample_ratio)
        self.augment_data = augment_data
        self._df = label_df
        self.train_loader = train_loader
        self.len_trl = len(train_loader.dataset)
        self.test_loader = test_loader
        self.len_tel = len(test_loader.dataset)

    def train(self, vocal=False):
        """
        Trains the network in the train set. Keeps track of the test and train losses over the epochs.
            :param vocal: If True print info to console.
            :return: None
        """
        if vocal: 
            print(f"Starting training.")
            print(f"Model is on device: {next(self.model.parameters()).device}")

        # setup
        self._class_index_dict = get_class_index_dict(self._df)
        best_params = self.model.state_dict()
        min_loss = float("inf")

        # training
        self.train_losses = []
        self.test_losses = []

        for epoch in range(self.epochs):
            # train on train set
            running_loss_train= 0
            self.model.train()
            for batch in tqdm(self.train_loader):
                feat, _ = batch
                feat.to(self.device)
                loss = train_batch_classification(self.model,
                                                  batch,
                                                  self.optimizer,
                                                  self.loss_fn,
                                                  self._class_index_dict, 
                                                  self.device)
                
                running_loss_train = running_loss_train + (loss * feat.size(0))

            epoch_loss_train = running_loss_train / self.len_trl 
            self.train_losses.append(epoch_loss_train)
                
            # get test set loss
            self.model.eval()
            running_loss_test = 0

            with torch.no_grad():
                for batch in self.test_loader:
                    feat, _ = batch
                    feat.to(self.device)
                    loss = evaluate_batch_loss(model=self.model,
                                   batch=batch, 
                                   loss_function=self.loss_fn, 
                                   class_index=self._class_index_dict,
                                   device=self.device)
                    running_loss_test = running_loss_test + (loss * feat.size(0))

            epoch_test_loss = running_loss_test / self.len_tel
            if epoch_test_loss < min_loss:
                best_params = self.model.state_dict()
                min_loss = epoch_test_loss
            self.test_losses.append(epoch_test_loss)

            if vocal: print(f"Epoch \"{epoch}\" done. | test-loss = {epoch_test_loss:.3f} | train-loss = {epoch_loss_train:.3f}")

        date = datetime.now().strftime("%m%d-%H%M")
        loss = self.test_losses[-1]
        self.model.load_state_dict(best_params)
        self.model_name = f"model_L-{min_loss:.4f}--{date}"

    def evaluate(self, vocal=False):
        """
        Returns the test loss and accuracy of the predictions based on the test set.
            params: vocal: if True, print the results to the console.
            return: 
        """
        final_loss = min(self.test_losses)
        correct = 0
        for b in tqdm(self.test_loader):
            _, tar = b
            predictions = predict_batch(model=self.model, batch=b, device=self.device)
            tar_num = label_to_int_index(label=list(tar), class_index_dict=self._class_index_dict)
            correct += np.sum([1 for x, y in zip(predictions, tar_num) if x==y])


        accuracy = correct / self.len_tel
        self.accuracy = accuracy
        if vocal: print(f"accuracy = {accuracy}, test-loss={final_loss}")

        return accuracy, final_loss

    def __str__(self):
        binds = 80
        sep = "-" * binds + "\n"
        self_str = "TrainSignLang:\n"
        epochs = "- Epochs: " + str(self.epochs) + "\n"
        lr = "- Learning Rate: " + str(self.lr) + "\n"
        batch = "- Batch Size: " + str(self.batch_size) + "\n"
        train_set_size = "- % Train-Set: " + str(self.train_set_size) + "\n"
        model = f"model: \n```\n{self.model}\n```\n"
        opt = f"optimizer:\n ```\n{self.optimizer}\n```\n"

        self_str += epochs + lr + train_set_size + batch + sep
        self_str += model + sep
        self_str += opt + sep

        return self_str

    def _gen_data_info(self):
        binds = 80
        sep = "-" * binds + "\n"
        testr = f"- Size Test-Set: {self.len_tel}\n"
        trstr = f"- Size Train-Set: {self.len_trl}\n"
        ttstr = f"- Total Size: {len(self._df)}\n"
        augstr = f"- Data Augmentation: {self.augment_data}"

        return trstr + testr + ttstr + sep

    def save_model(self, dir: str | os.PathLike, vocal=False):
        if self.model is None:
            print("ERROR: Model has not been initialized yet.")
            return 

        if not os.path.exists(dir): os.makedirs(dir)

        torch.save(self.model.state_dict(), os.path.join(dir, self.model_name + ".pth"))

        if vocal: print(f"Saved model: \"{self.model_name}\"")


    def save_info(self, info_dir: str | os.PathLike):
        if self.test_losses is None:
            print("ERROR: Model has not been trained yet. Can't save the model!")
            return
        sep = "-" * 80 + "\n"
        loss = self.test_losses[-1]
        accuracy = self.accuracy
        info = self.__str__() + self._gen_data_info() + f"- Accuracy: {accuracy:.3f}\n- Loss: {loss:.3f}\n{sep}"

        if not os.path.exists(info_dir): os.makedirs(info_dir)
        with open(os.path.join(info_dir, self.model_name + ".md"), "w") as w:
            w.write(info)


