from train.multi_model_train import MultiModelTrainSignLang
from preprocessing.preprocessing import get_unique_labels
from models.cnn_models import SingleConvSignLang_4
from torch.nn import BCELoss

import os
import pandas as pd


IMG_DIR = os.path.join(".", "data", "sign_lang_train") 
LABELS_PATH = os.path.join(IMG_DIR, "labels.csv")

def train(epochs, lr, batch_size, train_set_size, device, df):
    uq_labels = get_unique_labels(df)

    models = {}
    for l in uq_labels:
        models[l] = SingleConvSignLang_4()

    print("Num. Models:", len(models.keys()))
    suite = MultiModelTrainSignLang(epochs, lr, batch_size, train_set_size, device)
    suite.init_model(model=models,
                     loss_fn=BCELoss(),
                     optim="Adam")
    suite.init_data(image_dir=IMG_DIR,
                    label_df=df,
                    augment_data=False,
                    sample_ratio=1,
                    threshold=-1)
    print("ACC:", suite.evaluate())
    suite.train(vocal=True)

    print(suite.len_trl)


if __name__ == "__main__": 
    df = pd.read_csv(LABELS_PATH)

    EPOCHS = 10
    LR = 0.001
    batch_size = 32
    train_set_size = 0.8
    device = "cpu"

    train(EPOCHS, LR, batch_size, train_set_size, device, df)

       
