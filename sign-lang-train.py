from train.train_sign_lang import TrainSignLang
from models.cnn_models import ConvSignLangNN_7

from torch.nn import CrossEntropyLoss

import os

LABEL_PATH = os.path.join(".", "data", "sign_lang_train", "labels.csv") 
IMG_DIR = os.path.join(".", "data", "sign_lang_train")
"./models/state_dicts/"
MODEL_DIR = os.path.join(".", "models", "state_dicts")


if __name__ == "__main__":
    training_model = TrainSignLang(epochs=1, 
                                   train_set_size=0.8)
    training_model.init_data(image_dir=IMG_DIR, labels_path=LABEL_PATH)

    convs7 = ConvSignLangNN_7()
    optim = "Adam"
    loss_fn = CrossEntropyLoss()
    training_model.init_model(model=convs7, loss_fn=loss_fn, optim=optim)

    print(training_model)

    training_model.train()
    ac, loss = training_model.evaluate()
    print(f"Accuracy = {ac:.3f}")
    training_model.save_model(MODEL_DIR, vocal=True)
