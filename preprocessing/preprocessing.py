import torch
import pandas as pd
import os


IMAGE_DIR = os.path.join(".", "data", "sign_lang_train")
LABEL_DIR = os.path.join(IMAGE_DIR, "labels.csv")
TRAIN_SIZE = 0.8


def get_test_train_loader(batch_size=32, learning_rate=0.001, train_size=0.8):
    return 0, 1

def main():
    # get label info
    img_labels = pd.read_csv(LABEL_DIR) # [label] [img_name]
    print("Collecting label information from: <", LABEL_DIR, ">", sep="")

if __name__ == "__main__":
    main()

