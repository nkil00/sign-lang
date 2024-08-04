from sklearn.model_selection import train_test_split

from .utils import default_transform, augmented_transform
from .utils import filter_by_label
from .dataset import SignLanguageDataset

from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

import pandas as pd
import os

IMAGE_DIR = os.path.join("..", "data", "sign_lang_train")
LABEL_DIR = os.path.join(IMAGE_DIR, "labels.csv")

IMG_HEIGHT = 128

def get_unique_labels(df: pd.DataFrame = None, col_name="label", file_path: str = None):
    if df is None:
        df = pd.read_csv(file_path)
    cls = df[col_name].unique()

    return cls

def data_distribution(df: pd.DataFrame) -> dict:
    uq_labels = get_unique_labels(df)
    cls_cups = {cls:0 for cls in uq_labels}

    for _, row in df.iterrows():
        label = row.iloc[0]
        cls_cups[label] += 1
    
    return cls_cups

def augment_data(df: pd.DataFrame, train_cls_indiv: dict, threshold: int=300, img_dir: os.path = None) -> SignLanguageDataset:
    # 
    cls_uq = get_unique_labels(df)
    cls_cups = data_distribution(df)

    cls_to_augment = {l: train_cls_indiv[l] for l in train_cls_indiv.keys() if cls_cups[l] < threshold}
    cls_nto_augment = {l: train_cls_indiv[l] for l in train_cls_indiv.keys() if cls_cups[l] >= threshold}

    aug_datasets = []

    def_transform = default_transform(IMG_HEIGHT, IMG_HEIGHT)
    aug_transform = augmented_transform(IMG_HEIGHT, IMG_HEIGHT)

    for c in cls_to_augment.keys():
        aug_ds = SignLanguageDataset(annotations=cls_to_augment[c],
                                     transform=aug_transform,
                                     img_dir=img_dir)
        aug_datasets.append(aug_ds)
    
    naug_ds = SignLanguageDataset(annotations=df,
                                  transform=def_transform,
                                  img_dir=img_dir)

    conc_ds = aug_datasets + [naug_ds]

    train_dataset = ConcatDataset(conc_ds)

    return train_dataset

def create_data_sets(label_dir: os.path, img_dir: os.path, train_size:float=0.8, label_col_name:str="label", random_state:int=None, augment: bool=True):
    img_label_df = pd.read_csv(label_dir)
    unique_labels = get_unique_labels(img_label_df)

    train_labels, test_labels = train_test_split(img_label_df,
                                                 train_size=train_size, 
                                                 stratify=img_label_df[label_col_name],
                                                 random_state=None)
    train_labels_indiv = {l: filter_by_label(l, train_labels) for l in unique_labels}


    def_transform = default_transform(IMG_HEIGHT, IMG_HEIGHT)
    if augment:
        train_dataset = augment_data(df=train_labels, train_cls_indiv=train_labels_indiv, img_dir=img_dir)
    else:
        train_dataset = SignLanguageDataset(train_labels, transform=def_transform, img_dir=img_dir)
    
    test_dataset = SignLanguageDataset(test_labels, transform=def_transform, img_dir=img_dir)

    return train_dataset, test_dataset




def create_data_loaders(batch_size=32, train_size=0.8, label_dir: os.path = None, img_dir: os.path = None, augment_data:bool = True):
    train_ds, test_ds = create_data_sets(label_dir, img_dir, train_size=train_size, augment=augment_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def main():
    # get label info
    print("Preprocessing...")


if __name__ == "__main__":
    main()

