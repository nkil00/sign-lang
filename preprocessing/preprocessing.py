from sklearn.model_selection import train_test_split

from .utils import default_transform, augmented_transform
from .utils import filter_by_label
from .dataset import SignLanguageDataset

from torch.utils.data import ConcatDataset, DataLoader, Dataset

import numpy as np
import pandas as pd
import os

IMAGE_DIR = os.path.join("..", "data", "sign_lang_train")
LABEL_DIR = os.path.join(IMAGE_DIR, "labels.csv")

IMG_HEIGHT = 128

## helper
def get_unique_labels(df: pd.DataFrame = None, col_name="label", file_path: str | os.PathLike = ""):
    if df is None:
        df = pd.read_csv(file_path)
    cls = df[col_name].unique()

    return cls
def get_class_index_dict(df: pd.DataFrame, col_name = "label") -> dict:
    labels = get_unique_labels(df, col_name)
    class_idx_dict = {l: idx for idx, l in enumerate(labels)}

    return class_idx_dict

def stratified_sample(df, label_col, sample_fraction):
    # Ensure the sample fraction is between 0 and 1
    if not (0 < sample_fraction <= 1):
        raise ValueError("sample_fraction must be a float between 0 and 1.")
    
    # Calculate the sample size
    sample_size = int(len(df) * sample_fraction)

    # Calculate the fraction of each label in the original DataFrame
    label_distribution = df[label_col].value_counts(normalize=True)

    # Perform stratified sampling
    sampled_dfs = []

    for label, fraction in label_distribution.items():
        label_subset = df[df[label_col] == label]
        n_samples = int(round(fraction * sample_size))
        
        # Sample the calculated number of rows from this subset
        sampled_dfs.append(label_subset.sample(n=n_samples, replace=False))

    # Concatenate all sampled subsets into a single DataFrame
    stratified_sample_df = pd.concat(sampled_dfs, ignore_index=True)

    return stratified_sample_df

def data_distribution(df: pd.DataFrame) -> dict:
    uq_labels = get_unique_labels(df)
    cls_cups = {cls:0 for cls in uq_labels}

    for _, row in df.iterrows():
        label = row.iloc[0]
        cls_cups[label] += 1
    
    return cls_cups

def augment_data(df: pd.DataFrame, train_cls_indiv: dict, img_dir: str | os.PathLike,threshold: int=300) -> Dataset:
    # 
    cls_cups = data_distribution(df)

    cls_to_augment = {l: train_cls_indiv[l] for l in train_cls_indiv.keys() if cls_cups[l] < threshold}

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

def create_data_sets(label_df: pd.DataFrame, img_dir: str | os.PathLike, train_size:float=0.8, label_col_name:str="label", random_state:int=None, augment: bool=True, sample_ratio: float = 1):
    unique_labels = get_unique_labels(label_df)

    label_df = stratified_sample(label_df, "label", sample_ratio)

    train_labels, test_labels = train_test_split(label_df,
                                                 train_size=train_size, 
                                                 stratify=label_df[label_col_name],
                                                 random_state=random_state)
    train_labels_indiv = {l: filter_by_label(l, train_labels) for l in unique_labels}


    def_transform = default_transform(IMG_HEIGHT, IMG_HEIGHT)
    if augment:
        train_dataset = augment_data(df=train_labels, train_cls_indiv=train_labels_indiv, img_dir=img_dir)
    else:
        train_dataset = SignLanguageDataset(train_labels, transform=def_transform, img_dir=img_dir)
    
    test_dataset = SignLanguageDataset(test_labels, transform=def_transform, img_dir=img_dir)

    return train_dataset, test_dataset

def balance_labels(df: pd.DataFrame, threshold: int = 300, label_col: str = "label"):
    df_cpy = df.copy()
    # get labels that are above the threshold
    cups_classes = data_distribution(df_cpy)
    cls_abv_thrs = {cls: num  for cls, num in zip(cups_classes.keys(), cups_classes.values()) 
                    if cups_classes[cls] > threshold}

    # remove samples such that they are at the threshold
    for cl in cls_abv_thrs.keys():
        n = cls_abv_thrs[cl] - threshold
        label_idx = df_cpy[df_cpy[label_col] == cl].index
        idx_to_drop = np.random.choice(label_idx, n, replace=False)
        df_cpy.drop(idx_to_drop, inplace=True)

    return df_cpy


def create_data_loaders(label_df: pd.DataFrame, img_dir: str | os.PathLike, batch_size=32, train_size=0.8, augment_data:bool = True, sample_ratio: float = 1.0):
    if img_dir == "":  print("Please provide a path for the img files!")
        
    train_ds, test_ds = create_data_sets(label_df, img_dir, train_size=train_size, augment=augment_data, sample_ratio=sample_ratio)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader

def split_df_labels(df: pd.DataFrame, label_col: str, labels: list[str]):
    """
    Takes as input a dataframe creates for each label an own dataframe, if not specified otherwise.

    """
    df_c = df.copy()
    df_idx = df_c.index
    # 
    cls_cups = data_distribution(df)
    dfs = {l: pd.DataFrame() for l in cls_cups.keys()}

    for l in cls_cups.keys():
        # get df part for labels
        df_l = df[df[label_col] == l]
        l_idx = df_l.index
        n = len(df_l)

        # sample n random rows that are NOT "l"
        nl_idx = np.random.choice([i for i in df_idx if i not in l_idx], n, replace=False)
        df_nl = df.iloc[nl_idx]
        df_nl["label"] = "k"

        # concate the single df's
        df_l = pd.concat([df_l, df_nl])

        dfs[l] = df_l

    return dfs 







def main():
    # get label info
    print("Preprocessing...")


if __name__ == "__main__":
    main()

