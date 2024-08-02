import pandas as pd
import torch
from torchvision.transforms import v2
from torchvision import transforms


_normalization_def = v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])

def rotate_transform(H, W):
	return v2.Compose([transforms.Resize((H, W)),
	                   v2.RandomHorizontalFlip(p=0.5),
	                   transforms.RandomRotation(30),
	                   v2.ToDtype(torch.float32, scale=True),
	                   v2.ConvertImageDtype(dtype=torch.float32),
	                   _normalization_def
	                   ])


def augmented_transform(H, W):
	return v2.Compose([transforms.Resize((H, W)),
					   v2.RandomHorizontalFlip(p=1),
	                   v2.RandomVerticalFlip(p=1),
	                   transforms.ToTensor(),
#	                   _normalization_def
	                   ])


def default_transform(H, W):
	return transforms.Compose([v2.Resize((H, W)),
	                           transforms.ToTensor(),
#	                           _normalization_def
	                           ])


def filter_by_label(label: str, dataframe: pd.DataFrame, col_name: str = "label"):
	# copy df
	df = dataframe.copy(deep=True)
	return df[df[col_name] == label]
