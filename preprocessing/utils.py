from datetime import datetime 
import os

import pandas as pd

import torch
from torchvision.transforms import v2
from torchvision import transforms


_normalization_def = v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])

"""  -> Interesting Transforms:
1) RandomPerspective
2) RandomRotation
2) HorizontalFlip

"""
def rotate_transform(H, W):
	return v2.Compose([transforms.Resize((H, W)),
						v2.RandomHorizontalFlip(p=1),
						transforms.ToTensor(),
						_normalization_def
	                   ])


def augmented_transform(H, W):
	"""  """
	return v2.Compose([transforms.Resize((H, W)),
	       			   #v2.RandomVerticalFlip(p=.5),
					   v2.RandomHorizontalFlip(p=1),
	                   transforms.ToTensor(),
	                   _normalization_def
	                   ])

def horizontal_transform(H, W):
	""" should be irrelevant if one uses left/right hand """
	return v2.Compose([v2.Resize((H, W)),
                       v2.RandomHorizontalFlip(p=1),
					   v2.RandomRotation(30),
                       v2.RandomPerspective(p=.5, distortion_scale=.5),
                       transforms.ToTensor(),
                       _normalization_def])

def default_transform(H, W):
	return transforms.Compose([v2.Resize((H, W)),
							   v2.RandomRotation(30),
							   v2.RandomPerspective(p=.5, distortion_scale=.5),
	                           transforms.ToTensor(),
	                           _normalization_def
	                           ])


def filter_by_label(label: str, dataframe: pd.DataFrame, col_name: str = "label"):
	# copy df
	df = dataframe.copy(deep=True)
	return df[df[col_name] == label]

def generate_info(epochs, batch_size, train_size, test_size, lr, model, optimizer, binds=80) -> str:
    sep = "-"*binds + "\n"
    date = f"# {datetime.now().strftime('%Y.%m-%d %H:%M:%S')}\n\n"
    ep  = f"- Epochs: {epochs}\n"
    bs = f"- Batch Size: {batch_size}\n"
    trs = f"- Trainset Size: {train_size}\n"
    tes = f"- Testset Size: {test_size}\n"
    lr = f"- Learning Rate: {lr}\n"
    modelstr = f"model:\n```\n{model}\n```\n"
    optim_str = f"optimizer:\n```\n{optimizer}\n```\n"
    
    info_str = date + sep + ep + lr + bs + trs + tes + lr + sep + modelstr + sep + optim_str

    return info_str
    
def write_info(info_str: str, file_path: str | os.PathLike, add_info: str = ""):
    with open(file_path, "w") as p:
        p.write(info_str)
        if add_info is not None:
            p.write(add_info)
