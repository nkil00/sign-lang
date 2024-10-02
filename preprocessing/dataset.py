from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class SignLanguageDataset(Dataset):
	def __init__(self, annotations, img_dir, transform=None):
		self.annotations = annotations
		self.img_dir = img_dir
		self.transform = transform

	def __len__(self):
		return len(self.annotations)

	def __getitem__(self, idx):
		img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 1])
		image = Image.open(img_path).convert("RGB")
		label = self.annotations.iloc[idx, 0]

		if self.transform:
			image = self.transform(image)

		return image, label

class KagSignLanguageDataset(Dataset):
    def __init__(self, annotations, image_data, transform=None):
        self.image_data = image_data
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image_raw = self.image_data.iloc[idx].reshape(28, 28)
        image = Image.fromarray(np.uint8(image_raw))
        label = self.annotations[idx]

        return image, label

