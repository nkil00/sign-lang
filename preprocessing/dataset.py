from PIL import Image
import os
from torch.utils.data import Dataset

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

