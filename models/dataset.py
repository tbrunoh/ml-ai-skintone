import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from urllib.request import urlopen

class SkinToneDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        if img_path.startswith("http://") or img_path.startswith("https://"):
            with urlopen(img_path) as response:
                image = Image.open(response).convert("RGB")
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            
        # print(type(image), image.shape, image.dtype)  # Check type, shape, and dtype of the image
        return image, label

