import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from urllib.request import urlopen
from mlaicore.models.dataset import BaseDataset
class SkinToneDataset(BaseDataset):
    def __init__(self, csv_file, transform=None):
        super(SkinToneDataset, self).__init__(transform=transform)
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        image = self.load_image(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        # print(type(image), image.shape, image.dtype)  # Check type, shape, and dtype of the image
        return image, label

