from models.model import SkinToneCNNModel
from models.dataset import SkinToneDataset
from mlaicore.utils.utils import Utils
from torchvision import transforms
from dotenv import load_dotenv
import os
load_dotenv()

device = Utils.set_accelerator()

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize all images to 128x128
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

dataset = SkinToneDataset(csv_file=os.getenv("DATASET_PATH"), transform=transform)



