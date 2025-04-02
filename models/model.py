
from mlaicore.models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkinToneCNNModel(BaseModel):
    def __init__(self, num_classes=10,image_size=512,channels=3):
        super(BaseModel, self).__init__()
        
        # Adjust convolution layers based on image size
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv2

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization after conv3
        
        self.pool = nn.MaxPool2d(2, 2)  # Reduce spatial dimensions

        self.dropout1 = nn.Dropout(0.5)  # Dropout after fc1
        self.dropout2 = nn.Dropout(0.5)  # Dropout after fc2
        
        # Calculate final feature map size
        final_size = image_size // 8  # 3 pooling layers reduce size by 2^3 = 8

        # Fully connected layers
        self.fc1 = nn.Linear(128 * final_size * final_size, 256)
        self.fc2 = nn.Linear(256, num_classes)  # Output: 10 classes (Monk Scale)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.shape[0], -1)  # Flatten for FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Apply dropout
        x = self.fc2(x)
        x = self.dropout2(x)  # Apply dropout
        return x
        