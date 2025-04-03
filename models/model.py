from mlaicore.models.base_model import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkinToneCNNModel(BaseModel):
    def __init__(self, num_classes=10, image_size=512, channels=3):
        super(BaseModel, self).__init__()
        
        # Enhanced architecture with more feature extraction capacity
        self.conv_channels = [channels, 32, 64, 128, 256, 512]
        self.num_conv_layers = len(self.conv_channels) - 1
        self.kernel_size = 3
        self.padding = 1
        self.stride = 1
        self.pool_size = 2
        self.dropout_rate = 0.3  # Reduced dropout for better training
        self.image_size = image_size
        
        # Create convolutional blocks (conv + bn + relu + dropout)
        self.conv_blocks = nn.ModuleList()
        
        for i in range(self.num_conv_layers):
            block = nn.Sequential(
                nn.Conv2d(
                    self.conv_channels[i], 
                    self.conv_channels[i+1], 
                    kernel_size=self.kernel_size, 
                    stride=self.stride, 
                    padding=self.padding
                ),
                nn.BatchNorm2d(self.conv_channels[i+1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(self.pool_size, self.pool_size) if (i % 2 == 1 or i == 0) else nn.Identity()  # Pool every other layer
            )
            self.conv_blocks.append(block)
        
        # Global Average Pooling to reduce parameter count and improve generalization
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers with better architecture
        self.classifier = nn.Sequential(
            nn.Linear(self.conv_channels[-1], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Apply convolutional blocks
        for block in self.conv_blocks:
            x = block(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x
    
    def configure_optimizers(self, learning_rate=0.001):
        """Configure model optimizers"""
        # Use AdamW instead of regular Adam for better weight decay handling
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=1e-4  # Add weight decay for regularization
        )
        
        # Use a learning rate scheduler to reduce LR when training plateaus
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }