import torch
import torch.nn as nn

class OrdinalRegressionLoss(nn.Module):
    def __init__(self, num_classes):
        super(OrdinalRegressionLoss, self).__init__()
        self.num_classes = num_classes
        self.loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        # Convert targets to binary encoding
        targets_ord = torch.zeros(targets.size(0), self.num_classes)
        for i, target in enumerate(targets):
            targets_ord[i, :target] = 1
        
        # Apply loss to each ordinal output
        loss = self.loss(predictions, targets_ord.to(predictions.device))
        return loss