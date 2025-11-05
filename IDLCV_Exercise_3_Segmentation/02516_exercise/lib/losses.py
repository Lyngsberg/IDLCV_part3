import torch
from torch import nn
import torch.nn.functional as F

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # Numerically stable binary cross-entropy
        loss = torch.mean(y_pred - y_true * y_pred + torch.log(1 + torch.exp(-y_pred)))
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)  # Make sure predictions are in [0,1]
        y_true = y_true.float()
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_true = y_true.float()
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
        return torch.mean(focal_loss)

class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # BCE part
        loss = torch.mean(y_pred - y_true * y_pred + torch.log(1 + torch.exp(-y_pred)))

        # Total variation regularization
        tv_h = torch.sum(torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))
        tv_w = torch.sum(torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]))
        total_variation = tv_h + tv_w

        return loss + 0.1 * total_variation
