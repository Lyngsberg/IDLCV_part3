import os
import numpy as np
import glob
import PIL.Image as Image

# pip install torchsummary
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
import torch.optim as optim
from time import time
from lib.model.EncDecModel import EncDec
#from lib.model.DilatedNetModel import DilatedNet
#from lib.model.UNetModel import UNet, UNet2
from lib.losses import BCELoss, DiceLoss, FocalLoss, BCELoss_TotalVariation
from lib.dataset.PhCDataset import PhC


def main():
    # Dataset
    size = 128
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    batch_size = 6
    trainset = PhC(train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    testset = PhC(train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

    print(f"Loaded {len(trainset)} training images")
    print(f"Loaded {len(testset)} test images")

    # Training setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EncDec().to(device)
    # model = UNet().to(device)
    # model = UNet2().to(device)
    # model = DilatedNet().to(device)
    # summary(model, (3, 256, 256))
    learning_rate = 0.001
    opt = optim.Adam(model.parameters(), learning_rate)

    loss_fn = BCELoss()
    # loss_fn = DiceLoss()
    # loss_fn = FocalLoss()
    # loss_fn = BCELoss_TotalVariation()
    epochs = 20

    # Training loop
    X_test, Y_test = next(iter(test_loader))
    model.train()
    for epoch in range(epochs):
        tic = time()
        print(f'* Epoch {epoch+1}/{epochs}')

        avg_loss = 0
        for X_batch, y_true in train_loader:
            X_batch = X_batch.to(device)
            y_true = y_true.to(device)

            opt.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            opt.step()

            avg_loss += loss / len(train_loader)

        print(f' - loss: {avg_loss}')

    torch.save(model, "model.pth")
    print("Training has finished!")


if __name__ == "__main__":
    import multiprocessing
    # On HPC, this prevents DataLoader crashes
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("fork", force=True)

    main()
