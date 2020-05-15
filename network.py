import numpy as np
import torch
import torch.nn as nn


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=.2, inplace=True),
            nn.Linear(4*4*128, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, X):
        X = self.conv_layers(X)
        X = torch.flatten(X, 1)
        X = self.classifier(X)
        return X