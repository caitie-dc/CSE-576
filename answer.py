import pdb

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms


# %%
class NN(nn.Module):
    def __init__(self, arr=[]):
        super(NN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(30 * 30 * 3, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# %%
class SimpleCNN(nn.Module):
    def __init__(self, arr=[]):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Conv2d(3, 8, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(1568, 5)

    def forward(self, x):
        x = self.conv_layer(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
        

# %%
basic_transformer = transforms.Compose([transforms.ToTensor()])

#Add color normalization to the transformer using 0.5 for mean and std for each channel. 
norm_transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])


# %%
class DeepCNN(nn.Module):
    def __init__(self, arr=[]):
        super(DeepCNN, self).__init__()
        size = 30
        layers = []
        in_channels = 3
        last_channels = 3
        for val in arr:
            if val == 'pool':
                layers.append(nn.MaxPool2d(2))
                size = size // 2
            else:
                layers.append(nn.Conv2d(in_channels, val, 3))
                in_channels = val
                last_channels = val
                size = size - 2
        self.layers = nn.ModuleList(layers)
        self.fc1 = nn.Linear(size * size * last_channels, 5)

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                x = layer(x)
            else:
                x = layer(x)
                x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
        


# %%

aug_transformer = transforms.Compose([transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5, shear=10),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
