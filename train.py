# training procedure of my network

from matplotlib import transforms
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from nn.shallownet import ShallowNet
from utility.DatasetAnimals import DatasetAnimals

# train test split
transform = transforms.Compose([transforms.ToTensor(), \
    transforms.RandomResizedCrop((400, 400))])
dataset = DatasetAnimals("utility/animals.csv", transform=transform)
dataset_size = len(dataset)
index = list(range(len(dataset)))
np.random.seed(42)
np.random.shuffle(index)
ratio = 0.05
split = int(np.floor(ratio * dataset_size))
train_indices, val_indices = index[split:], index[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, 4, sampler=train_sampler)
val_loader = DataLoader(dataset, 4, sampler=val_sampler)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training epoch
for e in range(100):
    for batch_idx, (img, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}, shape: {img.shape}, label: {label}")
