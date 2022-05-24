# training procedure of my network

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from sklearn.metrics import classification_report

from nn.shallownet import ShallowNet
from utility.DatasetAnimals import DatasetAnimals

np.random.seed(42)

# train test split
transform = transforms.Compose([transforms.ToTensor(), \
    transforms.Resize((32, 32))])
dataset = DatasetAnimals("utility/animals.csv", transform=transform)
dataset_size = len(dataset)
index = list(range(len(dataset)))
np.random.shuffle(index)
ratio = 0.25
split = int(np.floor(ratio * dataset_size))
train_indices, val_indices = index[split:], index[:split]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, 4, sampler=train_sampler)
val_loader = DataLoader(dataset, 4, sampler=val_sampler)


net = ShallowNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

# training epoch
for epoch in range(2):
    running_loss = 0.0
    for batch_idx, (img, label) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 9:    # print every 10 mini-batches
            print(f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 10:.3f}')
            running_loss = 0.0

#print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))
