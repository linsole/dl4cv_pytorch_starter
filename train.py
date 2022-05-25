# training procedure of my network

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

train_loader = DataLoader(dataset, 5, sampler=train_sampler)
val_loader = DataLoader(dataset, 5, sampler=val_sampler)


net = ShallowNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

# training epoch
for epoch in range(100):
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

classes = ['cat', 'dog', 'panda']
torch.save(net.state_dict(), './shallow_net.pth')
net.load_state_dict(torch.load('./shallow_net.pth'))
"""
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')
"""
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
