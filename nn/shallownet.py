# Implement my first model: shallownet

# import the necessary packages
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# implement class by subclassing nn.Module
class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(90000, 3)

    def forward(self, x):
        x = self.conv(x)
        # if input and output have same dimension size, we can use this trick to save memory
        x[:] = F.relu(x)
        x = torch.flatten(x)
        x = self.fc(x)
        x[:] = F.softmax(x)
        return x

# script code for testing my ShallowNet class
if __name__ == "__main__":
    img = Image.open("C:\\Users\\linsole\\Project\\dl4cv_pytorch_starter\\dataset\\animals\\cats\\cats_00001.jpg")
    net = ShallowNet()
    transform = transforms.Compose([transforms.PILToTensor()])
    in_img = torch.unsqueeze(transform(img), 0)
    out_img = net(in_img.float())
    print(out_img.shape)
