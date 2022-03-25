# My custom Dataset class for Animals

# import the necessary packages
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms import CenterCrop
import matplotlib.pyplot as plt

# implement custom dataset, modify code from official pytorch tutorial:
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class DatasetAnimals(Dataset):
    def __init__(self, annotations_file, transform=None, label_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 1]
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 2]

        # use the passed transform list and apply preprocessor 
        # one by one to each image
        if self.transform:
            for preprocessor in self.transform:
                image = preprocessor(image)

        # also transform label if passed not-None value
        if self.label_transform:
            label = self.label_transform(label)
        return image, label

# code for testing the DatasetAnimals Class
if __name__ == "__main__":
    # instantiate the preprocessor for resizing, then instantiate 
    # dataset and data loader
    preprocessor = [CenterCrop([200])]
    dataset = DatasetAnimals("animals.csv", transform=preprocessor)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # iterate *once* through the data loader to see the feature and label size, 
    # beware that the Animals dataset contain both 3-channel and 1-channel 
    # images, so may get runtime error during iteration if batch_size > 1
    train_features, train_labels = next(iter(data_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0]
    print(f"Feature size: {img.size()}")
    label = train_labels[0]
    print(f"Label: {label}")

    # plot the tensor image using pyplot, use permute to change the ordering
    # of the image channels
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
