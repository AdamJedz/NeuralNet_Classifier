from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy as np
import torch
import matplotlib.pyplot as plt

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class_names = ['plane', '  car', ' bird', '  cat', ' deer', '  dog', ' frog', 'horse', ' ship', 'truck']


def download_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=.2),
        transforms.RandomRotation(degrees=30),
        transforms.ToTensor()
    ])

    train_data = datasets.CIFAR10(root='./Data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./Data', train=False, download=True, transform=transform)
    return train_data, test_data


def create_loaders(train, test, batch_size=10):
    torch.manual_seed(0)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def show_sample(train_loader):

    np.set_printoptions(formatter=dict(int=lambda x:f'{x:5}'))

    for images, labels in train_loader:
        break

    print('Label:', labels.numpy())
    print('Class:', *np.array([class_names[i] for i in labels]))

    im = make_grid(images, nrow=5)
    plt.figure(figsize=(10, 4))
    plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
    plt.show()


if __name__ == "__main__":

    logging.info('Downloading CIFAR10 dataset.')
    train_data, test_data = download_data()
    logging.info('CIFAR10 downloaded.')

    train_loader, test_loader = create_loaders(train_data, test_data)
    logging.info('Data loaders created')

    logging.info('Showing random 10 pictures from train dataset.')
    show_sample(train_loader)
