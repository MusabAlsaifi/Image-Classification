# Imports
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms, datasets


# Define transformations for training and testing/validation
TRAIN_TRANSFORM = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.RandomRotation(30),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# transforms for testing and validation
TEST_TRANSFORM = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
                                            
def data_loader(dataset: datasets, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    """
    Create a DataLoader for the given dataset.

    Parameters:
    - dataset (datasets): The dataset to be loaded.
    - batch_size (int): Number of samples in each batch.
    - shuffle (bool): Whether to shuffle the dataset.
    - device (torch.device): The device to move the data to (e.g., "cuda" or "cpu"). Can be None for CPU.

    Returns:
    - DataLoader: DataLoader for the specified dataset.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=os.cpu_count() if str(device) == "cuda" else 0,
                            pin_memory=True if str(device) == "cuda" else False)
    
    return dataloader
    
def trainloader(device: torch.device, dataset: str, batch_size: int, transform=TRAIN_TRANSFORM) -> DataLoader:
    """
    Create a DataLoader for the training dataset.

    Parameters:
    - device (torch.device): The device to move the data to (e.g., "cuda" or "cpu").
    - dataset (torch.utils.data.Dataset): The training dataset. Default is train_data.
    - batch_size (int): Number of the batch size.
    - transform (torchvision.transforms.Compose): The transformation applied to the data. Default is TRAIN_TRANSFORM.

    Returns:
    - trainloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    """
    train_data = datasets.ImageFolder(root=dataset, transform=transform)
    trainloader = data_loader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        device=device
    )
    
    return trainloader


def validateloader(device: torch.device, dataset: str, batch_size: int, transform=TEST_TRANSFORM) -> DataLoader:
    """
    Create a DataLoader for the validation dataset.

    Parameters:
    - device (torch.device): The device to move the data to (e.g., "cuda" or "cpu").
    - dataset (torch.utils.data.Dataset): The training dataset. Default is train_data.
    - batch_size (int): Number of the batch size.
    - transform (torchvision.transforms.Compose): The transformation applied to the data. Default is TEST_TRANSFORM.

    Returns:
    - validateloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    """
    validata_data = datasets.ImageFolder(root=dataset, transform=transform)
    validateloader = data_loader(
        dataset=validata_data,
        batch_size=batch_size,
        shuffle=False,
        device=device
    )
    
    return validateloader


def testloader(device: torch.device, dataset: str, batch_size: int, transform=TEST_TRANSFORM) -> DataLoader:
    """
    Create a DataLoader for the test dataset.

    Parameters:
    - device (torch.device): The device to move the data to (e.g., "cuda" or "cpu").
    - dataset (torch.utils.data.Dataset): The training dataset. Default is train_data.
    - batch_size (int): Number of the batch size.
    - transform (torchvision.transforms.Compose): The transformation applied to the data. Default is TEST_TRANSFORM.

    Returns:
    - testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    """
    test_data = datasets.ImageFolder(root=dataset, transform=transform)
    testloader = data_loader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        device=device
    )
    
    return testloader

def process_image(image_path: str, transform=TEST_TRANSFORM) -> torch.Tensor:
    '''
    Process an image for use in a PyTorch model.

    Parameters:
    - image_path (str): Path to the input image.
    - transform (torchvision.transforms.Compose): The transformation applied to the image. Default is TEST_TRANSFORM.

    Returns:
    - torch.Tensor: Processed image tensor with batch dimension.
    '''
    # Process a PIL image for use in a PyTorch model
    img = Image.open(image_path).convert("RGB")

    # Apply the transformations to the image
    img_tensor = transform(img)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor

def imshow(image: torch.Tensor, ax: str=None, title: str=None):
    """
    Display an image in a Matplotlib subplot.

    Parameters:
    - image (torch.Tensor): Image tensor to be displayed.
    - ax (matplotlib.axes.Axes, optional): Matplotlib subplot axes. If not provided, a new subplot is created.
    - title (str, optional): Title for the subplot.

    Returns:
    - matplotlib.axes.Axes: The Matplotlib subplot axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.cpu().numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.set_title(title)

    return ax