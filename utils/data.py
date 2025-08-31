import torchvision
import torch
from torch.utils.data import DataLoader, random_split
from utils.basics import displayImage

from config import Config

def loadData(folderPath: str, splitPercentages: list[float], batchSize: int) -> list[DataLoader]:
    '''
        Load images from a specified folder path, splits them into sets and 
        return a DataLoader object for each set.

        @param folderPath: The path to the folder containing the images.
        @param splitPercentages: A list of floats representing the percentages of train set, test set etc
        @param batchSize: The number of samples per batch to load.

        @return: A list of DataLoader objects, one for each set.'''
    if (sum(splitPercentages) != 1.0):
        raise ValueError("The sum of {splitPercentages} must be equal to 1.0")
    
    # Create the transform object
    transoforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)), # Resize images to 128x128 pixels
        torchvision.transforms.ToTensor() # Convert images to PyTorch tensors and rescale pixel values to [0, 1]
    ])

    # Load the dataset from the specified folder path
    dataset = torchvision.datasets.ImageFolder(
        root=folderPath,
        transform=transoforms
    )

    print(len(dataset))

    # Create the ImageFolder datasets
    datasets: list[torchvision.datasets.ImageFolder] = random_split(dataset, splitPercentages)

    # Create DataLoader objects for each dataset
    loaders: list[DataLoader] = [
        DataLoader(dataset, batch_size=batchSize, shuffle=True) for dataset in datasets
    ]
        
    return loaders


def extractDataFromLoader(loader: DataLoader)-> tuple[torch.Tensor, torch.Tensor]:
    '''
        Extracts all images and labels from a DataLoader and returns them as tensors.

        @param loader: The DataLoader to extract data from.

        @return: A tuple containing two tensors:
            - images: A tensor of shape (N, C, H, W) where N is the number of images,
              C is the number of channels, H is the height, and W is the width.
            - labels: A tensor of shape (N,) containing the corresponding labels for each image (values 0 or 1)
    '''
    images = []
    labels = []

    for batchImages, batchLabels in loader:
        images.append(batchImages)
        labels.append(batchLabels)

    # Concatenate all batches into single tensors
    imagesTensor = torch.cat(images)
    labelsTensor = torch.cat(labels)

    return imagesTensor, labelsTensor