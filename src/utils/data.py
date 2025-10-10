import torchvision
import torch
import os
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.transforms import v2
from torchvision.transforms.v2 import Transform
from torchvision.datasets import ImageFolder
from src.utils.basics import displayImage

from src.config import Config

def loadData(folderPath: str, batchSize: int, dataAugmentationTechniques: list[Transform] = []) -> tuple[DataLoader | None, DataLoader | None, DataLoader | None]:
    '''
        Load images from a specified folder path, splits them into training, validation and test sets and 
        return a DataLoader object for each set.

        @param folderPath: The path to the folder containing the images.
        @param batchSize: The number of samples per batch to load.

        @return: A tuple with 3 loaders: train, validation and test'''
    splitPercentages = [Config.TRAINING_SPLIT, Config.VALIDATION_SPLIT, Config.TEST_SPLIT]
    if (sum(splitPercentages) != 1.0):
        raise ValueError(f"The sum of {splitPercentages} must be equal to 1.0")
    
    # Base transformations (always applied)
    basicTransformations: list[any] = [
        torchvision.transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)), # Resize images to 128x128 pixels
        torchvision.transforms.ToTensor() # Convert images to PyTorch tensors and rescale pixel values to [0, 1]
    ]

    # Define the transformations for the train set
    trainTransforms = torchvision.transforms.Compose(
        dataAugmentationTechniques + basicTransformations
    )

    # Define the transformations for the other sets (validation, test)
    otherTransforms = torchvision.transforms.Compose(
        basicTransformations
    )

    # Load the dataset from the specified folder path, without transforms
    baseDataset = torchvision.datasets.ImageFolder(
        root=folderPath
    )

    # Convert split percentages to actual lengths
    totalSize = len(baseDataset)
    splitSizes: list[int] = [round(percentage * totalSize) for percentage in splitPercentages]

    # Adjust last split to fix rounding errors
    splitSizes[-1] = totalSize - sum(splitSizes[:-1])

    subSets: list[Subset] = random_split(baseDataset, splitSizes)

    trainIndices, validationIndices, testIndices = [subSet.indices for subSet in subSets]



    # Create the datasets
    trainDataset = Subset(ImageFolder(root=folderPath, transform=trainTransforms), indices=trainIndices)
    validationDataset = Subset(ImageFolder(root=folderPath, transform=otherTransforms), indices = validationIndices)
    testDataset = Subset(ImageFolder(root=folderPath, transform=otherTransforms), indices=testIndices)

    trainLoader: DataLoader | None = None
    validationLoader: DataLoader | None = None
    testLoader: DataLoader | None = None

    # Create DataLoader object for each dataset, only if the dataset is not empty
    if (len(trainDataset.indices) > 0):
        trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True)

    if (len(validationDataset.indices) > 0):
        validationLoader = DataLoader(validationDataset, batch_size=batchSize, shuffle=True)
    
    if (len(testDataset.indices) > 0):
        testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=True)

    
    return (trainLoader, validationLoader, testLoader)


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