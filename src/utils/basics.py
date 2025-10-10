import torch
import torchvision
import matplotlib.pyplot as plt
from typing import Any
from datetime import datetime
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import math
from src.config import Config

def displayImage(image: torch.Tensor | Any):
    if isinstance(image, torch.Tensor):
        if (image.ndim == 3 and image.shape[1] == image.shape[2]):
            # Convert from (C, W, H) to (W, H, C)
            image = image.permute(1, 2, 0)
        
        plt.imshow(image)
        plt.axis('off')
        plt.show()



def measureAccuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    '''
        Measure the accuracy of the predictions against the true labels.

        @param predictions: The predicted outputs from the model (values between 0 and 1)
        @param labels: The true labels (values 0 or 1)

        @return: The accuracy as a float value between 0 and 1.
    '''
    # Convert predictions to binary (0 or 1) using a threshold of 0.5
    predictedClasses = predictions.clone()
    predictedClasses[predictedClasses >= 0.5] = 1
    predictedClasses[predictedClasses < 0.5] = 0

    # Calculate the number of correct predictions
    correct_predictions = predictedClasses.eq(labels).sum().item()
    
    # Calculate accuracy
    accuracy = correct_predictions / labels.size(0)
    
    return accuracy


def getCurrentTimeRepresentation() -> str:
    return datetime.now().strftime("%Y-%m-%d %H.%M")


def saveGraph(xValues: list[float], yValuesList: list[list[float]], colors: list[str], labels: list[str], title: str, xLabel: str, yLabel: str, fileName: str):
    '''
        Saves a graph with multiple lines.

        @param xValues: The x values for the graph.
        @param yValuesList: A list of lists, where each inner list contains the y values for a graph-line.
        @param colors: A list of colors for each graph-line.
        @param labels: A list of labels for each graph-line.
        @param title: The title of the graph.
        @param xLabel: The label for the x axis.
        @param yLabel: The label for the y axis.
        @param fileName: The name of the file to save the graph to.
    '''
    plt.figure()
    for yValues, color, label in zip(yValuesList, colors, labels):
        plt.plot(xValues, yValues, color=color, label=label)
    
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.legend()
    plt.grid(True)

    plt.savefig(fileName)


def saveModel(model: torch.nn.Module, path: str) -> None:
    '''
        Saves the model to the specified path.

        @param model: The model to save.
        @param path: The path to save the model to.
    '''
    torch.save(model.state_dict(), path)


def loadModel(model: torch.nn.Module, path: str) -> None:
    '''
        Loads the learnt paramemeters from the specified path into the given model.

        @param model: The model to load the state dict into.
        @param path: The path to load the model from.
    '''
    loaded = torch.load(path, weights_only=False)
    if (not loaded):
        print(f"Couldn't load model from path {path}!")
        return None
    model.load_state_dict(loaded)
    model.eval()



def imageAugmentation(image: torch.Tensor, transforms: torchvision.transforms.transforms.Compose, times: int = 6) -> None:
    '''
        Applies the given {transforms} to the given {image}, as many times as {times} says.
        The generated augmented images are plotted into a graph.
    '''
    if (image is None):
        raise ValueError("I was expecting an {image} object but got None.")
    
    if (transforms is None):
        raise ValueError("I was expecting a {transforms} object but got None.")

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.flatten()

    for i in range(times):
        augmentedImage = transforms(image)
        augmentedImage = augmentedImage.permute(1,  2, 0)

        axes[i].imshow(augmentedImage)
        axes[i].set_xlabel(f"Augmented #{i+1}", fontsize=10, color=random.choice(['red', 'green']))

    plt.tight_layout()
    plt.show()



def plotTestResults(images: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor, classToIndex: dict[str, int], saveTo: str = None):
    # How many plots (images) will contain each graph
    plotsPerGraph: int = Config.MAX_PLOTS_PER_ROW * Config.MAX_ROWS_PER_GRAPH

    # How many graphs will be made
    totalGraphs: int = math.ceil(images.size()[0] / plotsPerGraph)

    print(totalGraphs)

    for i in range(totalGraphs):
        plotGrid(images[i * plotsPerGraph : (i + 1) * plotsPerGraph])
    
    
    # fig, axes = plt.subplots(3, 4, figsize=(10, 6))
    # axes = axes.flatten()

    # indexToClass: dict[int, str] = {v: k for k, v in classToIndex.items()}
    # names = ["Roza", "Mpezos", "Wtf", "Aleksandra", "Roza", "Mpezos", "Wtf", "Aleksandra", "Petros", "Makhs", "Takhs", "Lakhs"]

    # for i in range(12):
    #     fixedImage: torch.Tensor = images[i].permute(1, 2, 0)
    #     axes[i].imshow(fixedImage)
    #     axes[i].set_title(names[i])
    #     axes[i].set_xlabel(indexToClass[labels[i].item()], color='green' if labels[i].item() == predictions[i].item() else 'red')

    # plt.tight_layout()
    # plt.show()


def plotGrid(images: torch.Tensor, labels: torch.Tensor) -> None:
    '''
        Makes one plot for each image inside the given {images}, in a WxH grid where
            W = Config.MAX_PLOTS_PER_ROW 
            H = Config.MAX_ROWS_PER_GRAPH 

        If the images are not quite enough to fill the graph, then only a part
        of the graph is filled.

        If images are too many, then only WxH images will be displayed.
    '''
    maxImages: int = Config.MAX_PLOTS_PER_ROW * Config.MAX_ROWS_PER_GRAPH
    totalImages: int = images.size(0)
    
    fix, axes = plt.subplots(Config.MAX_PLOTS_PER_ROW, Config.MAX_ROWS_PER_GRAPH)
    axes = axes.flatten()
    
    for i in range(min(totalImages, maxImages)):
        fixedImage: torch.Tensor = images[i].permute(1, 2, 0)
        axes[i].imshow(fixedImage)

    plt.tight_layout()
    plt.show()