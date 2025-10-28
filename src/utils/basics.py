import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from mpl_toolkits.axes_grid1 import ImageGrid
import random
import math
from src.config import Config


def displayImage(sample: np.ndarray | Image.Image | torch.Tensor):
    '''
        Creates an image from the given {sample} (1D or 3D) and then
        displays that image.

        Note: If sample is already a Pillow Image, the it just displays it
    '''
    image: Image.Image
    if (isinstance(sample, Image.Image)):
        image = sample
    elif (isinstance(sample, np.ndarray)):
        image: Image.Image = fromNdarrayToImage(sample)
    elif (isinstance(sample, torch.Tensor)):
        sample = sample.numpy()
        image = fromNdarrayToImage(sample)
    
    plt.imshow(image)
    plt.title(getattr(image, "title", "No title"))
    plt.show()

def fromNdarrayToImage(sample: np.ndarray) -> Image.Image:
    '''
        Creates and returns an Image.Image object from the given {sample} ndarray.
        If the given {sample} has 1 dimension, it transforms it to 3 dimensions (width, height, channels)

        Otherwise it keeps it as be.
    '''
    # Get a copy of the given {sample}
    data = sample.copy()

    # Check if {sample} is 1D, and if that's true transform it to 3D (width, height, channels)
    if (sample.ndim == 1):
        data = unflattenRGBImage(data, channelsFirst=False)
    elif (sample.ndim == 3):
        # Make sure that the format is (width, height, channels) and not (channels, width, height)
        if (sample.shape[1] == sample.shape[2]):
            data = data.reshape((sample.shape[1], sample.shape[2], sample.shape[0]))
        
    image: Image.Image = Image.fromarray(data, mode="RGB")
    
    return image


def unflattenRGBImage(image: np.ndarray, channelsFirst: bool = True) -> np.ndarray:
    '''
        Transforms the given (?,) array to (3,r,r) or (r,r,3) image,
        where r = sqrt(? / 3)

        @param channelsFirst: If true, the shape will be (3,r,r), otherwise it will be (r,r,3)

        for example transforms (3072,) to (3, 32, 32) or (32, 32, 3)
    '''
    if (image.ndim == 3):
        return image
    channelLength = int(image.shape[0] / 3)

    # Get the size of the image (size = width = height)
    r = int(math.sqrt(channelLength))

    red: np.ndarray = image[0 : channelLength * 1].reshape((r, r))
    green: np.ndarray = image[channelLength * 1 : channelLength * 2].reshape((r, r))
    blue: np.ndarray = image[channelLength * 2 : channelLength * 3].reshape((r, r))

    data: np.ndarray 
    if (channelsFirst):
        data = np.stack((red, green, blue), axis=0)
    else:
        data = np.stack((red, green, blue), axis=-1)

    return data


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



def plotTestResults(images: torch.Tensor, labels: torch.Tensor, names: list[str], locations: list[str], predictions: torch.Tensor, classToIndex: dict[str, int], saveTo: str = None):
    # How many plots (images) will contain each graph
    plotsPerGraph: int = Config.MAX_PLOTS_PER_ROW * Config.MAX_ROWS_PER_GRAPH

    # How many graphs will be made
    totalGraphs: int = math.ceil(images.size()[0] / plotsPerGraph)

    for i in range(totalGraphs):
        startIndex: int = i * plotsPerGraph
        endIndex: int = (i + 1) * plotsPerGraph
        plotGrid(
            images[startIndex : endIndex], 
            labels[startIndex : endIndex], 
            names[startIndex : endIndex], 
            locations[startIndex : endIndex],
            predictions[startIndex : endIndex], 
            classToIndex
        )


def plotGrid(images: torch.Tensor, labels: torch.Tensor, names: list[str], locations: list[str], predictions: torch.Tensor, classToIndex: dict[str, int]) -> None:
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
    # Make the dictionary {0: 'Cat', 1: 'Dog'}
    indexToClass: dict[int, str] = {v: k for k, v in classToIndex.items()}

    
    fig, axes = plt.subplots(Config.MAX_ROWS_PER_GRAPH, Config.MAX_PLOTS_PER_ROW, figsize=(Config.FIGURE_WIDTH, Config.FIGURE_HEIGHT), constrained_layout=True)
    axes = axes.flatten()

    
    for i in range(min(totalImages, maxImages)):
        # Apply transformations and set channels to HxWxC
        fixedImage: torch.Tensor = images[i].permute(1, 2, 0)

        # If prediction is 0.012 (cat) then the model is 98.8% that image is cat 
        certainty: float = max(predictions[i].item(), 1 - predictions[i].item())
        predictedClass: int = round(predictions[i].item())
        color: str = 'green' if labels[i].item() == predictedClass else 'red'


        axes[i].imshow(fixedImage)
        # Animal's name
        axes[i].set_title(names[i] , pad=Config.TITLE_PADDING, fontsize=Config.TITLE_FONT_SIZE)

        # Animal's location
        axes[i].text(Config.SUBTITLE_X, Config.SUBTITLE_Y, locations[i], transform=axes[i].transAxes, ha="center", va="bottom", fontsize=Config.SUBTITLE_FONT_SIZE, color="gray")
        

        axes[i].set_xlabel(f"{indexToClass[labels[i].item()]} ({certainty * 100:.2f}%) \n", color=color, labelpad=Config.LABEL_PADDING, fontsize=Config.LABEL_FONT_SIZE)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

        # Draw a border around the image with width=3. Border is green if prediction is correct, otherwise is red
        rect = patches.Rectangle(
            (-2, -2),                
            fixedImage.shape[0] + 4,          
            fixedImage.shape[1] + 4,          
            linewidth=3,
            edgecolor=color,
            facecolor='none',     
            clip_on =False
        )
        axes[i].add_patch(rect)

    fig.show()