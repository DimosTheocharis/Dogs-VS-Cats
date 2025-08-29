import torchvision
from torch.utils.data import DataLoader
from utils.basics import displayImage

def loadData(folderPath: str):
    # Create the transform object
    transoforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)), # Resize images to 128x128 pixels
        torchvision.transforms.ToTensor() # Convert images to PyTorch tensors and rescale pixel values to [0, 1]
    ])

    # Load the dataset from the specified folder path
    dataset = torchvision.datasets.ImageFolder(
        root=folderPath,
        transform=transoforms
    )

    # Create a DataLoader to iterate through the dataset in batches
    dataLoader = DataLoader(dataset, batch_size=512, shuffle=True)

    firstBatch = next(iter(dataLoader))

    images, labels = firstBatch

    displayImage(images[0])

    print(f"Images batch shape: {images[0].shape}")
    print(f"Labels batch shape: {labels[0].shape}")

    # for images, labels in dataLoader:
    #     print(images.shape)
    #     print(labels.shape)