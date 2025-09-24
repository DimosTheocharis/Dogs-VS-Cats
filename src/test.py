import torch
import torch.nn as nn
from src.cnn import ConvolutionalNeuralNetwork
from src.utils.data import loadData, extractDataFromLoader
from src.utils.basics import measureAccuracy


def testModel(
    model: ConvolutionalNeuralNetwork,
    testLoader: torch.utils.data.DataLoader
):
    if (not model):
        print("No model provided!")
        return
    
    # Combine all batches of the test set into single tensors
    testImages, testLabels = extractDataFromLoader(testLoader)
    
    with torch.no_grad():
        testPredictions = model(testImages).squeeze()

        # Transform raw logits to probabilities
        testPredictions = nn.functional.sigmoid(testPredictions)

        # Compute the model's accuracy in the test set
        testAccuracy = measureAccuracy(testPredictions, testLabels)

        print(f"Accuracy on test set: {testAccuracy:.4f}")