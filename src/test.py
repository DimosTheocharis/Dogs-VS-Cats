import torch
import torch.nn as nn
from src.cnn import ConvolutionalNeuralNetwork
from src.utils.data import loadData, extractDataFromLoader
from src.utils.basics import measureAccuracy, plotTestResults


def testModel(
    model: ConvolutionalNeuralNetwork,
    testLoader: torch.utils.data.DataLoader
):
    if (not model):
        print("No model provided!")
        return
    
    # Combine all batches of the test set into single tensors
    testImages, testLabels, testNames, testLocations = extractDataFromLoader(testLoader)
    
    with torch.no_grad():
        # Make predictions
        testPredictions = model(testImages).squeeze()

        # Transform raw logits to probabilities
        testPredictions = nn.functional.sigmoid(testPredictions)

        # Compute the model's accuracy in the test set
        testAccuracy = measureAccuracy(testPredictions, testLabels)

        print(f"Accuracy on test set: {testAccuracy:.4f}")

    # Convert predictions to binary (0 or 1) using a threshold of 0.5
    predictedClasses = testPredictions.clone()
    predictedClasses[predictedClasses >= 0.5] = 1
    predictedClasses[predictedClasses < 0.5] = 0

    plotTestResults(testImages, testLabels, testNames, testLocations, testPredictions, testLoader.dataset.dataset.class_to_idx)