import torch
import torch.nn as nn
from time import time
import os

from src.cnn import ConvolutionalNeuralNetwork
from src.utils.basics import measureAccuracy, saveGraph, getCurrentTimeRepresentation, saveModel
from src.logger import Logger

def runExperiment(
    experimentId: int,
    model: ConvolutionalNeuralNetwork,
    trainLoader: torch.utils.data.DataLoader,
    validationImages: torch.Tensor,
    validationLabels: torch.Tensor,
):
    experimentName = setupExperiment(experimentId)

    # Define the logger
    logger = Logger(f"experiments/{experimentName}", "logs", appendTimestamp=False)

    # Log the model structure and dataset information
    logger.logData([
        f"Model structure: {model}",
        "\n\n",
        f"\nTraining samples: {len(trainLoader.dataset)}",
        f"\nValidation samples: {len(validationImages)}"
    ], printToConsole=True)

    # I will calculate the loss of my network's prediction with Binary Cross-Entropy Loss
    criterion: torch.nn.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()

    # Set Adam as optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model._learningRate, weight_decay=model._weightDecay)


    trainingLosses: list[float] = []
    validationLosses: list[float] = []
    trainingAccuracies: list[float] = [] # How well the model performs each epoch on the training set
    validationAccuracies: list[float] = [] # How well the model performs each epoch on unseen data (the validation set)

    start = time()

    # For each epoch
    for epoch in range(model._epochs):
        print(epoch)
        model.train()
        batchTrainingLosses: list[float] = []
        batchTrainingAccuracies = []

        # For each batch in the training set
        for index, (batchImages, batchLabels) in enumerate(trainLoader):
            # Zero the gradients
            model.zero_grad()
            
            # Forward pass (remove the extra dimension)
            trainingPredictions = model(batchImages).squeeze()

            # Compute the loss in this batch (also provide same data types)
            trainingLoss = criterion(trainingPredictions, batchLabels.type_as(trainingPredictions))
            batchTrainingLosses.append(trainingLoss.item())

            # Transform raw logits to probabilities
            trainingPredictions = nn.functional.sigmoid(trainingPredictions)

            # Compute the model's accuracy in this batch of training set
            batchTrainingAccuracy = measureAccuracy(trainingPredictions, batchLabels)
            batchTrainingAccuracies.append(batchTrainingAccuracy)

            # Back propagation
            trainingLoss.backward()
            optimizer.step()
        
        # Compute the model's accuracy in this epoch of training set
        trainingAccuracy = sum(batchTrainingAccuracies) / len(batchTrainingAccuracies)
        trainingAccuracies.append(trainingAccuracy)

        trainingLoss = sum(batchTrainingLosses) / len(batchTrainingLosses)
        trainingLosses.append(trainingLoss)

        # Validate the model
        model.eval()
        with torch.no_grad():
            validationPredictions = model(validationImages).squeeze()

            # Compute the validation loss
            validationLoss = criterion(validationPredictions, validationLabels.type_as(validationPredictions))
            validationLosses.append(validationLoss.item())

            # Transform raw logits to probabilities
            validationPredictions = nn.functional.sigmoid(validationPredictions)

            # Compute the model's accuracy in this epoch of training set
            validationAccuracy = measureAccuracy(validationPredictions, validationLabels)
            validationAccuracies.append(validationAccuracy)

        # Log the results
        logger.logData([
            f"\nEpoch [{epoch+1}/{model._epochs}]",
            f"\nTraining Loss: {trainingLoss:.4f}",
            f"\nTraining Accuracy: {trainingAccuracy:.4f}",
            f"\nValidation Accuracy: {validationAccuracy:.4f}"
        ], printToConsole=True)


    end = time()

    logger.logData([f"\n\nTraining duration: {end - start:.2f} seconds"], printToConsole=True)

    # Plot the training results
    saveGraph(
        xValues=list(range(1, model._epochs + 1)),
        yValuesList=[trainingLosses, trainingAccuracies, validationAccuracies, validationLosses],
        colors=['red', 'blue', 'green', 'orange'],
        labels=['Training Loss', 'Training Accuracy', 'Validation Accuracy', 'Validation Loss'],
        title='Training Results over Epochs',
        xLabel='Epochs',
        yLabel='Value',
        fileName=f"experiments/{experimentName}/training.png"
    )


    # Save the trained model
    saveModel(model, f"experiments/{experimentName}/model.pth")



def setupExperiment(id: int, appendTimestamp: bool = True) -> str:
    '''
        Creates the experiment directory.

        @returns the experiment name.
    '''
    experimentName = f"experiment_{str(id).zfill(2)}"
    if (appendTimestamp):
        experimentName += " " + getCurrentTimeRepresentation()
    
    experimentFilePath = f"experiments/{experimentName}"

    if not os.path.exists(experimentFilePath):
        os.makedirs(experimentFilePath)
    
    return experimentName 
