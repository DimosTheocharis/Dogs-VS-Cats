import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from time import time

from utils.basics import displayImage, measureAccuracy
from utils.data import loadData, extractDataFromLoader
from src.config import Config
from src.cnn import ConvolutionalNeuralNetwork, LayerType, ActivationFunction
from src.logger import Logger

random.seed(21)
torch.manual_seed(21)

trainLoader, validationLoader, testLoader = loadData(Config.DATASET_PATH, [Config.TRAINING_SPLIT, Config.VALIDATION_SPLIT, Config.TEST_SPLIT], batchSize=Config.BATCH_SIZE)

# Combine all batches of the validation set into single tensors
validationImages, validationLabels = extractDataFromLoader(validationLoader)


model = ConvolutionalNeuralNetwork(
    layers=[
      # 3x128x128
      {
        "type": LayerType.Convolutional,
        "inChannels": 3,
        "outChannels": 32,
        "kernelSize": 3,
        "stride": 1,
        "padding": 1,
        "activationFunction": ActivationFunction.ReLU,
        "batchNorm": False
      },
      # 32x128x128
      {
        "type": LayerType.MaxPooling,
        "kernelSize": 8,
        "stride": 8
      },
      # 32x32x32
      {
          "type": LayerType.Flatten
      },
      {
          "type": LayerType.Linear,
          "inFeatures": 32 * (Config.IMAGE_SIZE // 8) * (Config.IMAGE_SIZE // 8),
          "outFeatures": 1,
          "activationFunction": None,
          "batchNorm": False
      }
    ],
    epochs=5,
    learningRate=0.001,
    batchSize=Config.BATCH_SIZE,
    weightDecay=0.0001
)

logger = Logger("logs/firstModel", "trainingResults", appendTimestamp=True)

logger.logData([
    f"Model structure: {model}",
    "\n\n",
    f"\nTraining samples: {len(trainLoader.dataset)}",
    f"\nValidation samples: {len(validationLoader.dataset)}"
], printToConsole=True)


######################################## TRAINING THE MODEL ########################################
train = True



# if (train):
#     # I will calculate the loss of my network's prediction with Binary Cross-Entropy Loss
#     criterion: torch.nn.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()

#     # Set Adam as optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=model._learningRate, weight_decay=model._weightDecay)

#     trainingLosses: list[float] = []
#     validationLosses: list[float] = []
#     trainingAccuracies: list[float] = [] # How well the model performs each epoch on the training set
#     validationAccuracies: list[float] = [] # How well the model performs each epoch on unseen data (the validation set)

#     start = time()

#     # For each epoch
#     for epoch in range(model._epochs):
#         model.train()
#         batchTrainingLosses: list[float] = []
#         batchTrainingAccuracies = []

#         # For each batch in the training set
#         for index, (batchImages, batchLabels) in enumerate(trainLoader):
#             # Zero the gradients
#             model.zero_grad()
            
#             # Forward pass (remove the extra dimension)
#             trainingPredictions = model(batchImages).squeeze()

#             # Compute the loss in this batch (also provide same data types)
#             trainingLoss = criterion(trainingPredictions, batchLabels.type_as(trainingPredictions))
#             batchTrainingLosses.append(trainingLoss.item())

#             # Transform raw logits to probabilities
#             trainingPredictions = nn.functional.sigmoid(trainingPredictions)

#             # Compute the model's accuracy in this batch of training set
#             batchTrainingAccuracy = measureAccuracy(trainingPredictions, batchLabels)
#             batchTrainingAccuracies.append(batchTrainingAccuracy)

#             # Back propagation
#             trainingLoss.backward()
#             optimizer.step()
        
#         # Compute the model's accuracy in this epoch of training set
#         trainingAccuracy = sum(batchTrainingAccuracies) / len(batchTrainingAccuracies)
#         trainingAccuracies.append(trainingAccuracy)

#         trainingLoss = sum(batchTrainingLosses) / len(batchTrainingLosses)
#         trainingLosses.append(trainingLoss)

#         # Validate the model
#         model.eval()
#         with torch.no_grad():
#             validationPredictions = model(validationImages).squeeze()

#             # Compute the validation loss
#             validationLoss = criterion(validationPredictions, validationLabels.type_as(validationPredictions))
#             validationLosses.append(validationLoss.item())

#             # Transform raw logits to probabilities
#             validationPredictions = nn.functional.sigmoid(validationPredictions)

#             # Compute the model's accuracy in this epoch of training set
#             validationAccuracy = measureAccuracy(validationPredictions, validationLabels)
#             validationAccuracies.append(validationAccuracy)

#         # Log the results
#         logger.logData([
#             f"\nEpoch [{epoch+1}/{model._epochs}]",
#             f"\nTraining Loss: {trainingLoss:.4f}",
#             f"\nTraining Accuracy: {trainingAccuracy:.4f}",
#             f"\nValidation Accuracy: {validationAccuracy:.4f}"
#         ], printToConsole=True)


#     end = time()

#     print(f"Training duration: {end - start:.2f} seconds")

#     # Plot the training results
#     plt.plot(range(1, model._epochs + 1), trainingLosses, color='red', label='Training Loss')
#     plt.plot(range(1, model._epochs + 1), trainingAccuracies, color='blue', label='Training Accuracy')
#     plt.plot(range(1, model._epochs + 1), validationAccuracies, color='green', label='Validation Accuracy')
#     plt.plot(range(1, model._epochs + 1), validationLosses, color='orange', label='Validation Loss')
#     plt.legend()
#     plt.title('Training Loss over Epochs')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.show()

