import random
import torch
import torch.nn as nn

from utils.basics import displayImage, measureAccuracy
from utils.data import loadData
from config import Config
from cnn import ConvolutionalNeuralNetwork, LayerType, ActivationFunction

random.seed(21)
torch.manual_seed(21)

trainLoader, validationLoader, testLoader = loadData(Config.DATASET_PATH, [0.2, 0.05, 0.75], batchSize=Config.BATCH_SIZE)

print(f"Number of training batches: {len(trainLoader)}")
print(f"Number of validation batches: {len(validationLoader)}")
print(f"Number of test batches: {len(testLoader)}")

validationImages = []
validationLabels = []
for images, labels in validationLoader:
    validationImages.append(images)
    validationLabels.append(labels)

validationImages = torch.cat(validationImages)
validationLabels = torch.cat(validationLabels)


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
    epochs=20,
    learningRate=0.001,
    batchSize=Config.BATCH_SIZE,
    weightDecay=0.0001
)

print(model)

######################################## TRAINING THE MODEL ########################################
train = True

if (train):
    # I will calculate the loss of my network's prediction with Binary Cross-Entropy Loss
    criterion: torch.nn.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()


    # Set Adam as optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model._learningRate, weight_decay=model._weightDecay)

    # For each epoch
    for epoch in range(model._epochs):
        model.train()
        trainingLosses: list[float] = []
        accuracies = []

        # For each batch in the training set
        for index, (batchImages, batchLabels) in enumerate(trainLoader):
            # Zero the gradients
            model.zero_grad()
            
            # Forward pass (remove the extra dimension)
            trainPredictions = model(batchImages).squeeze()


            # Compute the loss (also provide same data types)
            loss = criterion(trainPredictions, batchLabels.type_as(trainPredictions))
            trainingLosses.append(loss.item())



            # Compute the model's accuracy in this batch
            batchAccuracy = measureAccuracy(trainPredictions, batchLabels)
            accuracies.append(batchAccuracy)

            # Back propagation
            loss.backward()
            optimizer.step()
        
        trainingAccuracy = sum(accuracies) / len(accuracies)

        # Validate the model
        model.eval()
        with torch.no_grad():
            validationPredictions = nn.functional.sigmoid(model(validationImages)).squeeze()
            validationAccuracy = measureAccuracy(validationPredictions, validationLabels)


        print(f"Epoch [{epoch+1}/{model._epochs}], Training Loss: {sum(trainingLosses)/len(trainingLosses):.4f}, Training Accuracy: {trainingAccuracy:.4f}, Validation Accuracy: {validationAccuracy:.4f}")
        

