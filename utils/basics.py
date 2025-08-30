import torch
import matplotlib.pyplot as plt
from typing import Any


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