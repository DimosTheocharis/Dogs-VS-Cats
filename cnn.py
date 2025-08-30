import torch
import torch.nn as nn
from typing import List, Dict
from enum import Enum

class LayerType(Enum):
    Linear = "Linear"
    Convolutional = "Convolutional"
    MaxPooling = "MaxPooling"
    AveragePooling = "AveragePoooling"
    Flatten = "Flatten"
    Dropout = "Dropout"


class ActivationFunction(Enum):
    ReLU = "ReLU"
    Sigmoid = "Sigmoid"
    Tanh = "Tanh"
    Softplus = "Softplus"
    LeakyReLU = "LeakyReLU"

class ConvolutionalNeuralNetwork(nn.Module):
    '''
        The constructor expect values for the hyper-parameters:
        * epochs, default 100
        * learningRate, default 0.01
        * batchSize, default 128
        * weightDecay, default 0
        * and a list of layers:


        * Convolutional layer: {
            "type": LayerType, 
            "inChannels: int, 
            "outChannels": int, 
            "kernelSize": int, 
            "stride": int, 
            "padding": int, 
            "activationFunction": ActivationFunction, 
            "batchNorm": boolean
        }

        * Pooling layers: {
            "type": LayerType, 
            "kernelSize": int, 
            "stride": int
        }

        * Linear layers: {
            "type": LayerType, 
            "inFeatures": int, 
            "outFeatures": int, 
            "activationFunction": ActivationFunction
            "batchNorm": boolean
        }

        * Dropout layers: {
            "type" LayerType, 
            "rate": float
        }

        * Flatten layer: {"type: LayerType}
    '''
    def __init__(self,
            layers: List[Dict] = [],
            epochs: int = 100,
            learningRate: float = 0.01,
            batchSize: int = 128,
            weightDecay: float = 0
        ):
        
        super().__init__()
        
        self._epochs: int = epochs
        self._learningRate: float = learningRate
        self._batchSize: int | None = batchSize
        self._weightDecay: float = weightDecay
        self._flattenedOut: bool = False

        self._model: nn.Sequential = nn.Sequential()
        for layer in layers:
            if layer["type"] == LayerType.Convolutional:
                self._model.append(nn.Conv2d(layer["inChannels"], layer["outChannels"], layer["kernelSize"], layer["stride"], layer["padding"]))
                
                if "batchNorm" in layer and layer["batchNorm"] == True:
                    self._model.append(nn.BatchNorm2d(layer["outChannels"]))

                if layer["activationFunction"] == ActivationFunction.ReLU:
                    self._model.append(nn.ReLU())
                elif layer["activationFunction"] == ActivationFunction.Sigmoid:
                    self._model.append(nn.Sigmoid())
                elif layer["activationFunction"] == ActivationFunction.Softplus:
                    self._model.append(nn.Softplus())
                elif layer["activationFunction"] == ActivationFunction.LeakyReLU:   
                    self._model.append(nn.LeakyReLU())
                elif layer["activationFunction"] == ActivationFunction.Tanh:
                    self._model.append(nn.Tanh())

            elif layer["type"] == LayerType.MaxPooling:
                self._model.append(nn.MaxPool2d(layer["kernelSize"], stride=layer["stride"]))

            elif layer["type"] == LayerType.AveragePooling:
                self._model.append(nn.AvgPool2d(layer["kernelSize"], stride=layer["stride"]))

            elif layer["type"] == LayerType.Flatten:
                self._model.append(nn.Flatten())
                self._flattenedOut = True

            elif layer["type"] == LayerType.Dropout:
                if (self._flattenedOut):
                    self._model.append(nn.Dropout(layer["rate"]))
                else:
                    self._model.append(nn.Dropout2d(layer["rate"]))

            elif layer["type"] == LayerType.Linear:
                self._model.append(nn.Linear(layer["inFeatures"], layer["outFeatures"], bias=True, dtype=torch.float32))
                
                if "batchNorm" in layer and layer["batchNorm"] == True:
                    self._model.append(nn.BatchNorm1d(layer["outFeatures"]))

                if layer["activationFunction"] == ActivationFunction.ReLU:
                    self._model.append(nn.ReLU())
                elif layer["activationFunction"] == ActivationFunction.Sigmoid:
                    self._model.append(nn.Sigmoid())
                elif layer["activationFunction"] == ActivationFunction.Softplus:
                    self._model.append(nn.Softplus())
                elif layer["activationFunction"] == ActivationFunction.LeakyReLU:   
                    self._model.append(nn.LeakyReLU())
                elif layer["activationFunction"] == ActivationFunction.Tanh:
                    self._model.append(nn.Tanh())

    def forward(self, x: torch.Tensor):
        t = self._model(x)

        return t

    def __repr__(self):
        return super().__repr__() + f"\nEpochs = {self._epochs}" + f"\nLearning rate = {self._learningRate}" + f"\nBatch size = {self._batchSize}" + f"\nWeight decay = {self._weightDecay}"
    
    def oneLineDescription(self) -> str:
        return f"{self._epochs}e {self._learningRate}lr {self._batchSize}bs {self._weightDecay}wd"