
from src.architectures.base import BaseArchitecture
from src.cnn import LayerType, ActivationFunction
from src.config import Config

class MediumArchitecture(BaseArchitecture):
    def __init__(
        self,
        name="Medium Architecture",
        id=2,
        description="A medium sized architecture with 4 convolutional layers, 3 liean layers",
        layers=[
            # 3x128x128
            {
                "type": LayerType.Convolutional,
                "inChannels": 3,
                "outChannels": 16,
                "kernelSize": 3,
                "stride": 1,
                "padding": 1,
                "activationFunction": ActivationFunction.ReLU,
                "batchNorm": False
            },
            # 16x128x128
            {
                "type": LayerType.MaxPooling,
                "kernelSize": 2,
                "stride": 2
            },
            # 16x64x64
            {
                "type": LayerType.Convolutional,
                "inChannels": 16,
                "outChannels": 32,
                "kernelSize": 3,
                "stride": 1,
                "padding": 1,
                "activationFunction": ActivationFunction.ReLU,
                "batchNorm": False
            },
            # 32x64x64
            {   
                "type": LayerType.MaxPooling,
                "kernelSize": 2,
                "stride": 2
            },
            # 32x32x32
            {
                "type": LayerType.Convolutional,
                "inChannels": 32,
                "outChannels": 64,
                "kernelSize": 3,
                "stride": 1,
                "padding": 1,
                "activationFunction": ActivationFunction.ReLU,
                "batchNorm": False
            },
            # 64x32x32
            {   
                "type": LayerType.MaxPooling,
                "kernelSize": 2,
                "stride": 2
            },
            # 64x16x16
            {
                "type": LayerType.Convolutional,
                "inChannels": 64,
                "outChannels": 128,
                "kernelSize": 3,
                "stride": 1,
                "padding": 1,
                "activationFunction": ActivationFunction.ReLU,
                "batchNorm": False
            },
            # 128x16x16
            {   
                "type": LayerType.MaxPooling,
                "kernelSize": 2,
                "stride": 2
            },
            # 128x8x8
            {
                "type": LayerType.Flatten
            },
            {
                "type": LayerType.Linear,
                "inFeatures": 128 * (Config.IMAGE_SIZE // 16) * (Config.IMAGE_SIZE // 16),
                'outFeatures': 1024,
                "activationFunction": ActivationFunction,
                "batchNorm": False
            },
            {
                "type": LayerType.Linear,
                "inFeatures": 1024,
                'outFeatures': 128,
                "activationFunction": ActivationFunction,
                "batchNorm": False
            },
            {
                "type": LayerType.Linear,
                "inFeatures": 128,
                'outFeatures': 1,
                "activationFunction": None,
                "batchNorm": False
            }
        ]    
    ) -> None:
        super().__init__(
            name=name,
            id=id,
            description=description,
            layers=layers
        )