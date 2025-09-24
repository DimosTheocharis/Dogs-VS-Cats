from src.architectures.base import BaseArchitecture
from src.cnn import LayerType, ActivationFunction
from src.config import Config

class DoubleBarrelRegularizedArchitecture(BaseArchitecture):
    def __init__(
        self,
        name="A series with double convolutional layers",
        id=4,
        description="""A CNN with 4 sets of consecutive convolutional layers where the first one doubles the channels and the second retains them.
            After each set, a dropout layer and a max pooling layer follow. In the end of the network there is a flatten layer with 8192 nodes.
            The last layers are linear droping the nodes, followed by dropout layers.
        """,
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
                "batchNorm": True
            },
            # 16x128x128
            {
                "type": LayerType.Convolutional,
                "inChannels": 16,
                "outChannels": 16,
                "kernelSize": 3,
                "stride": 1,
                "padding": 1,
                "activationFunction": ActivationFunction.ReLU,
                "batchNorm": True
            },
            # 16x128x128
            {
                "type": LayerType.Dropout,
                "rate": 0.2
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
                "batchNorm": True
            },
            # 32x64x64
            {
                "type": LayerType.Convolutional,
                "inChannels": 32,
                "outChannels": 32,
                "kernelSize": 3,
                "stride": 1,
                "padding": 1,
                "activationFunction": ActivationFunction.ReLU,
                "batchNorm": True
            },
            # 32x64x64
            {
                "type": LayerType.Dropout,
                "rate": 0.2
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
                "batchNorm": True
            },
            # 64x32x32
            {
                "type": LayerType.Convolutional,
                "inChannels": 64,
                "outChannels": 64,
                "kernelSize": 3,
                "stride": 1,
                "padding": 1,
                "activationFunction": ActivationFunction.ReLU,
                "batchNorm": True
            },
            # 64x32x32
            {
                "type": LayerType.Dropout,
                "rate": 0.2
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
                "batchNorm": True
            },
            # 128x16x16
            {
                "type": LayerType.Convolutional,
                "inChannels": 128,
                "outChannels": 128,
                "kernelSize": 3,
                "stride": 1,
                "padding": 1,
                "activationFunction": ActivationFunction.ReLU,
                "batchNorm": True
            },
            # 128x16x16
            {
                "type": LayerType.Dropout,
                "rate": 0.2
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
                "batchNorm": True
            },
            {
                "type": LayerType.Dropout,
                "rate": 0.35
            },
            {
                "type": LayerType.Linear,
                "inFeatures": 1024,
                'outFeatures': 128,
                "activationFunction": ActivationFunction,
                "batchNorm": True
            },
            {
                "type": LayerType.Dropout,
                "rate": 0.35
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