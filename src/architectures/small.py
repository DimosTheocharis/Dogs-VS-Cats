from src.cnn import LayerType, ActivationFunction
from src.config import Config
from src.architectures.base import BaseArchitecture


class SmallArchitecture(BaseArchitecture):
	'''
	The architecture consists of:
	- Convolutional layer with 16 filters, 3x3 kernel, ReLU activation
	- Max pooling layer with 8x8 kernel and stride of 2
	- Flatten layer
	- Output layer with 1 unit and no activation function
	'''
	def __init__(
		self, 
		name="Small Architecture",
		id=1,
		description="A small CNN architecture with a few layers for quick experimentation.",
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
				"inFeatures": 16 * (Config.IMAGE_SIZE // 8) * (Config.IMAGE_SIZE // 8),
				"outFeatures": 1,
				"activationFunction": None,
				"batchNorm": False
			}
		]
	) -> None:
		print("pw re file")
		super().__init__(
			name=name,
			id=id,
			description=description,
			layers=layers
		)
      