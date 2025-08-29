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