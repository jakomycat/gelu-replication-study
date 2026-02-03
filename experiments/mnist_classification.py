import torch

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import MNISTClassification 
from src.data_loader import get_mnist_loaders
from src.train import train_model
# from src.visualization import plot_training_history

# Obtain MNIST loaders
train_loader, test_loader = get_mnist_loaders("data", batch_size=128)

# Select the device
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = "cpu"

