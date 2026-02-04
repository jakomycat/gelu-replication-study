import torch

from torch import nn
from torch.utils.data import random_split

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import MNISTClassification 
from src.data_loader import get_mnist_loaders
from src.train import train_model
from src.visualization import plot_mnist_classification

# Obtain MNIST loaders
valex_loader, trainex_loader, train_loader, test_loader = get_mnist_loaders("data", batch_size=128, valex_size=5000)

# Select the device
if torch.accelerator.is_available():
    device = torch.accelerator.current_accelerator()
else:
    device = "cpu"

# 
experiments_config = [
    {
        "name": "GELU within dropout",
        "activation": nn.GELU(approximate="tanh"),
        "dropout": 0.0
    },
    {
        "name": "GELU with dropout",
        "activation": nn.GELU(approximate="tanh"),
        "dropout": 0.5
    },
    {
        "name": "ELU within dropout",
        "activation": nn.ELU(alpha=1.0),
        "dropout": 0.0
    },
    {
        "name": "ELU with dropout",
        "activation": nn.ELU(alpha=1.0),
        "dropout": 0.5
    },
    {
        "name": "ReLU within dropout",
        "activation": nn.ReLU(),
        "dropout": 0.0
    },
    {
        "name": "ReLU with dropout",
        "activation": nn.ReLU(),
        "dropout": 0.5
    }
]

# Loop
all_history = {} # This to save all history train

for config in experiments_config:
    print(f"Experimento {config["name"]}...\n")

    # Learning rate tuning
    learning_rates = [1e-3, 1e-4, 1e-5]

    best_lr = None
    best_acc = 0.0
    
    for lr in learning_rates:
        # Model configuration
        model = MNISTClassification(activation=config["activation"], dropout_rate=config["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_function = torch.nn.CrossEntropyLoss()

        # Train to find best lr
        train_model(
            train_loader=trainex_loader,
            test_loader=test_loader,
            device=device,
            model=model,
            loss_function=loss_function,
            optimizer=optimizer,
            epochs=5
        )

        # Calculate accuracy
        model.eval()

        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in valex_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        acc = correct / total

        # Compare this with the best
        if acc > best_acc:
            best_acc = acc
            best_lr = lr
            
    print(f"El mejor learning rate para {config["name"]} es: {best_lr}")

    # Train model with best lr
    final_model = MNISTClassification(activation=config["activation"], dropout_rate=config["dropout"]).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_lr)
    loss_function = torch.nn.CrossEntropyLoss()

    history = train_model(
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        model=final_model,
        loss_function=loss_function,
        optimizer=optimizer,
        epochs=50
    )
    
    all_history[config["name"]] = history

# Plot results
current_script_path = os.path.dirname(os.path.abspath(__file__))

save_dir = os.path.join(current_script_path, "..", "results", "Figure-2")
save_dir = os.path.normpath(save_dir)

plot_mnist_classification(all_history, save_dir)