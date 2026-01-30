from torch import nn

# Architecture from MNIST classification
class MNISTClassification(nn.Module):
    def __init__(self, activation, dropout_rate):
        super().__init__()

        layers = [
            # Input layer
            nn.Linear(784, 128),
            activation,
            nn.Dropout(p=dropout_rate)  
        ]

        # Hidden layers 
        for _ in range(8):
            layers.append(nn.Linear(128, 128))
            layers.append(activation)
            layers.append(nn.Dropout(p=dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(128, 10))
        layers.append(activation) 

        self.architecture = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        logits = self.architecture(x)

        return logits
    
# Architecture form MNIST autoencoder