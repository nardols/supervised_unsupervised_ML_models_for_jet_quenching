# code inspered from https://github.com/GLAZERadr/Multi-Layer-Perceptron-Pytorch/blob/main/model/MultiLayerPerceptron.ipynb

import torch
import torch.nn as nn

# hidden size must be a list with 2 numerical elements
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_size[0]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size[1], 1) # binary classification or regression
        )

    def forward(self, x):
        return self.layers(x).squeeze(1)
    
