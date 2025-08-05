import torch.nn as nn

class ConfidenceScorer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  
        )

    def forward(self, x):
        return self.model(x)
