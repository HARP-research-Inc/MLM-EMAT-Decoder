import torch.nn as nn
import torch

class SaliencyModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
            nn.Sigmoid()  # Output in [0,1]
        )

    def forward(self, combined_vec):
        return self.model(combined_vec)
