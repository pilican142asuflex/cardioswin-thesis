import torch
import torch.nn as nn

class TemporalFusion(nn.Module):
    def __init__(self, feature_dim, mode="avg"):
        super().__init__()
        self.mode = mode
        
        if mode == "gru":
            self.gru = nn.GRU(feature_dim, feature_dim, batch_first=True)

    def forward(self, features):
        # features: (B, T, D)
        
        if self.mode == "avg":
            return torch.mean(features, dim=1)
        
        elif self.mode == "gru":
            _, h = self.gru(features)
            return h[-1]