import timm
import torch.nn as nn

class SwinBackbone(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

    def forward(self, x):
        return self.model(x)