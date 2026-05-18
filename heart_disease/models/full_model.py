import torch
import torch.nn as nn
from .swin_backbone import SwinBackbone
from .temporal_fusion import TemporalFusion
from .classifier import ClassifierHead

class HeartDiseaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.backbone = SwinBackbone(
            config.MODEL_NAME,
            config.PRETRAINED
        )
        
        feature_dim = self.backbone.model.num_features
        
        self.temporal = TemporalFusion(
            feature_dim,
            config.TEMPORAL_MODE
        )
        
        self.classifier = ClassifierHead(
            feature_dim,
            config.NUM_CLASSES
        )

    def forward(self, x):
        
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)

    # Convert grayscale to 3-channel
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        features = self.backbone(x)
        features = features.view(B, T, -1)
        
        fused = self.temporal(features)
        logits = self.classifier(fused)
        
        return logits