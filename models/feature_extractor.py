import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, backbone, projection_head=None):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone
        self.projection_head = projection_head

    def forward(self, x, return_projection=False):
        features = self.backbone(x)

        if return_projection and self.projection_head is not None:
            projections = self.projection_head(features)
            return features, projections

        return features
