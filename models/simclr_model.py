import torch
import torch.nn as nn
import torchvision.models as models

class SimCLRModel(nn.Module):
    def __init__(self, base_model='resnet50', out_dim=128):
        super(SimCLRModel, self).__init__()
        self.out_dim = out_dim
        
        if base_model == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif base_model == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unknown base_model: {base_model}")
        
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )
    
    def forward(self, x):
        return self.backbone(x)
