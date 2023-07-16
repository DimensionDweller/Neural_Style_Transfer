import torch
from torchvision import models
from torch import nn

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]
        
    def forward(self, x):
        features = []
        for name, layer in self.model._modules.items():
            x = layer(x)
            if name in self.chosen_features:
                features.append(x)
        return features
