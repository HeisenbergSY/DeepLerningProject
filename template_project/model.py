import torch
from torch import nn
from torchvision.models import mobilenet_v3_large

class MobileNetV3Binary(nn.Module):
    def __init__(self):
        super(MobileNetV3Binary, self).__init__()
        self.mobilenet = mobilenet_v3_large(pretrained=True)
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.sigmoid(x)
        return x