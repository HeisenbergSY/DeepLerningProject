import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights

class VGG16Binary(nn.Module):
    def __init__(self):
        super(VGG16Binary, self).__init__()
        self.vgg16 = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        for param in self.vgg16.parameters():
            param.requires_grad = False
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.vgg16(x)
        return x

class ResNet50Binary(nn.Module):
    def __init__(self):
        super(ResNet50Binary, self).__init__()
        self.resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in self.resnet50.parameters():
            param.requires_grad = False
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.resnet50(x)
        return x

class MobileNetV3Binary(nn.Module):
    def __init__(self):
        super(MobileNetV3Binary, self).__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        self.mobilenet = mobilenet_v3_large(weights=weights)
        for param in self.mobilenet.parameters():
            param.requires_grad = True
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(self.mobilenet.classifier[0].in_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.mobilenet(x)
        return x
