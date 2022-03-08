# -*- coding: UTF-8 -*-
"""
модель для определения типа комманды.

"""


import torch.nn as nn
from torchvision.models import resnet18


class ClassifierLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_size: int = 256, dropout: float = 0.25):
        super(ClassifierLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.classifier = nn.Sequential(nn.ReLU(),
                                        nn.Linear(in_features, hidden_size),
                                        nn.Dropout(p=dropout),
                                        nn.ReLU(),
                                        nn.Linear(hidden_size, out_features),
                                        )

    def forward(self, x):
        return self.classifier(x)


class FootballTeamClassifier(nn.Module):
    def __init__(self, classes_num: int, hidden_size=256):
        super(FootballTeamClassifier, self).__init__()

        self.classes_num = classes_num

        self.backbone = resnet18(pretrained=False, progress=False)
        self.backbone.fc = ClassifierLayer(self.backbone.fc.in_features, classes_num, hidden_size)

    def forward(self, x):
        return self.backbone(x)

