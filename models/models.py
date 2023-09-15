import torch
import torch.nn as nn
from models.feature_extractor import *
from models.label_adjustor import *
from models.label_predictor import *


class SupervisedNet(nn.Module):

    def __init__(self, feature_extractor, label_predictor):
        super(SupervisedNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.label_predictor = label_predictor

    def forward(self, x):
        label_predictor_input, _ = self.feature_extractor(x)
        predictor_label = self.label_predictor(label_predictor_input)
        return predictor_label


class AdjustedNet(nn.Module):

    def __init__(self, feature_extractor, label_adjustor):
        super(AdjustedNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.label_adjustor = label_adjustor

    def forward(self, x):
        _, label_adjustor_input = self.feature_extractor(x)
        adjustor_label = self.label_adjustor(label_adjustor_input)
        return adjustor_label
