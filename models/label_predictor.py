import torch
import torch.nn as nn


class LabelPredictor(nn.Module):

    def __init__(self, num_classes):
        super(LabelPredictor, self).__init__()
        # decision layers
        self.dc_bn1 = nn.BatchNorm1d(num_classes)
        self.dc_se1 = nn.SELU()

        self.dc_conv2 = nn.Conv1d(num_classes, 64, kernel_size=1)
        self.dc_bn2 = nn.BatchNorm1d(64)
        self.dc_se2 = nn.SELU()

        self.dc_conv3 = nn.Conv1d(64, num_classes, kernel_size=1)
        self.dc_bn3 = nn.BatchNorm1d(num_classes)

        self.adaptiveAvgPool1d_2 = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        out = self.dc_bn1(x)
        out = self.dc_se1(out)

        embedded_out = self.dc_conv2(out)
        out = self.dc_bn2(embedded_out)
        out = self.dc_se2(out)

        out = self.dc_conv3(out)
        out = self.dc_bn3(out)

        out = self.adaptiveAvgPool1d_2(out)
        out = torch.flatten(out, 1)

        return out, torch.max(embedded_out, dim=1).values


if __name__ == '__main__':
    x = torch.randn(size=(32, 52, 50))
    model = LabelPredictor(num_classes=52)
    res, tsne_data = model(x)
    print(res.shape)