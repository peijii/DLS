import torch
import numpy as np
import torch.nn as nn


class DlsBlock(nn.Module):

    def __init__(self, num_classes, factor=3):
        super(DlsBlock, self).__init__()
        # factor = torch.tensor(factor, requires_grad=True)
        # self.factor = torch.nn.Parameter(factor)
        self.factor = factor

        self.linear1 = nn.Linear(in_features=num_classes, out_features=num_classes)
        self.linear2 = nn.Linear(in_features=num_classes, out_features=num_classes)
        self.bn1 = nn.BatchNorm1d(num_classes, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(num_classes, track_running_stats=False)
        # self.register_parameter("alpha", self.factor)
        self._init_weights()

    def forward(self, x: torch.Tensor, y) -> torch.Tensor:
        #print('factor parameter value: ', list(self.parameters())[0].data)
        out = self.linear1(x)
        out = self.bn1(out)

        out, out_ = self.custom(out, y, self.factor)
        out = torch.add(out, out_)

        out = self.linear2(out)
        out = self.bn2(out)

        out, out_ = self.custom(out, y, self.factor)
        out = torch.add(out, out_)

        assert all(torch.argmax(out, dim=1) == torch.argmax(y, dim=1))

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()

    def inverse(self, x):
        zero = torch.zeros_like(x)
        one = torch.ones_like(x)
        x = torch.where(x == 0, one, zero)
        return x

    def custom(self, x, y, factor):
        max_value = torch.max(torch.abs(x), dim=1)[0].unsqueeze(-1) * factor
        y_inverse = self.inverse(y)
        out_ = y * max_value
        out = y_inverse * x
        return out, out_


class LabelAdjustor(nn.Module):

    def __init__(self,
                 num_classes: int,
                 block = DlsBlock
                 ):
        super(LabelAdjustor, self).__init__()

        self.block1 = block(num_classes=num_classes, factor=3.0)
        self.block2 = block(num_classes=num_classes, factor=4.0)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):

        out = self.block1(x, y)
        out = self.block2(out, y)
        out = self.softmax(out)
        assert all(torch.argmax(out, dim=1) == torch.argmax(y, dim=1))

        return out


if __name__ == '__main__':

    # step 1: Data
    x = torch.randn(size=(32, 52))
    classes = 52
    batch_size = 32
    label = np.random.randint(0, classes, size=(batch_size, 1))
    true_label = torch.LongTensor(label)
    true_label = torch.zeros(batch_size, classes).scatter_(1, true_label, 1)

    # step 2: Model
    model = LabelAdjustor(classes=52)

    # step 3: loss function
    print(model(x, true_label).shape)


