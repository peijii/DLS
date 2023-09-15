import torch
import torch.nn as nn
from typing import List, Union, TypeVar, Tuple, Optional, Callable, Type, Any

T = TypeVar('T')


def convnxn(in_planes: int, out_planes: int, kernel_size: Union[T, Tuple[T]], stride: int = 1,
            groups: int = 1, dilation=1) -> nn.Conv1d:
    """nxn convolution and input size equals output size
    O = (I-K+2*P) / S + 1
    """
    if stride == 1:
        k = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding_size = int((k - 1) / 2)  # s = 1, to meet output size equals input size
        return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding_size,
                         dilation=dilation,
                         groups=groups, bias=False)
    elif stride == 2:
        k = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding_size = int((k - 2) / 2)  # s = 2, to meet output size equals input size // 2
        return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding_size,
                         dilation=dilation,
                         groups=groups, bias=False)
    else:
        raise Exception('No such stride, please select only 1 or 2 for stride value.')


# ===================== DWResBlock ==========================
# The DepthWise Residual Block is used to ensure we can train a
# deep network and also make the data flow more efficient
# input  : (eg. [batch_size, 10, 20])   groups=10, group_width=12, in_planes=36.
# output : [batch_size, 48, 500]         in_planes=48
# the input size and output size should be an integer multiple of groups
# ===================== DWResBlock ==========================

class DWResBlock(nn.Module):
    expansion: int = 3

    def __init__(
            self,
            in_planes: int,
            out_planes: int,
            kernel_size: Union[T, Tuple[T]],
            stride: int = 1,
            groups: int = 11,  # 10 sparse sEMG channel
            input_length: int = 20,  # window length
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(DWResBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.groups = groups
        self.input_length = input_length
        self.stride = stride
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.conv1x1_1 = convnxn(in_planes, out_planes, kernel_size=1, stride=1, groups=groups)
        self.bn1 = norm_layer(out_planes)
        self.conv3x3 = convnxn(out_planes, out_planes, kernel_size=3, stride=stride, groups=groups)
        self.bn2 = norm_layer(out_planes)
        self.conv1x1_2 = convnxn(out_planes, out_planes * self.expansion, kernel_size=1, stride=1, groups=groups)
        self.bn3 = norm_layer(out_planes * self.expansion)
        self.act = nn.SELU(inplace=True)
        self.dropout = nn.Dropout(0.05)
        if stride != 1 or in_planes != out_planes * self.expansion:
            self.downsample = nn.Sequential(
                convnxn(in_planes, out_planes * self.expansion, kernel_size=1, stride=stride, groups=groups),
                norm_layer(out_planes * self.expansion)
            )
        else:
            self.downsample = None

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        out = self.conv1x1_1(x)
        out = self.bn1(out)

        out = self.conv3x3(out)
        out = self.bn2(out)

        out = self.dropout(out)

        out = self.conv1x1_2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Sequential):
                for n in m:
                    if isinstance(n, nn.Conv1d):
                        nn.init.kaiming_normal_(n.weight.data)
                        if n.bias is not None:
                            n.bias.data.zero_()
                    elif isinstance(n, nn.BatchNorm1d):
                        n.weight.data.fill_(1)
                        n.bias.data.zero_()

    def get_flops(self):
        flops = 0.0
        # conv1x1_1
        flops += ((2 * (self.in_planes / self.groups) * 1 - 1) * (
                self.out_planes / self.groups) * self.input_length) * self.groups
        # conv3x3
        flops += ((2 * (self.out_planes / self.groups) * self.kernel_size - 1) * (
                self.out_planes / self.groups) * self.input_length) * self.groups

        # conv1x1_2
        flops += ((2 * (self.out_planes / self.groups) * 1 - 1) * (
                self.out_planes * self.expansion / self.groups) * self.input_length) * self.groups
        # downsample
        if self.stride != 1 or self.in_planes != self.out_planes * self.expansion:
            flops += ((2 * (self.in_planes / self.groups) * 1 - 1) * (
                    self.out_planes * self.expansion / self.groups) * self.input_length) * self.groups
        # identity add
        flops += self.input_length
        # relu
        flops += self.input_length * self.out_planes * self.expansion

        return flops

    def get_parameters(self):
        parameters = 0.0
        # conv1x1_1
        parameters += (self.in_planes / self.groups) * 1 * (self.out_planes / self.groups) * self.groups
        # bn1
        parameters += 2 * self.out_planes
        # conv3x3
        parameters += (self.in_planes / self.groups) * self.kernel_size * (self.out_planes / self.groups) * self.groups
        # bn2
        parameters += 2 * self.out_planes
        # conv1x1_2
        parameters += (self.out_planes / self.groups) * 1 * (
                self.out_planes * self.expansion / self.groups) * self.groups
        # bn3
        parameters += 2 * self.out_planes * self.expansion
        # downsample
        if self.stride != 1 or self.in_planes != self.out_planes * self.expansion:
            parameters += (self.in_planes / self.groups) * 1 * (
                    self.out_planes * self.expansion / self.groups) * self.groups
            parameters += 2 * self.out_planes * self.expansion

        return parameters

# ===================== GLConvBlock ===========================
# The Channel Integrate Block is used to extracted the features from
# every channels also features between channels
# input  : (eg. [batch_size, 36, 500])   groups=3, group_width=12
# output : [batch_size, 48, 500]
# After this block, groups+1, and this block only use once.
# ===================== GLConvBlock ==========================

class GLConvBlock1(nn.Module):

    def __init__(
            self,
            in_planes: int,
            rate: int = 1,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 10,
            input_length: int = 20,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            flag: bool = True,
    ):
        super(GLConvBlock1, self).__init__()
        self.in_planes = in_planes
        self.kernel_size = kernel_size
        self.groups = groups
        self.input_length = input_length
        self.rate = rate
        self.flag = flag
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.group_width = int(in_planes / self.groups)

        self.global_conv = convnxn(in_planes, self.group_width * self.rate, kernel_size=1, stride=stride, groups=1)
        self.local_conv = convnxn(in_planes, in_planes * self.rate, kernel_size=3, stride=stride, groups=groups)
        if self.flag:
            self.bn = norm_layer((self.group_width + in_planes) * self.rate)
            self.act = nn.SELU(inplace=True)

        self._init_weights()

    def forward(self, x):

        global_output = self.global_conv(x)
        local_output = self.local_conv(x)
        out = torch.cat((local_output, global_output), 1)
        if self.flag:
            out = self.bn(out)
            out = self.act(out)

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_flops(self):
        flops = 0.0
        # global conv
        flops += ((2 * self.in_planes * 1 * 1 - 1) * (
                self.group_width * self.rate) * self.input_length)
        # local conv
        flops += ((2 * (self.in_planes / self.groups) * self.kernel_size * 1 - 1) * (
                self.in_planes * self.rate / self.groups) * self.input_length) * self.groups
        if self.flag:
            # act
            flops += self.input_length * (self.group_width * self.rate + self.in_planes * self.rate)

        return flops

    def get_parameters(self):
        parameters = 0.0
        # global conv
        parameters += self.in_planes * self.group_width * self.rate * 1
        # local conv
        parameters += (self.in_planes / self.groups) * self.kernel_size * (self.in_planes * self.rate / self.groups) * self.groups
        if self.flag:
            # bn1
            parameters += 2 * (self.group_width + self.in_planes) * self.rate

        return parameters


class GLConvBlock2(nn.Module):

    def __init__(
            self,
            in_planes: int,
            rate: int = 1,
            stride: int = 1,
            kernel_size: int = 3,
            groups: int = 11,
            input_length: int = 20,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            flag: bool = True,
    ):
        super(GLConvBlock2, self).__init__()
        self.groups = groups
        self.input_length = input_length
        self.rate = rate
        self.kernel_size = kernel_size
        self.flag = flag

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.group_width = int(in_planes / self.groups)

        self.global_conv1 = convnxn(self.group_width * (self.groups - 1), int(self.group_width * self.rate), kernel_size=1, stride=stride, groups=1)
        self.global_conv2 = convnxn(int(self.group_width * (self.rate + 1)), int(self.group_width * self.rate), kernel_size=1, stride=stride, groups=1)
        self.local_conv = convnxn(int(self.group_width * (self.groups - 1)), int((self.group_width * (self.groups - 1)) * self.rate), kernel_size=self.kernel_size, stride=stride, groups=self.groups-1)
        if self.flag:
            self.bn = norm_layer(int((self.group_width * (self.groups - 1)) * self.rate + self.group_width * self.rate))
            self.act = nn.SELU(inplace=True)

        self._init_weights()

    def forward(self, x):
        local_data = x[:, :(self.groups - 1) * self.group_width, :]
        global_data = x[:, (self.groups - 1) * self.group_width:, :]

        global_output = self.global_conv1(local_data)
        global_output = torch.cat((global_data, global_output), 1)
        global_output = self.global_conv2(global_output)

        local_output = self.local_conv(local_data)

        output = torch.cat((local_output, global_output), 1)

        if self.flag:
            output = self.bn(output)
            output = self.act(output)

        return output

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_flops(self):
        flops = 0.0
        # global conv1
        flops += ((2 * self.group_width * (self.groups - 1) * 1 * 1 - 1) * (
                self.group_width * self.rate) * self.input_length)
        # global conv2
        flops += ((2 * (self.group_width * (self.rate + 1)) * 1 * 1 - 1) * (
                self.group_width * self.rate) * self.input_length)
        # local conv
        flops += (2 * ((self.group_width * (self.groups - 1)) / (self.groups - 1)) * 1 * self.kernel_size - 1) * ((
                (self.group_width * (self.groups - 1)) * self.rate) / (self.groups - 1)) * (self.groups - 1)
        if self.flag:
            # act
            flops += (self.group_width * (self.groups - 1)) * self.rate + self.group_width * self.rate

        return flops

    def get_parameters(self):
        parameters = 0.0
        # global conv1
        parameters += self.group_width * (self.groups - 1) * self.group_width * self.rate * 1
        # global conv2
        parameters += self.group_width * (self.rate + 1) * self.group_width * self.rate * 1
        # local conv
        parameters += self.group_width * (self.groups - 1) * (self.group_width * (self.groups - 1)) * self.rate / (self.groups-1) * self.kernel_size
        if self.flag:
            # bn1
            parameters += 2 * (self.group_width * (self.groups - 1)) * self.rate + self.group_width * self.rate
        return parameters


# ===================== DWInceptionBlock ==========================
# The inception block is used to extracted different scale feature
# of the input signal.
# input  : (eg. [batch_size, 48, 500])   groups=4, group_width=12, in_planes=48
# output : [batch_size, 96, 500]         out_planes=96
# the input size and output size should be an integer multiple of groups
# ===================== DWInceptionBlock ==========================
class GLInceptionBlock(nn.Module):
    expansion: int = 2

    def __init__(
            self,
            in_planes: int,
            groups: int = 11,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(GLInceptionBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.in_planes = in_planes
        self.groups = groups
        self.group_width = int(in_planes / groups)
        self.expansion = GLInceptionBlock.expansion

        self.branch1_1 = GLConvBlock2(in_planes, rate=self.expansion, kernel_size=5, stride=1, groups=self.groups, flag=False)
        self.branch1_2 = GLConvBlock2(in_planes * self.expansion, rate=1, kernel_size=5, stride=1, groups=self.groups, flag=True)

        self.branch2_1 = GLConvBlock2(in_planes, rate=self.expansion, kernel_size=11, stride=1, groups=self.groups, flag=False)
        self.branch2_2 = GLConvBlock2(in_planes * self.expansion, rate=1, kernel_size=11, stride=1, groups=self.groups, flag=True)

        self.branch3_1 = GLConvBlock2(in_planes, rate=self.expansion, kernel_size=21, stride=1, groups=self.groups, flag=False)
        self.branch3_2 = GLConvBlock2(in_planes * self.expansion, rate=1, kernel_size=21, stride=1, groups=self.groups, flag=True)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        branch1 = self.branch1_1(x)
        branch1_out = self.branch1_2(branch1)

        branch2 = self.branch2_1(x)
        branch2_out = self.branch2_2(branch2)

        branch3 = self.branch3_1(x)
        branch3_out = self.branch3_2(branch3)

        outputs = [
            torch.cat([branch1_out[:, int(i * self.group_width * self.expansion):int((i + 1) * self.group_width * self.expansion), :],
                       branch2_out[:, int(i * self.group_width * self.expansion):int((i + 1) * self.group_width * self.expansion), :],
                       branch3_out[:, int(i * self.group_width * self.expansion):int((i + 1) * self.group_width * self.expansion), :]], 1)
            for i in range(self.groups)]

        out = torch.cat(outputs, 1)

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_flops(self):
        # flops
        flops = 0.0
        # branch1
        flops += self.branch1_1.get_flops()
        flops += self.branch1_2.get_flops()
        # branch2
        flops += self.branch2_1.get_flops()
        flops += self.branch2_2.get_flops()
        # branch3
        flops += self.branch3_1.get_flops()
        flops += self.branch3_2.get_flops()

        return flops

    def get_parameters(self):
        # parameters
        parameters = 0.0
        # branch1
        parameters += self.branch1_1.get_parameters()
        parameters += self.branch1_2.get_parameters()
        # branch2
        parameters += self.branch2_1.get_parameters()
        parameters += self.branch2_2.get_parameters()
        # branch3
        parameters += self.branch3_1.get_parameters()
        parameters += self.branch3_2.get_parameters()

        return parameters


# ===================== DWCBAMBlock ==========================
# The squeeze excation block is used as a attention menchanism.
# input  : (eg. [batch_size, 48, 500])   groups=4, group_width=12, in_planes=48
# output : [batch_size, 48, 500]         out_planes=48
# the input size and output size should be an integer multiple of groups
# ===================== DWSEblock ==========================
class ChannelAttentionModule(nn.Module):

    def __init__(
            self,
            channels: int,
            reduction: int = 4,
            groups: int = 11,
            input_length: int = 20,
    ) -> None:
        super(ChannelAttentionModule, self).__init__()
        self.group_width = int(channels // groups)
        self.groups = groups
        self.reduction = reduction
        self.channels = channels
        self.input_length = input_length
        # average scale
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # max scale
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc_groupID = ['fc' + str(i + 1) for i in range(self.groups)]

        for groupID in self.fc_groupID:
            setattr(self, groupID, nn.Sequential(
                nn.Linear(self.group_width, self.group_width // reduction, bias=False, device='cuda'),
                nn.Linear(self.group_width // reduction, self.group_width, bias=False, device='cuda')
            ))

        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y1 = self.avg_pool(x).view(b, c)
        y2 = self.max_pool(x).view(b, c)
        avg_out = []
        max_out = []
        for id, attr in enumerate(self.fc_groupID):
            data1 = y1[:, id * self.group_width:id * self.group_width + self.group_width]
            data2 = y2[:, id * self.group_width:id * self.group_width + self.group_width]
            func = getattr(self, attr)
            avg_out.append(func(data1).view(b, self.group_width, 1))
            max_out.append(func(data2).view(b, self.group_width, 1))

        avg_out = torch.cat(avg_out, 1)
        max_out = torch.cat(max_out, 1)
        out = avg_out + max_out
        y = self.sigmoid(out)
        return x * y.expand_as(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)

            elif isinstance(m, nn.Sequential):
                for n in m:
                    if isinstance(n, nn.Linear):
                        nn.init.normal_(n.weight.data, 0, 1)

    def get_flops(self):
        # flops
        flops = 0.0
        # MaxPooling + AveragePooling
        flops += self.channels * self.input_length * 2
        # n group fc flops
        flops += (((2 * self.group_width - 1) * (self.group_width // self.reduction)) + (2 * (self.group_width // self.reduction) - 1) * self.group_width) * self.groups
        # sigmoid
        flops += self.group_width * self.groups * self.input_length * 4

        return flops

    def get_parameters(self):
        # parameters
        parameters = 0.0
        # n group fc flops
        parameters += (self.group_width * (self.group_width // self.reduction)) * 2 * self.groups

        return parameters


class SpatialAttentionModule(nn.Module):

    def __init__(
            self,
            channels: int,
            groups: int = 11,
            input_length: int = 20
    ) -> None:
        super(SpatialAttentionModule, self).__init__()
        self.group_width = int(channels // groups)
        self.groups = groups
        self.input_length = input_length
        self.spatialAttConv = convnxn(2 * self.groups, self.groups, kernel_size=3, groups=self.groups)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, l = x.size()
        out = []
        for id in range(self.groups):
            data = x[:, id * self.group_width:id * self.group_width + self.group_width]
            avg_out = torch.mean(data, dim=1, keepdim=True)
            max_out, _ = torch.max(data, dim=1, keepdim=True)
            out_ = torch.cat([avg_out, max_out], dim=1)
            out.append(out_)

        out = torch.cat(out, 1)
        out = self.spatialAttConv(out)
        out = self.sigmoid(out)

        out = torch.cat([out[:, i].unsqueeze(1).expand((b, self.group_width, l)) for i in range(self.groups)], 1)
        out = self.sigmoid(out)

        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_flops(self):
        # flops
        flops = 0.0
        # Max + Mean
        flops += self.group_width * self.input_length * 2 * self.groups
        # spatialAttConv
        flops += 2 * (2 * self.groups - 1) * 1 * 3 * self.groups / self.groups
        # sigmoid
        flops += self.group_width * self.groups * self.input_length * 4
        return flops

    def get_parameters(self):
        # parameters
        parameters = 0.0
        # spatial att conv
        parameters += (2 * self.groups / self.groups) * (self.groups / self.groups) * self.groups * 3
        return parameters


class DWCBAM(nn.Module):

    def __init__(
            self,
            in_planes: int,
            reduction: int,
    ) -> None:
        super(DWCBAM, self).__init__()

        self.channel_attention = ChannelAttentionModule(in_planes, reduction)
        self.spatial_attention = SpatialAttentionModule(in_planes)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out

    def get_flops(self):
        flops = 0.0
        # channel attention
        flops += self.channel_attention.get_flops()
        # spatial attention
        flops += self.spatial_attention.get_flops()
        return flops

    def get_parameters(self):
        parameters = 0.0
        # channel attention
        parameters += self.channel_attention.get_parameters()
        # spatial attention
        parameters += self.spatial_attention.get_parameters()
        return parameters


# =================== FeatureExtractor ======================
# The FeatureExtractor class is used as a backbone as to  
# extract the features of multichannel physiological signals
# =================== FeatureExtractor ======================
class FeatureExtrctor(nn.Module):

    def __init__(
            self,
            block1: GLInceptionBlock = GLInceptionBlock,
            block2: DWResBlock = DWResBlock,
            block3: DWCBAM = DWCBAM,
            layers: List[int] = [22, 33],
            num_classes: int = 52,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d

        self.groups = 11
        self.input_channel = 10
        self.conv1 = GLConvBlock1(in_planes=10, rate=1, kernel_size=3, flag=True)
        self.input_channel += 1
        self.layer1 = self._make_layers(block1, block2, block3, layers[0], 1, expansion=2)
        self.layer2 = self._make_layers(block1, block2, block3, layers[1], 1, expansion=2)
        self.adaptiveAvgPool1d = nn.AdaptiveAvgPool1d(50)
        self.feature_extractor_out = nn.Conv1d(layers[1] * block2.expansion, num_classes, kernel_size=1)

    def _make_layers(self, block1: GLInceptionBlock, block2: DWResBlock, block3: DWCBAM, planes: int, blocks: int, expansion):
        layers = []
        for i in range(blocks):
            block1.expansion = expansion
            layers.append(block1(in_planes=self.input_channel, groups=self.groups))
            self.input_channel = int(self.input_channel * block1.expansion * 3)
            layers.append(block2(in_planes=self.input_channel, out_planes=planes, kernel_size=3, stride=1, groups=self.groups))
            self.input_channel = planes * block2.expansion
            layers.append(block3(self.input_channel, reduction=4))

        return nn.Sequential(*layers)

    def _forward_imp(self, x: torch.Tensor):

        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.adaptiveAvgPool1d(out)
        feature_out = self.feature_extractor_out(out)

        return feature_out, torch.max(feature_out, dim=-1).values

    def forward(self, x: torch.Tensor):
        return self._forward_imp(x)
