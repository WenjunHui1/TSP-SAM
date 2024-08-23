import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3D, self).__init__()
        
        self.stride = stride
        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
        
class Resnet3D(nn.Module):
    def __init__(self):
        super(Resnet3D, self).__init__()
        self.inplanes = 64
        layers = [2, 2, 2, 2]
        
        self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Bottleneck3D, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(Bottleneck3D, 64, layers[1], stride=1)
        self.layer3 = self._make_layer(Bottleneck3D, 64, layers[2], stride=1)
        self.layer4 = self._make_layer(Bottleneck3D, 64, layers[3], stride=1)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) 
       
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = torch.mean(x, 2)
        return out
        
    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers
        
def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion

    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv3d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm3d(outplanes),
        )