import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv


class SimConv(nn.Module):
    # Normal Conv with ReLU activation
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SimSPPF(nn.Module):
    # Simplified SPPF with ReLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2
        )

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class ASPP(nn.Module):
    # Atrous Spatial Pyramid Pooling
    def __init__(self, in_channel=512, out_channel=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(
            in_channel, out_channel, 3, 1, padding=6, dilation=6
        )
        self.atrous_block12 = nn.Conv2d(
            in_channel, out_channel, 3, 1, padding=12, dilation=12
        )
        self.atrous_block18 = nn.Conv2d(
            in_channel, out_channel, 3, 1, padding=18, dilation=18
        )
        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode="bilinear")
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(
            torch.cat(
                [
                    image_features,
                    atrous_block1,
                    atrous_block6,
                    atrous_block12,
                    atrous_block18,
                ],
                dim=1,
            )
        )
        return net


class BasicConv(nn.Module):

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            )
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=True) if relu else None
        else:
            self.conv = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
            self.bn = None
            self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):
    # Receptive Field Block
    def __init__(
        self,
        in_planes,
        out_planes,
        stride=1,
        scale=0.1,
        map_reduce=8,
        vision=1,
        groups=1,
    ):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(
                in_planes,
                inter_planes,
                kernel_size=1,
                stride=1,
                groups=groups,
                relu=False,
            ),
            BasicConv(
                inter_planes,
                2 * inter_planes,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=groups,
            ),
            BasicConv(
                2 * inter_planes,
                2 * inter_planes,
                kernel_size=3,
                stride=1,
                padding=vision,
                dilation=vision,
                relu=False,
                groups=groups,
            ),
        )
        self.branch1 = nn.Sequential(
            BasicConv(
                in_planes,
                inter_planes,
                kernel_size=1,
                stride=1,
                groups=groups,
                relu=False,
            ),
            BasicConv(
                inter_planes,
                2 * inter_planes,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                groups=groups,
            ),
            BasicConv(
                2 * inter_planes,
                2 * inter_planes,
                kernel_size=3,
                stride=1,
                padding=vision + 2,
                dilation=vision + 2,
                relu=False,
                groups=groups,
            ),
        )
        self.branch2 = nn.Sequential(
            BasicConv(
                in_planes,
                inter_planes,
                kernel_size=1,
                stride=1,
                groups=groups,
                relu=False,
            ),
            BasicConv(
                inter_planes,
                (inter_planes // 2) * 3,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=groups,
            ),
            BasicConv(
                (inter_planes // 2) * 3,
                2 * inter_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
            ),
            BasicConv(
                2 * inter_planes,
                2 * inter_planes,
                kernel_size=3,
                stride=1,
                padding=vision + 4,
                dilation=vision + 4,
                relu=False,
                groups=groups,
            ),
        )

        self.ConvLinear = BasicConv(
            6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False
        )
        self.shortcut = BasicConv(
            in_planes, out_planes, kernel_size=1, stride=stride, relu=False
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))


class SPPFCSPC(nn.Module):
    # https://yolov5.blog.csdn.net/article/details/126354660?spm=1001.2014.3001.5502
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1, x2, x3, self.m(x3)), 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))
