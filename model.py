"""
Implementation of YOLOv3 architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
""" 
Information about architecture config:
Tuple is structured by (filters, kernel_size, stride) 
Every conv is a same convolution. 
List is structured by "B" indicating a residual block followed by the number of repeats
"S" is for scale prediction block and computing the yolo loss
"U" is for upsampling the feature map and concatenating with a previous layer
"""
config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point is Darknet-53
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

## 三層的大小
# torch.Size([2, 512, 13, 13])
# torch.Size([2, 256, 26, 26])
# torch.Size([2, 128, 52, 52])

#image=384
# torch.Size([2, 512, 12, 12])
# torch.Size([2, 256, 24, 24])
# torch.Size([2, 128, 48, 48])



class seghead(nn.Module):
    def __init__(self):
        super().__init__()

        self.seg=nn.Sequential(
            nn.Conv2d(896,500,1),
            nn.ReLU(),
            nn.BatchNorm2d(500),

            nn.Conv2d(500,300,3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(300),

            nn.Conv2d(300,250,3,padding=1),
            nn.ReLU(),
        )
        #150 ,52 ,52

    def forward(self,getlayer):
        l1 = F.interpolate(getlayer[0], size=(52, 52), mode='bilinear', align_corners=False)
        l2 = F.interpolate(getlayer[1], size=(52, 52), mode='bilinear', align_corners=False)
        feature=torch.cat((l1, l2,getlayer[2]), dim=1).to(torch.float32) #batch, 896,52,52

        return self.seg(feature)

    
class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

        self.segneedlayer=[]
        self.seg=seghead()
    def forward(self, x):
        ob_outputs = []  # for each scale
        route_connections = []
        self.segneedlayer=[]
        for layer in self.layers:
            if isinstance(layer, ScalePrediction):
                ob_outputs.append(layer(x))
                self.segneedlayer.append(x)
                continue

            x = layer(x)

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)

            elif isinstance(layer, nn.Upsample):
                x = torch.cat([x, route_connections[-1]], dim=1)
                route_connections.pop()

        # print(len(self.segneedlayer))
        # print(self.segneedlayer[0].shape)
        # print(self.segneedlayer[1].shape)
        # print(self.segneedlayer[2].shape)

        #seg head
        segout=self.seg(self.segneedlayer)
        return ob_outputs , segout

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels

        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats,))

            elif isinstance(module, str):
                if module == "S":
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes),
                    ]
                    in_channels = in_channels // 2

                elif module == "U":
                    layers.append(nn.Upsample(scale_factor=2),)
                    in_channels = in_channels * 3


        return layers


if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 384
    model = YOLOv3(num_classes=num_classes)
    # print(model)


    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    # assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    # assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    # assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")


    for name, param in model.named_parameters():
        if 'pred' in name:
            print(name)
