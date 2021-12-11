import torch.nn as nn
import torch

# (S, Filters, Stride, Padding)
# [(S, Filters, Stride, Padding), ..more tuples.. , num_repeats_all_tuples]
# "M" --> maxpool
architecture_config = [
    (7, 64, 2, 3),
    'M',
    (3, 192, 1, 1),
    'M',
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    'M',
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    'M',
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.01)

    def forward(self, x):
        conv = self.conv(x)
        batch_norm = self.batchnorm(conv)
        return self.leakyrelu(batch_norm)



class Yolov1(nn.Module):
    # default 3 in_channels for RGB image input
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.arch = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.arch)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        darknet = self.darknet(x)
        flat = torch.flatten(darknet, start_dim=1)
        return self.fcs(flat)

    def _create_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels

        for a in arch:
            if type(a) == tuple:
                layers += [CNNBlock(
                    in_channels=in_channels,
                    out_channels=a[1],
                    kernel_size=a[0],
                    stride=a[2],
                    padding=a[3],
                )]

                in_channels = a[1] # update in_channels

            elif type(a) == str: # 'M'
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ]
            elif type(a) == list:
                conv1 = a[0] # tuple
                conv2 = a[1] # tuple
                repeat_conv = a[2] # int
                for _ in range(repeat_conv):
                    layers += [CNNBlock(
                        in_channels=in_channels,
                        out_channels=conv1[1], # maps to next in_channels
                        kernel_size=conv1[0],
                        stride=conv1[2],
                        padding=conv1[3],
                    )]

                    layers += [CNNBlock(
                        in_channels=conv1[1], # get from above out_channels
                        out_channels=conv2[1],
                        kernel_size=conv2[0],
                        stride=conv2[2],
                        padding=conv2[3],
                    )]

                    in_channels = conv2[1] # update in_channels

        return nn.Sequential(*layers)


    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        DIM = 496 # paper uses 4096 instead of 496
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, DIM),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(DIM, S * S * (C + (5 * B))), # (S, S, 30), C + 5*B = 30
        )


# def test(S, B, C):
#     model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
#     x = torch.randn((2, 3, 448, 448))
#     print(model(x).shape)


# if __name__ == '__main__':
#     S = 7
#     B = 2
#     C = 20
#
#     # output:
#     # torch.Size([2, 1470]) --> 7 * 7 * 30 = 1470
#     test(S, B,C)




