import torch
import torch.nn as nn
import torchvision

class DeepLabv3(nn.Module):
    def __init__(self, n_channels, n_classes, use_sigmoid=False):
        super(DeepLabv3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.DeepLabv3_Client = DeepLabv3_Client(n_channels)
        self.DeepLabv3_Server = DeepLabv3_Server(n_channels, n_classes, use_sigmoid)

    def forward(self, x):
        out = self.DeepLabv3_Server(self.DeepLabv3_Client(x))
        return out

class DeepLabv3_Client(nn.Module):
    def __init__(self, n_channels):
        super(DeepLabv3_Client, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(3, n_channels)

    def forward(self, x):
        out = self.inc(x)
        return out
    
class DeepLabv3_Server(nn.Module):
    def __init__(self, n_channels, n_classes, use_sigmoid = False):
        super(DeepLabv3_Server, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.network = torchvision.models.segmentation.deeplabv3_resnet50(num_classes = self.n_classes)
        self.network.backbone.conv1 = nn.Identity()
        self.network.backbone.bn1 = nn.Identity()
        self.network.backbone.relu = nn.Identity()
        if use_sigmoid:
            self.soft = nn.Sigmoid()
        else:
            self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        out = self.network(x)['out']
        out = self.soft(out)

        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=7, padding=3, stride = 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def get_gflops(model, input_size):
    from ptflops import get_model_complexity_info
    import re
    macs, params = get_model_complexity_info(model, input_size, as_strings=True, print_per_layer_stat=False, verbose=False)
    flops = eval(re.findall(r'([\d.]+)', macs)[0])*2
    flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
    print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
    print('Number of parameters: {:<8}'.format(params))

if __name__ == '__main__':
    client = DeepLabv3_Client(n_channels=64)
    server = DeepLabv3_Server(n_channels=64, n_classes=32)
    super_model = DeepLabv3(n_channels=64, n_classes=32)
    dummy = torch.ones((1,3,256,256))
    client_out = client(dummy)
    print('Client GFLOPS: ')
    print(tuple(dummy.shape))
    get_gflops(client, tuple(dummy.shape[1:]))
    print('Server GFLOPS: ')
    get_gflops(server, tuple(client_out.shape[1:]))
    print('Super Model GFLOPS: ')
    get_gflops(super_model, tuple(dummy.shape[1:]))
    server_out = server(client_out)

    print(client_out.shape)
    print(server_out.shape)
    # print(server)