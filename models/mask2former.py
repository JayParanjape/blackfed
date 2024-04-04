import sys
sys.path.append("../mmsegmentation/configs")
from mmseg.apis import init_model
from torch import nn
import torch

class Mask2Former_Client(nn.Module):
    def __init__(self, n_channels):
        super(Mask2Former_Client, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(3, n_channels)

    def forward(self, x):
        out = self.inc(x)
        return out
    
class Mask2Former_Server(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Mask2Former_Server, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        updated_cfg = {
            'num_classes':n_classes
        }
        self.config_path = "../mmsegmentation/configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py"
        self.network = init_model(self.config_path, cfg_options = updated_cfg)
        self.network.data_preprocessor = nn.Identity()
        self.network.backbone.conv1 = nn.Identity()
        self.network.backbone.bn1 = nn.Identity()
        self.network.backbone.relu = nn.Identity()

    def forward(self, x):
        out = self.network(x)['out']

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


if __name__=='__main__':
    # config_path = "../mmsegmentation/configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py"
    # model = init_model(config_path)
    # model.backbone.conv1 = nn.Identity()
    # print(model)

    client = Mask2Former_Client(n_channels=64)
    server = Mask2Former_Server(n_channels=64, n_classes=20)
    print(server)
    dummy = torch.ones((32,3,256,256))
    client_out = client(dummy)
    print(client_out.shape)

    client_out = client_out.to('cuda:5')
    server = server.to('cuda:5')
    server_out = server(client_out)

    print(server_out.shape)
