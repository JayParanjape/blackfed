#adapted from https://github.com/milesial/Pytorch-UNet/blob/master/

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, use_sigmoid=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.dropout = nn.Dropout(0.5)
        self.use_sigmoid = use_sigmoid

    def forward(self, x0):
        x1 = self.inc(x0)
        x1 = self.dropout(x1)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.dropout(x)
        x = self.up3(x, x2)
        x = self.dropout(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        if self.use_sigmoid:
            logits = F.sigmoid(logits, dim=1)
        else:
            logits = F.softmax(logits, dim=1)
        return logits


class UNet_Client(nn.Module):
    def __init__(self, n_channels):
        super(UNet_Client, self).__init__()
        self.n_channels = n_channels

        self.inc = DoubleConv(n_channels, 64)

    def forward(self, x):
        out = self.inc(x)
        return out

class UNet_Server(nn.Module):
    def __init__(self, n_classes, bilinear=False):
        super(UNet_Server, self).__init__()
        self.n_classes = n_classes
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))


    def forward(self, x1):
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        sigmoid_logits = F.sigmoid(logits)
        return sigmoid_logits

class BlackSplit(nn.Module):
    def __init__(self, num_clients, n_channels, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.num_clients = num_clients
        self.clients = []
        self.server = UNet_Server(n_classes)
        for i in range(num_clients):
            self.clients.append(UNet_Client(n_channels))

    def forward(self, x, turn):
        assert turn<self.num_clients
        x1 = self.clients[turn](x)
        out = self.server(x1)
        return out


class NoLabel_UNet_Server(nn.Module):
    def __init__(self, start_channels, num_layers):
        super().__init__()
        self.start_channels = start_channels
        self.num_layers = num_layers
        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.encoder_layers.append(Down(self.start_channels*(2**(i)), self.start_channels*(2**(i+1))))
            self.decoder_layers.append(Up(self.start_channels*(2**(num_layers-i)), self.start_channels*(2**(num_layers-i-1))))

    def forward(self, x):
        #here, x is a BXCXHXW feature map
        xn = [x]
        for i in range(len(self.encoder_layers)):
            if i==len(self.encoder_layers)-1:
                x_last = self.encoder_layers[i](xn[-1])
            else:
                xn.append(self.encoder_layers[i](xn[-1]))
        
        for i in range(len(self.decoder_layers)):
            if i==0:
                x_out = self.decoder_layers(x_last, xn[-1])
            else:
                x_out = self.decoder_layers(x_out, xn[-i-1])
        
        return x_out


class NoLabel_UNet_Client(nn.Module):
    def __init__(self, start_channels, num_layers, num_classes):
        super().__init__()
        self.start_channels = start_channels
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.encoder_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])

        self.inc = DoubleConv(3, self.start_channels)

        for i in range(num_layers):
            self.encoder_layers.append(Down(self.start_channels*(2**(i)), self.start_channels*(2**(i+1))))
            self.decoder_layers.append(Up(self.start_channels*(2**(num_layers-i)), self.start_channels*(2**(num_layers-i-1))))

        self.send_channels = self.start_channels*(2**(self.num_layers))
        self.outc = (OutConv(self.start_channels, self.num_classes))


    def forward(self, x, mode):
        #here, x is a BXCXHXW feature map
        #mode: ['encode', 'decode']
        x = self.inc(x)
        if mode=='encode':
            xn = [x]
            for i in range(len(self.encoder_layers)):
                if i==len(self.encoder_layers)-1:
                    x_last = self.encoder_layers[i](xn[-1])
                else:
                    xn.append(self.encoder_layers[i](xn[-1]))
            self.xn = xn
            return x_last
        
        else:        
            x_out = x
            for i in range(len(self.decoder_layers)):
                x_out = self.decoder_layers(x_out, xn[-i-1])
            
            logits = self.outc(x_out)
            sigmoid_logits = F.sigmoid(logits)
                    
            return sigmoid_logits
            
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
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)