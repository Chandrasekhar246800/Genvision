import torch
import torch.nn as nn


# -----------------------------
# Double Convolution Block
# -----------------------------
class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.conv(x)


# -----------------------------
# Encoder Block
# -----------------------------
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DoubleConv(3, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

    def forward(self, x):

        x1 = self.conv1(x)

        x2 = self.conv2(self.pool(x1))

        x3 = self.conv3(self.pool(x2))

        x4 = self.conv4(self.pool(x3))

        x5 = self.bottleneck(self.pool(x4))

        return x1, x2, x3, x4, x5



# -----------------------------
# Attention Gate
# -----------------------------
class AttentionGate(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):

        g1 = self.W_g(g)

        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)

        psi = self.psi(psi)

        return x * psi



# -----------------------------
# Decoder Block
# -----------------------------
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = DoubleConv(128, 64)

    def forward(self, x5, x4, x3, x2, x1):

        d4 = self.up4(x5)
        x4 = self.att4(d4, x4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        x3 = self.att3(d3, x3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        x2 = self.att2(d2, x2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        x1 = self.att1(d1, x1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.dec1(d1)

        return d1



# -----------------------------
# Attention U-Net
# -----------------------------
class AttentionUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=3):
        super(AttentionUNet, self).__init__()

        self.encoder = Encoder()

        self.decoder = Decoder()

        self.final_conv = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        x1, x2, x3, x4, x5 = self.encoder(x)

        d1 = self.decoder(
            x5,
            x4,
            x3,
            x2,
            x1
        )

        output = self.final_conv(d1)

        return output