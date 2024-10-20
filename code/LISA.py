import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.prelu = nn.PReLU(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class Linear_block(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = nn.Conv2d(in_c, out_channels=out_c, kernel_size=kernel, groups=groups, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class Depth_Wise(nn.Module):
     def __init__(self, in_c, out_c, residual = False, kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
        super(Depth_Wise, self).__init__()
        # Depth-wise
        self.conv = Conv_block(in_c, in_c, groups=in_c, kernel=kernel, padding=padding, stride=(1, 1))
        # Point-wise
        self.project = Linear_block(in_c, out_c, kernel=(1, 1), padding=(0, 0), stride=(1, 1))
        self.residual = residual
     def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output

class MSCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv0_1 = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim)
        self.conv0_2 = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim)

        self.conv1_1 = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim)
        self.conv1_2 = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim)

        self.conv2_1 = nn.Conv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim)
        self.conv2_2 = nn.Conv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)
        attn = attn + attn_0 + attn_1 + attn_2

        attn = self.conv3(attn)

        return attn * u

class Hamburger(nn.Module):
    def __init__(self, channels):
        super(Hamburger, self).__init__()
        cheese = channels // 2
        self.conv1 = nn.Conv2d(channels, cheese, kernel_size=1)
        self.conv2 = nn.Conv2d(cheese, cheese, kernel_size=1)
        self.conv3 = nn.Conv2d(cheese, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        return x + residual

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_msca=True):
        super(EncoderBlock, self).__init__()
        self.conv = Depth_Wise(in_channels, out_channels)
        self.msca = MSCA(out_channels) if use_msca else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = x + self.msca(x) 
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv = Depth_Wise(in_channels, out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class LISA(nn.Module):
    def __init__(self, num_classes=19):
        super(LISA, self).__init__()
        
        # Encoder
        self.enc1 = EncoderBlock(3, 32)
        self.enc2 = nn.Sequential(
            EncoderBlock(32, 64),
            EncoderBlock(64, 64)
        )
        self.enc3 = nn.Sequential(
            EncoderBlock(64, 142),
            EncoderBlock(142, 142)
        )
        self.enc4 = nn.Sequential(
            EncoderBlock(142, 320),
            EncoderBlock(320, 320),
            EncoderBlock(320, 320)
        )
        self.enc5 = EncoderBlock(320, 512)
    
        # Hamburger
        self.hamburger = Hamburger(512)
        
        # Decoder
        self.dec4 = DecoderBlock(512 + 320, 320)
        self.dec3 = DecoderBlock(320 + 142, 142)
        self.dec2 = DecoderBlock(142 + 64, 64)
        self.dec1 = DecoderBlock(64 + 32, 32)
        
        # Final classification
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        e5 = self.enc5(F.max_pool2d(e4, 2))
        
        e5 = self.hamburger(e5)
        
        d4 = self.dec4(e5, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        # Final classification
        out = self.final(d1)
        
        return out    


if __name__ == "__main__":
    model = LISA()
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params}")