import torch 
import torch.nn as nn 
from config import Config


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb))
    
    def forward(self, t):
        emb = t * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, 1)
        self.key = nn.Conv2d(in_channels, in_channels//8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
         
    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query(x).view(B, -1, H*W).permute(0,2,1)
        k = self.key(x).view(B, -1, H*W)
        v = self.value(x).view(B, -1, H*W)
        
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attn.permute(0,2,1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, groups=8):
        super().__init__()

        valid_groups = [g for g in range(1, min(groups, out_channels)+1) 
                       if out_channels % g == 0]
        self.groups = valid_groups[-1] if valid_groups else 1
        self.out_channels = out_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(self.groups, out_channels),
            nn.Mish()
        )
        
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(self.groups, out_channels),
            nn.Mish()
        )
        
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        if out_channels > 1:
            self.attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels//8, 1),
                nn.Mish(),
                nn.Conv2d(out_channels//8, out_channels, 1),
                nn.Sigmoid()
            )

    def forward(self, x, t_emb):
        residual = self.res_conv(x)
        
        x = self.conv1(x)
        t_emb = self.time_mlp(t_emb)[:, :, None, None]
        x = x + t_emb
        x = self.conv2(x)
        
        if self.out_channels > 1:
            attn = self.attn(x)
            x = x * attn
        
        return x + residual


class BasicScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = Config()
        
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(512),
            nn.Linear(512, 512),
            nn.Mish(),
            nn.Linear(512, 512)
        )
        
        self.down1 = ResidualBlock(1, 64, 512)
        self.down2 = ResidualBlock(64, 128, 512)
        self.down3 = ResidualBlock(128, 256, 512)
        
        self.mid1 = ResidualBlock(256, 256, 512)
        self.mid2 = SelfAttention(256)
        self.mid3 = ResidualBlock(256, 256, 512)
        
        self.up1 = ResidualBlock(256+256, 128, 512)
        self.up2 = ResidualBlock(128+128, 64, 512)
        self.up3 = ResidualBlock(64+64, 1, 512, groups=1)  # 最后一层指定groups=1
        
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t):

        t_emb = self.time_embed(t)
        
        x1 = self.down1(x, t_emb)                  # (B,64,28,28)
        x2 = self.down2(self.downsample(x1), t_emb) # (B,128,14,14)
        x3 = self.down3(self.downsample(x2), t_emb) # (B,256,7,7)

        x_mid = self.mid1(x3, t_emb)                 # (B,256,7,7)
        x_mid = self.mid2(x_mid)
        x_mid = self.mid3(x_mid, t_emb)

        x_up = torch.cat([x_mid, x3], dim=1)          # (B,256+256,7,7)
        x_up = self.up1(x_up, t_emb)                     # (B,128,7,7)
        
        x_up = self.upsample(x_up)                    # (B,128,14,14)
        x_up = torch.cat([x_up, x2], dim=1)          # (B,128+128,14,14)
        x_up = self.up2(x_up, t_emb)                     # (B,64,14,14)
        
        x_up = self.upsample(x_up)                       # (B,64,28,28)
        x_up = torch.cat([x_up, x1], dim=1)               # (B,64+64,28,28)
        x_up = self.up3(x_up, t_emb)                     # (B,1,28,28)

        return x_up
