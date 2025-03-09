import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class PUPHead(nn.Module):
    def __init__(self, img_size=224, embed_dim=768, num_classes=150,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), align_corners=False, num_upsample_layers=4):
        super(PUPHead, self).__init__()
        # self.img_size = img_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.num_upsample_layers = num_upsample_layers

        self.norm = norm_layer(embed_dim)

        self.conv_0 = nn.Conv2d(embed_dim, 256, kernel_size=3, stride=1, padding=1)
        self.bn_0 = nn.BatchNorm2d(256)

        self.conv_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn_1 = nn.BatchNorm2d(256)

        self.conv_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn_2 = nn.BatchNorm2d(256)

        self.conv_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn_3 = nn.BatchNorm2d(256)

        self.conv_4 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)

    def forward(self, x, img_size):
        # Transformer outputs are (B, N, C). Reshape to (B, C, H, W)
        B, N, C = x.shape
        h = w = int(math.sqrt(N))
        x = x.permute(0, 2, 1).reshape(B, C, h, w)  # (B, C, H, W)
        x = self.norm(x.permute(0, 2, 3, 1))  # Apply normalization
        x = x.permute(0, 3, 1, 2)  # Back to (B, C, H, W)

        x = self.conv_0(x)
        x = self.bn_0(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=self.align_corners)

        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=self.align_corners)

        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.relu(x, inplace=True)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=self.align_corners)

        x = self.conv_3(x)
        x = self.bn_3(x)
        x = F.relu(x, inplace=True)
        x = self.conv_4(x)

        # Resize to target output resolution
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=self.align_corners)
        return x


class ViT_PUP(nn.Module):
    def __init__(self, cfg, vit):
        super(ViT_PUP, self).__init__()
        self.vit = vit
        self.decode_head = PUPHead(
            num_classes=cfg.DATA.NUMBER_CLASSES
        )

    def forward(self, x):
        features = self.vit(x)
        # features = features[:, 21:, :]
        seg_map = self.decode_head(features, 224)
        return seg_map
