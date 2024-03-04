import torch
import torch.nn as nn

from timm.models.layers import to_2tuple

# 没有任何作用
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=128, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

# Overlapping Cross-Attention Block
class OCAB(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = 1

# Hybrid Attention Block
class HAB(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = 0

# Residual Hybrid Attention Group
class RHAG(nn.Module):
    def __init__(self) -> None:
        super(RHAG, self).__init__()

    def forward(self, x):
        x = 1

class HAT(nn.Module):
    def __init__(self,input_channels=9, embed_channels=64, out_channels=1) -> None:
        super(HAT, self).__init__()

        # shallow feature extraction
        self.conv_fist = nn.Conv2d(input_channels, embed_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        convfirst = self.conv_fist(x)
        