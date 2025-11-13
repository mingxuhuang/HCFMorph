import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as nnf
import pdb

from einops import rearrange, repeat
from vmamba2d import SS2D
from functools import partial
from typing import Optional, Callable
from vmamba2d import VSSBlock


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (4 // 2) * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W , "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2,  :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2,  :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2,  :]  # B H/2 W/2 T/2 C
        x3 = x[:, 1::2, 1::2,  :]  # B H/2 W/2 T/2 C

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 4 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # import math
        x = self.expand(x)
        # print(x.shape)
        # B, L, C = x.shape
        # root = math.pow(L,1/3)

        # root = int(math.ceil(root))

        # x = x.view(B, root,root,root,C)

        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)

        return x


class PatchMerging2(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (4 // 2) * dim, bias=False)
        self.norm = norm_layer(4 * dim)

        self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        self.BN1 = nn.InstanceNorm2d(dim)
        self.BN2 = nn.InstanceNorm2d(dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.InstanceNorm2d(dim),
            nn.SELU()
        )

        self.pconv = nn.Sequential(
            nn.Conv2d(in_channels=4 * dim, out_channels=(4 // 2) * dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d((4 // 2) * dim))

    def forward(self, x, H, W, H_x):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0,  0, W % 2, 0, H % 2))
        H_x = self.BN1(self.conv3(H_x))
        L_x = self.BN2(x.permute(0, 3, 1, 2,))
        xl = self.conv((H_x + L_x)).permute(0, 2, 3,  1)

        x0 = x[:, 0::2, 0::2,  :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2,  :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2,  :]  # B H/2 W/2 T/2 C
        x3 = x[:, 1::2, 1::2,  :]  # B H/2 W/2 T/2 C

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 T/2 8*C

        xconv = self.pconv(x.permute(0, 3, 1, 2))

        x = xconv.permute(0, 2, 3, 1).view(B, -1, 2 * C)  # B H/2*W/2*T/2 8*C

        # x = self.norm(x)
        # x = self.reduction(x)

        return x, xconv, xl.permute(0, 3, 1, 2)


class PatchMergingConv(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim):
        super().__init__()

        self.conv3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        self.BN1 = nn.InstanceNorm2d(dim)
        self.BN2 = nn.InstanceNorm2d(dim)
        self.pconv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.InstanceNorm2d(dim),
            nn.SELU()
        )

    def forward(self, x, H, W,  H_x):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W , "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x_f = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x_f = nnf.pad(x_f, (0, 0, 0, W % 2, 0, H % 2))
        H_x = self.BN1(self.conv3(H_x))
        L_x = self.BN2(x_f.permute(0, 3, 1, 2))
        x_f = (H_x + L_x)

        xconv = self.pconv(x_f)
        x = xconv.permute(0, 2, 3, 1).view(B, -1, C)

        return x, xconv, xconv


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, downsample=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm, reduce_factor=4)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        B, C = x.shape[0], x.shape[-1]
        assert C == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        x_mamba = self.mamba(x_norm)
        x = x_mamba.type(x.dtype)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class MambaDownLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, downsample=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        # patch merging layer
        self.d = downsample
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm, reduce_factor=4)
        else:
            self.downsample = PatchMergingConv(dim=dim)

    def forward(self, x, H, W, Hx):
        B, C = x.shape[0], x.shape[-1]
        assert C == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        x_mamba = self.mamba(x_norm)
        x = x_mamba.type(x.dtype)

        if self.d is not None:
            x_down, xconv, xl = self.downsample(x, H, W,  Hx)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W,  x_down, Wh, Ww,  xconv, xl
        else:
            x_down, xconv, xl = self.downsample(x, H, W, Hx)
            return x, H, W,  x, H, W,  xconv, xl


class VMambaDownLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, downsample=None, depths=2, imgsize=(32, 160, 160)):
        # dim, depths, d_state = 16, d_conv = 4, expand = 2, downsample = None, imgsize = (32, 160, 160)

        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0.,
                d_state=d_state,
                imgsize=imgsize
            )
            for i in range(depths)])
        #     Mamba(
        #     d_model=dim,  # Model dimension d_model
        #     d_state=d_state,  # SSM state expansion factor
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand,  # Block expansion factor
        # )
        # patch merging layer
        self.d = downsample
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm, reduce_factor=4)
        else:
            self.downsample = PatchMergingConv(dim=dim)

    def forward(self, x, H, W, Hx):
        B, C = x.shape[0], x.shape[-1]
        assert C == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        for blk in self.mamba:
            x_mamba = blk(x_norm)
        x = x_mamba.type(x.dtype)

        if self.d is not None:
            x_down, xconv, xl = self.downsample(x, H, W, Hx)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W,  x_down, Wh, Ww,  xconv, xl
        else:
            x_down, xconv, xl = self.downsample(x, H, W,  Hx)
            return x, H, W, x, H, W,  xconv, xl


class MambaUpLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        # patch merging layer
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels=dim * 3, out_channels=dim, kernel_size=1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.InstanceNorm2d(dim),
            nn.SELU()
        )

    def forward(self, x, Hx):
        Lx = self.Conv(torch.cat((self.upsample(x), Hx), 1))
        B, C, H, W = Lx.shape
        x_norm = Lx.view(B, C, -1).permute(0, 2, 1)
        assert C == self.dim
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        x_mamba = self.mamba(x_norm)
        x = x_mamba.type(x.dtype)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x += Lx
        return x


class VMambaLayer(nn.Module):
    def __init__(self, dim, depths, d_state=16, d_conv=4, expand=2, downsample=None,
                 imgsize=(32, 160, 160)):  # (32,192,320)
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # self.mamba = Mamba(
        #     d_model=dim,  # Model dimension d_model
        #     d_state=d_state,  # SSM state expansion factor
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand,  # Block expansion factor
        # )
        self.mamba = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0.,
                d_state=d_state,
                imgsize=imgsize
            )
            for i in range(depths)])
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=nn.LayerNorm, reduce_factor=4)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        B, C = x.shape[0], x.shape[-1]
        assert C == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        for blk in self.mamba:
            x_mamba = blk(x_norm)
        x = x_mamba.type(x.dtype)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W,  x_down, Wh, Ww
        else:
            return x, H, W,  x, H, W


class VMambaLayer_up(nn.Module):
    def __init__(self, dim, depths, d_state=16, d_conv=4, expand=2, upsample=PatchExpanding):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        # self.mamba = Mamba(
        #     d_model=dim,  # Model dimension d_model
        #     d_state=d_state,  # SSM state expansion factor
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand,  # Block expansion factor
        # )
        self.mamba = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=0.,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0.,
                d_state=d_state,
            )
            for i in range(depths)])
        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpanding(dim, norm_layer=nn.LayerNorm)
        else:
            self.upsample = None
        self.concat_back_dim = nn.Linear(int(dim), int(dim // 2))

    def forward(self, x, skip):
        x = x.permute(0, 2, 3,  1)
        skip = skip.permute(0, 2, 3,  1)
        if self.upsample is not None:
            x = self.upsample(x)

        x = torch.cat([x, skip], -1)
        B,  H, W, C = x.size()

        x = x.view(B,  H * W, C)

        for blk in self.mamba:
            x = blk(x)
        x = self.concat_back_dim(x)
        x = x.view(B, -1,  H, W)
        return x


class FullConv3DCrossAttention(nn.Module):
    def __init__(self, in_channels, base_resolution=[32,32]):
        super().__init__()
        self.in_channels = in_channels
        self.base_res = torch.tensor(base_resolution)
        self.patch_size = base_resolution

        # Q/K/V生成（保持C通道）
        self.query_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 注意力计算（深度可分离卷积）
        self.attn_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels * 2, kernel_size=3,
                      padding=1, groups=in_channels * 2),  # 深度卷积
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),  # 点卷积
            nn.Sigmoid()
        )

        # 输出处理（新增IN+SiLU）
        self.post_norm = nn.InstanceNorm2d(in_channels)
        self.act = nn.SiLU()

    def forward(self, x1, x2):
        B, C,  H, W = x1.shape
        current_res = torch.tensor([ H, W], device=x1.device)

        # 动态分Patch逻辑
        if torch.any(current_res > self.base_res.to(x1.device)):
            pH, pW = self.patch_size
            nH, nW =  H // pH, W // pW

            # 分Patch与均值池化
            def patch_reduce(x):
                return x.unfold(2, pH, pH).unfold(3, pW, pW).mean(( -2, -1))

            x1_patches = patch_reduce(x1)
            x2_patches = patch_reduce(x2)

            # 生成Q/K/V
            Q = self.query_conv(x1_patches)
            K = self.key_conv(x2_patches)
            V = self.value_conv(x2_patches)

            # 注意力计算
            QK = torch.cat([Q, K], dim=1)
            attn = self.attn_conv(QK)
            attended = V * attn

            # 恢复分辨率
            attended = attended.view(B, C, nH, nW, 1, 1).expand(-1,  -1, -1, -1,  pH, pW)
            attended = attended.permute(0, 1,  2, 4, 3, 5).reshape(B, C,  H, W)
        else:
            # 全局模式
            Q = self.query_conv(x1)
            K = self.key_conv(x2)
            V = self.value_conv(x2)

            QK = torch.cat([Q, K], dim=1)
            attn = self.attn_conv(QK)
            attended = V * attn

        # 新增后处理
        out = self.post_norm(attended)
        out = self.act(out)
        return out


class DoubleConvBottleneck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 第一层瓶颈
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels),
            nn.SiLU()
        )

        # 第二层瓶颈
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.InstanceNorm2d(in_channels),
            nn.SiLU()
        )

        # 快捷连接
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.block1(x) + residual
        residual = x
        x = self.block2(x) + residual
        return x


class DualPathFusion(nn.Module):
    def __init__(self, in_channels, base_resolution=[ 48, 80]):
        super().__init__()
        # 路1: 拼接+双瓶颈
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            DoubleConvBottleneck(in_channels)
        )

        # 路2: Cross-Attention
        self.path2 = FullConv3DCrossAttention(in_channels, base_resolution=base_resolution)

        # 自适应融合
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

    def forward(self, x1, x2):
        # 路1处理
        concat = torch.cat([x1, x2], dim=1)
        f1 = self.path1(concat)

        # 路2处理
        f2 = self.path2(x1, x2)

        # 动态权重融合
        weights = self.fusion(torch.cat([f1, f2], dim=1))
        w1, w2 = weights[:, 0:1], weights[:, 1:2]
        return w1 * f1 + w2 * f2


# if __name__ == '__main__':
#     model = MambaUpLayer(64, 16, 4, 2).to('cuda')
#     Lx = torch.randn(1, 128, 8, 24, 80).to('cuda')
#     Hx = torch.randn(1, 64, 16, 48, 160).to('cuda')
#     # model = PatchExpanding(192).to('cuda')
#     # input1 = torch.randn(1, 384, 3, 3, 3).cuda()
#     # input2 = torch.randn(1, 192, 6, 6, 6).cuda()
#     out = model(Lx, Hx)
#     print(out.shape)