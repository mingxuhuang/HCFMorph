import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as nnf


from einops import rearrange, repeat
from vmamba import SS2D
from functools import partial
from typing import Optional, Callable
from vmamba import VSSBlock
from crossattention import SwinCrossAttention3DBlock,ResolutionAwareSwinCrossAttention3D



class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8 // reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x, H, W, T):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = x[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = x[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = x[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = x[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = x[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = x[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 4*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        # import math
        x = self.expand(x)
        # print(x.shape)
        # B, L, C = x.shape
        # root = math.pow(L,1/3)
   
        # root = int(math.ceil(root))

        # x = x.view(B, root,root,root,C)

        B, H, W, D, C = x.shape
        x = rearrange(x, 'b h w d (p1 p2 p3 c)-> b (h p1) (w p2) (d p3)c', p1=2, p2=2, p3=2, c=C//8)
        x= self.norm(x)

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
        self.reduction = nn.Linear(8 * dim, (8 // reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)

        self.conv3 = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        self.BN1 = nn.InstanceNorm3d(dim)
        self.BN2 = nn.InstanceNorm3d(dim)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.InstanceNorm3d(dim),
            nn.SELU()
        )

        self.pconv = nn.Sequential(nn.Conv3d(in_channels=8 * dim, out_channels=(8 // reduce_factor) * dim, kernel_size=3, padding=1),
                                   nn.InstanceNorm3d((8 // reduce_factor) * dim))

    def forward(self, x, H, W, T,H_x):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))
        H_x = self.BN1(self.conv3(H_x))
        L_x = self.BN2(x.permute(0, 4, 1, 2,3))
        xl = self.conv((H_x+L_x)).permute(0, 2, 3,4, 1)


        x0 = xl[:, 0::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x1 = xl[:, 1::2, 0::2, 0::2, :]  # B H/2 W/2 T/2 C
        x2 = xl[:, 0::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x3 = xl[:, 0::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x4 = xl[:, 1::2, 1::2, 0::2, :]  # B H/2 W/2 T/2 C
        x5 = xl[:, 0::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x6 = xl[:, 1::2, 0::2, 1::2, :]  # B H/2 W/2 T/2 C
        x7 = xl[:, 1::2, 1::2, 1::2, :]  # B H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C

        xconv = self.pconv(x.permute(0, 4, 1, 2,3))

        x = xconv.permute(0, 2, 3,4, 1).view(B, -1, 2*C)  # B H/2*W/2*T/2 8*C

        # x = self.norm(x)
        # x = self.reduction(x)

        return x,xconv,xl.permute(0, 4, 1, 2,3)

class PatchMergingConv(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim):
        super().__init__()

        self.conv3 = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        self.BN1 = nn.InstanceNorm3d(dim)
        self.BN2 = nn.InstanceNorm3d(dim)
        self.pconv = nn.Sequential(
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.InstanceNorm3d(dim),
            nn.SELU()
        )

    def forward(self, x, H, W, T,H_x):
        """
        x: B, H*W*T, C
        """
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x_f = x.view(B, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x_f = nnf.pad(x_f, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))
        H_x = self.BN1(self.conv3(H_x))
        L_x = self.BN2(x_f.permute(0, 4, 1, 2,3))
        x_f = (H_x+L_x)


        xconv = self.pconv(x_f)
        x = xconv.permute(0, 2, 3, 4, 1).view(B, -1, C)

        return x,xconv,xconv

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

    def forward(self, x, H, W, T):
        B, C = x.shape[0], x.shape[-1]
        assert C == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        x_mamba = self.mamba(x_norm)
        x = x_mamba.type(x.dtype)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T


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
    def forward(self, x, H, W, T,Hx):
        B, C = x.shape[0], x.shape[-1]
        assert C == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        x_mamba = self.mamba(x_norm)
        x = x_mamba.type(x.dtype)

        if self.d is not None:
            x_down,xconv,xl = self.downsample(x, H, W, T,Hx)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt,xconv,xl
        else:
            x_down, xconv,xl = self.downsample(x, H, W, T, Hx)
            return x, H, W, T, x, H, W, T,xconv,xl


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

    def forward(self, x, H, W, T, Hx):
        B, C = x.shape[0], x.shape[-1]
        assert C == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        for blk in self.mamba:
            x_mamba = blk(x_norm)
        x = x_mamba.type(x.dtype)

        if self.d is not None:
            x_down, xconv, xl = self.downsample(x, H, W, T, Hx)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt, xconv, xl
        else:
            x_down, xconv, xl = self.downsample(x, H, W, T, Hx)
            return x, H, W, T, x, H, W, T, xconv, xl


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
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.Conv = nn.Sequential(
            nn.Conv3d(in_channels=dim*3, out_channels=dim, kernel_size=1),
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=1),
            nn.InstanceNorm3d(dim),
            nn.SELU()
        )
    def forward(self, x, Hx):

        Lx = self.Conv(torch.cat((self.upsample(x), Hx), 1))
        B,C,H,W,T =Lx.shape
        x_norm= Lx.view(B,C,-1).permute(0,2,1)
        assert C == self.dim
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        x_mamba = self.mamba(x_norm)
        x = x_mamba.type(x.dtype)
        x = x.permute(0, 2, 1).view(B, C,H,W,T)
        x+=Lx
        return x

class VMambaLayer(nn.Module):
    def __init__(self, dim, depths, d_state=16, d_conv=4, expand=2, downsample=None,imgsize=(32,160,160)):#(32,192,320)
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

    def forward(self, x, H, W, T):
        B, C = x.shape[0], x.shape[-1]
        assert C == self.dim
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        for blk in self.mamba:
            x_mamba = blk(x_norm)
        x = x_mamba.type(x.dtype)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            return x, H, W, T, x_down, Wh, Ww, Wt
        else:
            return x, H, W, T, x, H, W, T

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
        self.concat_back_dim = nn.Linear(int(dim),int(dim//2)) 
    def forward(self, x, skip):
        x = x.permute(0, 2, 3, 4, 1)
        skip = skip.permute(0, 2, 3, 4, 1)
        if self.upsample is not None:
            x = self.upsample(x)

        x = torch.cat([x,skip],-1)
        B,D,H,W,C = x.size()

        x= x.view(B, D*H*W, C)
    
        for blk in self.mamba:
            x = blk(x)
        x = self.concat_back_dim(x)
        x = x.view(B,-1,D,H,W)
        return x


class FullConv3DTransCrossAttention(nn.Module):
    def __init__(self, in_channels, base_resolution=[8, 48, 80], num_heads=8, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer("base_res", torch.tensor(base_resolution))
        self.patch_size = base_resolution
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        # 确保 PyTorch 版本支持
        assert hasattr(nnf, 'scaled_dot_product_attention'), \
            "This model requires PyTorch 2.0+ for memory-efficient attention. Please upgrade PyTorch."

        self.q_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.v_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # dropout 从原来的attn_weights后移到scaled_dot_product_attention的参数中
        self.dropout_p = dropout if self.training else 0.0

        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.BatchNorm3d(in_channels)
        self.act = nn.SiLU()
        self.ffn = nn.Sequential(
            nn.Conv3d(in_channels, in_channels * 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(in_channels * 4, in_channels, kernel_size=1),
        )

    def forward(self, x1, x2):
        B, C, D, H, W = x1.shape
        current_res = torch.tensor([D, H, W], device=x1.device)
        residual = x1

        q_proj = self.q_proj(x1)
        k_proj = self.k_proj(x2)
        v_proj = self.v_proj(x2)

        if torch.any(current_res > self.base_res):
            # --- 分片注意力模式 ---
            pD, pH, pW = self.patch_size
            assert D % pD == 0 and H % pH == 0 and W % pW == 0, \
                f"Input resolution {D, H, W} must be divisible by patch size {pD, pH, pW}"
            nD, nH, nW = D // pD, H // pH, W // pW
            N = nD * nH * nW
            P = pD * pH * pW

            def reshape_for_attention(x):
                # ... (这个函数保持不变)
                x_unfolded = x.unfold(2, pD, pD).unfold(3, pH, pH).unfold(4, pW, pW)
                x_permuted = x_unfolded.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
                x_reshaped = x_permuted.view(B, N, C, P)
                x_seq = x_reshaped.permute(0, 1, 3, 2).contiguous()
                return x_seq.view(B * N, P, C)

            Q = reshape_for_attention(q_proj)
            K = reshape_for_attention(k_proj)
            V = reshape_for_attention(v_proj)

            Q = self.norm1(Q)
            K = self.norm1(K)
            V = self.norm1(V)

            Q = Q.view(B * N, P, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B * N, P, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B * N, P, self.num_heads, self.head_dim).transpose(1, 2)

            # ############ 核心修改点 ############
            # 原来的内存密集型计算
            # attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5)
            # attn_weights = torch.softmax(attn_scores, dim=-1)
            # attn_weights = self.dropout(attn_weights)
            # attn_output = torch.matmul(attn_weights, V)

            # 使用内存高效的 scaled_dot_product_attention 替换
            # 它需要 [..., Seq_Len, Embed_Dim] 格式，我们已经是 [B*N*num_heads, P, head_dim]
            # 为了符合函数API，我们将B*N和num_heads合并
            q_bs_heads, k_bs_heads, v_bs_heads = [t.reshape(-1, P, self.head_dim) for t in (Q, K, V)]
            attn_output = nnf.scaled_dot_product_attention(
                q_bs_heads, k_bs_heads, v_bs_heads,
                dropout_p=self.dropout_p
            )
            # 恢复形状
            attn_output = attn_output.view(B * N, self.num_heads, P, self.head_dim)
            # ######################################

            attn_output = attn_output.transpose(1, 2).contiguous().view(B * N, P, C)

            # 还原形状 (这部分逻辑保持不变)
            attn_output = attn_output.view(B, N, P, C).permute(0, 1, 3, 2).contiguous()
            attn_output = attn_output.view(B, nD, nH, nW, C, pD, pH, pW)
            attn_output = attn_output.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
            attn_output = attn_output.view(B, C, D, H, W)

            attn_output = self.out_proj(attn_output)
        else:
            # 全局模式也可以使用内存高效注意力，逻辑类似
            # [B, C, D*H*W] -> [B, D*H*W, C]
            seq_len = D * H * W
            q_flat = q_proj.flatten(2).permute(0, 2, 1)
            k_flat = k_proj.flatten(2).permute(0, 2, 1)
            v_flat = v_proj.flatten(2).permute(0, 2, 1)

            q_norm = self.norm1(q_flat)
            k_norm = self.norm1(k_flat)
            v_norm = self.norm1(v_flat)

            q_heads = q_norm.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_heads = k_norm.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v_heads = v_norm.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # 同样使用高效注意力
            attn_output = nnf.scaled_dot_product_attention(
                q_heads, k_heads, v_heads,
                dropout_p=self.dropout_p
            )  # Output shape: [B, num_heads, seq_len, head_dim]

            attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, C)
            attn_output = self.out_proj(attn_output.permute(0, 2, 1).view(B, C, D, H, W))

        # 残差连接
        out = attn_output

        # Feed Forward Network
        out_flat = self.norm2(out)
        out_flat = self.act(out_flat)

        ffn_out = self.ffn(out_flat)
        out = out + ffn_out

        return out


class FullConv3DNCrossAttention(nn.Module):
    def __init__(self, in_channels, base_resolution=[8, 48, 80]):
        super().__init__()
        self.in_channels = in_channels
        self.base_res = torch.tensor(base_resolution)
        self.patch_size = base_resolution

        # Q/K/V生成（保持C通道）
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # 注意力计算（深度可分离卷积）
        self.attn_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels * 2, kernel_size=3,
                      padding=1, groups=in_channels * 2),  # 深度卷积
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),  # 点卷积
            nn.Sigmoid()
        )

        # 输出处理（新增IN+SiLU）
        self.post_norm = nn.InstanceNorm3d(in_channels)
        self.act = nn.SiLU()
        self.ffn = nn.Sequential(
            nn.InstanceNorm3d(in_channels), # 使用 InstanceNorm 稳定训练
            nn.SiLU(),
            nn.Conv3d(in_channels, in_channels * 4, kernel_size=1), # 升维
            nn.SiLU(),
            nn.Conv3d(in_channels * 4, in_channels, kernel_size=1)  # 降维
        )

    def forward(self, x1, x2):
        B, C, D, H, W = x1.shape
        # current_res = torch.tensor([D, H, W], device=x1.device)


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
        ffn_out = self.ffn(out)
        out = out + ffn_out
        return out

class FullConv3DCrossAttention(nn.Module):
    def __init__(self, in_channels, base_resolution=[8, 48, 80]):
        super().__init__()
        self.in_channels = in_channels
        self.base_res = torch.tensor(base_resolution)
        self.patch_size = base_resolution

        # Q/K/V生成（保持C通道）
        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        # 注意力计算（深度可分离卷积）
        self.attn_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels * 2, kernel_size=3,
                      padding=1, groups=in_channels * 2),  # 深度卷积
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),  # 点卷积
            nn.Sigmoid()
        )

        # 输出处理（新增IN+SiLU）
        self.post_norm = nn.InstanceNorm3d(in_channels)
        self.act = nn.SiLU()

    def forward(self, x1, x2):
        B, C, D, H, W = x1.shape
        current_res = torch.tensor([D, H, W], device=x1.device)

        # 动态分Patch逻辑
        if torch.any(current_res > self.base_res.to(x1.device)):
            pD, pH, pW = self.patch_size
            nD, nH, nW = D // pD, H // pH, W // pW

            # 分Patch与均值池化
            def patch_reduce(x):
                return x.unfold(2, pD, pD).unfold(3, pH, pH).unfold(4, pW, pW).mean((-3, -2, -1))

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
            attended = attended.view(B, C, nD, nH, nW, 1, 1, 1).expand(-1, -1, -1, -1, -1, pD, pH, pW)
            attended = attended.permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(B, C, D, H, W)
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


class CrossNonLocalBlockND(nn.Module):
    """
    Cross Attention NonLocal Block - 基于官方NonLocal代码改进
    支持3D/2D/1D，支持两个不同输入的特征图进行交叉注意力
    """

    def __init__(self,
                 in_channels_x,  # 输入x的通道数 (作为Query)
                 in_channels_context,  # 上下文context的通道数 (作为Key和Value)
                 inter_channels=None,  # 中间通道数
                 dimension=3,  # 维度: 3 for 3D, 2 for 2D, 1 for 1D
                 sub_sample=True,  # 是否下采样
                 bn_layer=True):  # 是否使用BatchNorm
        super(CrossNonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample
        self.in_channels_x = in_channels_x
        self.in_channels_context = in_channels_context
        self.inter_channels = inter_channels

        # 设置中间通道数
        if self.inter_channels is None:
            self.inter_channels = in_channels_x // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # 根据维度选择对应的操作
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=2)
            bn = nn.BatchNorm1d

        # Query卷积 - 来自输入x
        self.theta = conv_nd(in_channels=self.in_channels_x,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)

        # Key卷积 - 来自上下文context
        self.phi = conv_nd(in_channels=self.in_channels_context,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        # Value卷积 - 来自上下文context
        self.g = conv_nd(in_channels=self.in_channels_context,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        # 输出变换
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels_x,
                        kernel_size=1,
                        stride=1,
                        padding=0),
                bn(self.in_channels_x))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels_x,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        # 下采样
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, context):
        """
        Args:
            x: 输入特征图 [B, C_x, D, H, W] 作为Query
            context: 上下文特征图 [B, C_ctx, D, H, W] 作为Key和Value
        Returns:
            z: 增强后的特征图 [B, C_x, D, H, W]
        """
        batch_size = x.size(0)

        # 1. 生成Query (来自x)
        theta_x = self.theta(x)  # [B, C_inter, D, H, W]
        theta_x = theta_x.view(batch_size, self.inter_channels, -1)  # [B, C_inter, D*H*W]
        theta_x = theta_x.permute(0, 2, 1)  # [B, D*H*W, C_inter]

        # 2. 生成Key (来自context)
        phi_x = self.phi(context)  # [B, C_inter, D, H, W]
        phi_x = phi_x.view(batch_size, self.inter_channels, -1)  # [B, C_inter, D*H*W]

        # 3. 计算注意力权重
        f = torch.matmul(theta_x, phi_x)  # [B, D*H*W, D*H*W]
        f_div_C = nnf.softmax(f, dim=-1)  # 归一化注意力权重

        # 4. 生成Value (来自context)
        g_x = self.g(context)  # [B, C_inter, D, H, W]
        g_x = g_x.view(batch_size, self.inter_channels, -1)  # [B, C_inter, D*H*W]
        g_x = g_x.permute(0, 2, 1)  # [B, D*H*W, C_inter]

        # 5. 应用注意力权重
        y = torch.matmul(f_div_C, g_x)  # [B, D*H*W, C_inter]
        y = y.permute(0, 2, 1).contiguous()  # [B, C_inter, D*H*W]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # [B, C_inter, D, H, W]

        # 6. 输出变换 + 残差连接
        W_y = self.W(y)
        z = W_y + x  # 残差连接

        return z



class DoubleConvBottleneck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 第一层瓶颈
        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, kernel_size=1),
            nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.Conv3d(in_channels // 4, in_channels, kernel_size=1),
            nn.InstanceNorm3d(in_channels),
            nn.SiLU()
        )

        # 第二层瓶颈
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, kernel_size=1),
            nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.Conv3d(in_channels // 4, in_channels, kernel_size=1),
            nn.InstanceNorm3d(in_channels),
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
    def __init__(self, in_channels,base_resolution=[8,48,80]):
        super().__init__()
        # 路1: 拼接+双瓶颈
        self.path1 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),
            DoubleConvBottleneck(in_channels)
        )

        # 路2: Cross-Attention
        # self.path2 = FullConv3DNCrossAttention(in_channels,base_resolution=base_resolution)
        # self.path2 = FullConv3DTransCrossAttention(in_channels,base_resolution=base_resolution)
        # self.path2 = SwinCrossAttention3DBlock(in_channels,window_size=base_resolution)
        # self.path2 = FullConv3DCrossAttention(in_channels,base_resolution=base_resolution)
        # self.path2 = CrossNonLocalBlockND(in_channels,in_channels,dimension=3)
        self.path2 = ResolutionAwareSwinCrossAttention3D(in_channels,window_size=[2,6,6],max_resolution=base_resolution)#[2,6,10]，[2,4,4]
        # 自适应融合
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(in_channels // 2, 2, kernel_size=1),
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

if __name__ == '__main__':
    model = FullConv3DCrossAttention(16,base_resolution=[4,24,40])
    Lx = torch.randn(1,16,16,96,160)

    out = model(Lx,Lx)
    print(out.shape)