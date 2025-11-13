import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from einops import rearrange


# 您可能需要安装einops: pip install einops
# 这是一个非常有用的张量操作库

# --- 辅助模块和函数 ---

class Mlp(nn.Module):
    """ 标准MLP模块，用于Swin Transformer的FFN部分 """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    将特征图划分为不重叠的窗口
    输入: x (B, D, H, W, C)
    输出: windows (num_windows*B, window_size, window_size, window_size, C)
    """
    B, D, H, W, C = x.shape
    d, h, w = window_size
    x = x.view(B, D // d, d, H // h, h, W // w, w, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, d, h, w, C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    将窗口合并回原始特征图
    输入: windows (num_windows*B, window_size, window_size, window_size, C)
    输出: x (B, D, H, W, C)
    """
    d, h, w = window_size
    x = windows.view(B, D // d, H // h, W // w, d, h, w, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


# --- 核心注意力模块 ---

class WindowCrossAttention3D(nn.Module):
    """ 3D窗口化交叉注意力 """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义Q和K,V的投影层
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)  # K和V一起投影效率更高

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, xa):
        """
        x: 查询 (Query) 的窗口, B*nW, N, C
        xa: 键 (Key) 和 值 (Value) 的窗口, B*nW, N, C
        N = window_size_d * window_size_h * window_size_w
        """
        B_w, N, C = x.shape

        # 从x生成Q
        q = self.q_proj(x).reshape(B_w, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # 从xa生成K, V
        kv = self.kv_proj(xa).reshape(B_w, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        # 使用 PyTorch 2.0+ 内置的高效注意力
        attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                     dropout_p=self.attn_drop.p if self.training else 0.0)

        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B_w, N, C)

        # 输出投影
        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


# --- 主模块 (可直接替换您原来的类) ---

class SwinCrossAttention3DBlock(nn.Module):
    """
    Swin Transformer 风格的3D交叉注意力模块。
    这个模块可以作为您之前 FullConv3DTransCrossAttention 的直接替代品。
    """

    def __init__(self, in_channels, num_heads=8, window_size=(8, 8, 8), mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=LayerNorm):
        super().__init__()
        self.dim = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1_x1 = norm_layer(self.dim)
        self.norm1_x2 = norm_layer(self.dim)

        self.attn = WindowCrossAttention3D(
            self.dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()

        self.norm2 = norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x1, x2):
        B, C, D, H, W = x1.shape
        shortcut = x1

        x1 = x1.permute(0, 2, 3, 4, 1)
        x2 = x2.permute(0, 2, 3, 4, 1)

        x1_norm = self.norm1_x1(x1)
        x2_norm = self.norm1_x2(x2)

        pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x1_padded = F.pad(x1_norm, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        x2_padded = F.pad(x2_norm, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, Dp, Hp, Wp, _ = x1_padded.shape

        x1_windows = window_partition(x1_padded, self.window_size)
        x2_windows = window_partition(x2_padded, self.window_size)

        # 【修正】在这里将窗口的空间维度展平为序列长度
        Wd, Wh, Ww = self.window_size
        x1_windows = x1_windows.view(-1, Wd * Wh * Ww, C)  # 展平成 [B*nW, N, C]
        x2_windows = x2_windows.view(-1, Wd * Wh * Ww, C)  # 展平成 [B*nW, N, C]

        attn_windows = self.attn(x1_windows, x2_windows)

        # 【修正】在 window_reverse 之前，需要将形状恢复为5D，以便正确合并
        attn_windows = attn_windows.view(-1, Wd, Wh, Ww, C)

        attn_output = window_reverse(attn_windows, self.window_size, B, Dp, Hp, Wp)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            attn_output = attn_output[:, :D, :H, :W, :]

        attn_output = attn_output.permute(0, 4, 1, 2, 3)

        x = shortcut + self.drop_path(attn_output)

        ffn_input = x.permute(0, 2, 3, 4, 1)
        ffn_output = self.mlp(self.norm2(ffn_input))
        ffn_output = ffn_output.permute(0, 4, 1, 2, 3)

        x = x + self.drop_path(ffn_output)

        return x


class ResolutionAwareSwinCrossAttention3D(nn.Module):
    """
    分辨率感知的Swin Cross Attention 3D
    自动调整输入分辨率以提高计算效率
    """

    def __init__(self, in_channels, num_heads=8, window_size=(8, 8, 8), mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., max_resolution=(64, 64, 64),
                 norm_layer=LayerNorm):
        super().__init__()
        self.dim = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.max_resolution = max_resolution  # (D_max, H_max, W_max)

        # 归一化层
        self.norm1_x1 = norm_layer(self.dim)
        self.norm1_x2 = norm_layer(self.dim)
        self.norm2 = norm_layer(self.dim)

        # 注意力模块
        self.attn = WindowCrossAttention3D(
            self.dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        # MLP
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim,
                       act_layer=nn.GELU, drop=drop)

        # 下采样/上采样层
        self.downsample = nn.AdaptiveAvgPool3d(max_resolution)

    def calculate_scale_factor(self, input_shape, target_shape):
        """计算缩放因子"""
        scale_factors = []
        for i in range(3):
            if input_shape[i] > target_shape[i]:
                scale_factors.append(input_shape[i] / target_shape[i])
            else:
                scale_factors.append(1.0)
        return scale_factors

    def adaptive_downsample(self, x, target_shape):
        """
        自适应下采样
        如果输入尺寸大于目标尺寸，则进行下采样
        """
        B, C, D, H, W = x.shape
        target_D, target_H, target_W = target_shape

        # 检查是否需要下采样
        need_downsample = (D > target_D) or (H > target_H) or (W > target_W)

        if need_downsample:
            # 计算实际的目标尺寸（保持宽高比）
            scale_D = min(1.0, target_D / D)
            scale_H = min(1.0, target_H / H)
            scale_W = min(1.0, target_W / W)

            # 使用最小的缩放比例保持宽高比
            scale = min(scale_D, scale_H, scale_W)
            new_D = max(int(D * scale), 1)
            new_H = max(int(H * scale), 1)
            new_W = max(int(W * scale), 1)

            x_down = F.adaptive_avg_pool3d(x, target_shape)
            return x_down, (new_D, new_H, new_W), (D, H, W)
        else:
            return x, (D, H, W), (D, H, W)

    def adaptive_upsample(self, x, original_shape, current_shape):
        """自适应上采样回原始尺寸"""
        if current_shape != original_shape:
            x_up = F.interpolate(x, size=original_shape,
                                 mode='trilinear', align_corners=False)
            return x_up
        else:
            return x

    def forward_with_resolution_control(self, x1, x2):
        """
        带分辨率控制的前向传播
        """
        B, C, D, H, W = x1.shape
        shortcut = x1.clone()

        # 1. 自适应下采样
        x1_down, down_shape, orig_shape = self.adaptive_downsample(x1, self.max_resolution)
        x2_down, _, _ = self.adaptive_downsample(x2, self.max_resolution)

        D_down, H_down, W_down = down_shape

        # 2. 在降分辨率空间进行Cross Attention
        x1_perm = x1_down.permute(0, 2, 3, 4, 1)
        x2_perm = x2_down.permute(0, 2, 3, 4, 1)

        x1_norm = self.norm1_x1(x1_perm)
        x2_norm = self.norm1_x2(x2_perm)

        # 窗口划分和注意力计算
        pad_d = (self.window_size[0] - D_down % self.window_size[0]) % self.window_size[0]
        pad_h = (self.window_size[1] - H_down % self.window_size[1]) % self.window_size[1]
        pad_w = (self.window_size[2] - W_down % self.window_size[2]) % self.window_size[2]

        x1_padded = F.pad(x1_norm, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        x2_padded = F.pad(x2_norm, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, Dp, Hp, Wp, _ = x1_padded.shape

        x1_windows = window_partition(x1_padded, self.window_size)
        x2_windows = window_partition(x2_padded, self.window_size)

        Wd, Wh, Ww = self.window_size
        x1_windows = x1_windows.view(-1, Wd * Wh * Ww, C)
        x2_windows = x2_windows.view(-1, Wd * Wh * Ww, C)

        attn_windows = self.attn(x1_windows, x2_windows)

        attn_windows = attn_windows.view(-1, Wd, Wh, Ww, C)
        attn_output = window_reverse(attn_windows, self.window_size, B, Dp, Hp, Wp)

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            attn_output = attn_output[:, :D_down, :H_down, :W_down, :]

        attn_output = attn_output.permute(0, 4, 1, 2, 3)
#######NonLocal 没有残差 和 MLP
        # 3. 残差连接
        x_down =  attn_output + x1_down #+ shortcut[:, :, :D_down, :H_down, :W_down]
#############################Nonlocal 给下变得内容注释了
        # # 4. MLP部分（也在降分辨率空间）
        ffn_input = x_down.permute(0, 2, 3, 4, 1)
        ffn_output = self.mlp(self.norm2(ffn_input))
        ffn_output = ffn_output.permute(0, 4, 1, 2, 3)

        x_down = x_down + ffn_output
##########################################
        # 5. 上采样回原始分辨率
        x_out = self.adaptive_upsample(x_down, orig_shape, down_shape)

        return x_out

    def forward(self, x1, x2):
        return self.forward_with_resolution_control(x1, x2)


if __name__ == '__main__':
    model = ResolutionAwareSwinCrossAttention3D(16,window_size=(4,8,8),max_resolution=[8,48,160])
    x1 = torch.randn((1,16,32,192,320))
    c = model(x1,x1)
    print(c.shape)