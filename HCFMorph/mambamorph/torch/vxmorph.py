import torch
import torch.nn as nn
import torch.nn.functional as F


class VoxelMorph(nn.Module):
    """带特征提取接口和字典输出的VoxelMorph模型"""

    def __init__(self, feature_extractor=None, init_channels=16):
        """
        Args:
            feature_extractor: 外部定义的特征提取器，输入1通道图像，输出16通道特征
            init_channels: 配准网络初始通道数
        """
        super(VoxelMorph, self).__init__()

        # 特征提取器 (由外部提供)
        self.feature_extractor = feature_extractor
        if feature_extractor is None:
            inch = 2
        else:
            inch =32

        # U-Net部分 (输入通道改为32，因为要拼接固定和浮动图像的特征)
        self.encoder = UnetEncoder(in_channels=inch, init_channels=init_channels)
        self.decoder = UnetDecoder(init_channels=init_channels)

        # 位移场预测
        self.flow = nn.Conv3d(init_channels, 3, kernel_size=3, padding=1)

        # 初始化位移场权重为0，使训练开始时位移场很小
        self.flow.weight.data.normal_(0, 1e-5)
        self.flow.bias.data.zero_()

    def forward(self, fixed, moving, return_pos_flow=True, return_feature=False):
        """
        Args:
            fixed: 固定图像 [B, 1, D, H, W]
            moving: 浮动图像 [B, 1, D, H, W]
        Returns:
            dict: 包含两个键值:
                'moved_vol': 变形后的浮动图像 [B, 1, D, H, W]
                'pos_flow': 位移场 [B, 3, D, H, W]
        """
        # 提取特征 (如果提供了特征提取器)
        if self.feature_extractor is not None:
            fixed_feat = self.feature_extractor(fixed)  # [B, 16, D, H, W]
            moving_feat = self.feature_extractor(moving)  # [B, 16, D, H, W]
        else:
            # 如果没有提供特征提取器，直接使用原始图像
            fixed_feat = fixed
            moving_feat = moving

        # 将固定图像和浮动图像的特征拼接作为输入
        x = torch.cat([fixed_feat, moving_feat], dim=1)  # [B, 32, D, H, W]

        # U-Net编码器-解码器
        x1, x2, x3, x4 = self.encoder(x)
        y1 = self.decoder(x1, x2, x3, x4)

        # 预测位移场
        flow = self.flow(y1)

        # 调整位移场大小以匹配输入图像
        if flow.shape[2:] != fixed.shape[2:]:
            flow = F.interpolate(flow, size=fixed.shape[2:], mode='trilinear', align_corners=True)

        # 对浮动图像进行变形
        warped_moving = spatial_transform(moving, flow)

        return {'moved_vol': warped_moving, 'pos_flow': flow}


# 辅助组件保持不变
class UnetEncoder(nn.Module):
    """U-Net的编码器部分"""

    def __init__(self, in_channels=32, init_channels=16):
        super(UnetEncoder, self).__init__()
        self.channels = [init_channels * 2 ** i for i in range(4)]  # [16, 32, 64, 128]

        self.down1 = nn.Sequential(
            ConvBlock(in_channels, self.channels[0]),
            ConvBlock(self.channels[0], self.channels[0])
        )

        self.down2 = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(self.channels[0], self.channels[1]),
            ConvBlock(self.channels[1], self.channels[1])
        )

        self.down3 = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(self.channels[1], self.channels[2]),
            ConvBlock(self.channels[2], self.channels[2])
        )

        self.down4 = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(self.channels[2], self.channels[3]),
            ConvBlock(self.channels[3], self.channels[3])
        )

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return x1, x2, x3, x4


class UnetDecoder(nn.Module):
    """U-Net的解码器部分"""

    def __init__(self, init_channels=16):
        super(UnetDecoder, self).__init__()
        self.channels = [init_channels * 2 ** i for i in range(4)]  # [16, 32, 64, 128]

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvBlock(self.channels[3], self.channels[2]),
            ConvBlock(self.channels[2], self.channels[2])
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvBlock(self.channels[2] *2, self.channels[1]),
            ConvBlock(self.channels[1], self.channels[1])
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ConvBlock(self.channels[1] *2, self.channels[0]),
            ConvBlock(self.channels[0], self.channels[0])
        )

    def forward(self, x1, x2, x3, x4):
        y3 = self.up3(x4)
        y3 = torch.cat([y3, x3], dim=1)

        y2 = self.up2(y3)
        y2 = torch.cat([y2, x2], dim=1)

        y1 = self.up1(y2)
        return y1


class ConvBlock(nn.Module):
    """卷积块，包含卷积、实例归一化和LeakyReLU激活"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


def spatial_transform(source, flow):
    """使用位移场对源图像进行空间变换"""
    # 创建网格
    b, c, d, h, w = source.shape
    grid_d, grid_h, grid_w = torch.meshgrid(
        torch.linspace(-1, 1, d),
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w)
    )
    grid = torch.stack([grid_w, grid_h, grid_d], dim=3).float().to(source.device)  # (d, h, w, 3)
    grid = grid.unsqueeze(0).repeat(b, 1, 1, 1, 1)  # (b, d, h, w, 3)

    # 添加位移场
    new_grid = grid + flow.permute(0, 2, 3, 4, 1)

    # 采样
    warped_source = F.grid_sample(source, new_grid, align_corners=True, padding_mode='border')

    return warped_source


def gradient_loss(flow):
    """计算位移场的梯度损失，用于平滑正则化"""
    dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
    dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
    dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])

    return (torch.mean(dx) + torch.mean(dy) + torch.mean(dz)) / 3.0


def voxelmorph_loss(fixed, output_dict, lambda_param=0.01):
    """修改后的损失函数，接受字典输入"""
    warped_moving = output_dict['moved_vol']
    flow = output_dict['pos_flow']

    # 图像相似度损失 (MSE)
    similarity_loss = F.mse_loss(warped_moving, fixed)

    # 位移场平滑正则化
    smooth_loss = gradient_loss(flow)

    # 总损失
    total_loss = similarity_loss + lambda_param * smooth_loss

    return total_loss, similarity_loss, smooth_loss