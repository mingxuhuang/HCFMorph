import torch
import netron
from VMambaMorph.mambamorph.torch.TransMorph import CONFIGS as CONFIGS_TM
import VMambaMorph.mambamorph.torch.TransMorph as TransMorph

# 1. 初始化模型
config = CONFIGS_TM['MambaMorph']
config.img_size = (128, 128, 128)
config.base_resolution = [32, 32, 32]
model = TransMorph.MambaCMoFusionMorphFeat(config).cuda()
x1 = torch.randn(1, 1, 128, 128, 128).cuda()  # 第一个输入（如moving image）
x2 = torch.randn(1, 1, 128, 128, 128).cuda()  # 第二个输入（如fixed image）

a = model(x1, x2)

