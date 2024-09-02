import torch
import torch.nn as nn
import torch.distributions as dist
from torch import Tensor

class RKLDivLoss(nn.Module):
    __constants__ = ['reduction']

    def __init__(self, reduction: str = 'mean') -> None:
        super(RKLDivLoss, self).__init__()
        self.reduction = reduction

    def forward(self, global_mean: Tensor, global_std: Tensor, local_mean: Tensor, local_std: Tensor) -> Tensor:
        # 计算全局和本地的正态分布概率密度函数
        global_std = torch.clamp(global_std, min=1e-6)  # 防止标准差为零或负数
        local_std = torch.clamp(local_std, min=1e-6)

        # 创建正态分布
        global_dist = dist.Normal(global_mean, global_std)
        local_dist = dist.Normal(local_mean, local_std)

        # 计算KL散度
        kl_divergence = dist.kl_divergence(global_dist, local_dist)

        # 根据reduction参数应用不同的处理方式
        if self.reduction == 'mean':
            return kl_divergence.mean()
        elif self.reduction == 'sum':
            return kl_divergence.sum()
        elif self.reduction == 'batchmean':
            return kl_divergence.sum() / kl_divergence.size(0)  # 使用批次大小计算平均值
        else:  # 'none'
            return kl_divergence
