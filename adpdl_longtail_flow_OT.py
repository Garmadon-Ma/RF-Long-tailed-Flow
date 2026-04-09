import os
import sys
import glob
from pathlib import Path
import random
import pickle
import math
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import geotorch
HAS_GEOTORCH = True
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions


from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.covariance
from sklearn.covariance import EmpiricalCovariance

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入 GMFlow 操作库
HAS_GMFLOW_OPS = False
try:
    gmflow_root = project_root / "GMFlow-main" / "GMFlow-main"
    if gmflow_root.exists():
        sys.path.insert(0, str(gmflow_root))
    try:
        from lib.ops.gmflow_ops.gmflow_ops import (
            gm_to_sample, gm_to_mean, gm_mul_iso_gaussian, 
            iso_gaussian_mul_iso_gaussian, gm_to_iso_gaussian
        )
        HAS_GMFLOW_OPS = True
        print("Successfully imported GMFlow ops")
    except ImportError as e:
        # 如果没有编译的 CUDA 扩展，使用纯 PyTorch 实现
        HAS_GMFLOW_OPS = False
        print(f"Warning: GMFlow ops not available ({e}), using PyTorch fallback")
except Exception as e:
    HAS_GMFLOW_OPS = False
    print(f"Warning: Could not setup GMFlow ops ({e}), using PyTorch fallback")

from openood.utils import config
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators.metrics import compute_all_metrics
from openood.trainers.lr_scheduler import cosine_annealing

os.environ['PYTHONHASHSEED'] = str(0)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config_DPDL:
    def __init__(self):
        # 训练参数
        self.num_classes = 26
        self.feature_dim = 512
        self.epochs = 100
        self.batch_size = 128
        self.lr = 1e-1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        # DPDL参数

        
        # GMMFlow增强参数（原薛定谔桥参数，现用于GMMFlow）
        self.sb_sampling_batch_size = 1      # 采样批次大小
        self.sb_s_diagonal_init = 0.1        # 协方差矩阵对角线初始化值
        self.sb_is_diagonal = True           # 是否使用对角协方差矩阵
        self.sb_safe_t = 1e-2                # 时间采样安全边界，避免 t=1 的数值问题
        self.sb_epsilon = 0.5     # 熵正则化参数ε
        # self.sb_bridge_matching_weight = 0.1  # 已移除，只使用官方 transition loss
        # 原型参数
        self.max_protos_per_class = 8
        self.max_radius_degrees = 2
        
        # 路径参数
        self.save_dir = "results/dpdl_etf6"
        self.proto_allocation_save_path = os.path.join(self.save_dir, "proto_allocation.pkl")
        self.proto_allocation_load_path = self.proto_allocation_save_path
        self.config_files = [
            './configs/datasets/wifi/wifi_benchmark.yml',
            './configs/datasets/wifi/wifi_ood_benchmark.yml',
            './configs/networks/resnet18_32x32.yml',
            './configs/pipelines/test/test_ood.yml',
            './configs/preprocessors/base_preprocessor.yml',
            './configs/postprocessors/msp.yml',
        ]

class FlowOT(nn.Module):
    """
    FlowOT: 基于 FlowOT-main 的实际实现
    
    本实现严格参考了 FlowOT-main 的代码：
    - 参考文件: FlowOT-main/src/refinement.py
    - 参考文件: FlowOT-main/src/nets.py
    
    核心特性：
    1. 使用连续归一化流（CNF）通过 Neural ODE 实现流传输
    2. 计算 W2 距离（Wasserstein-2）作为最优传输的正则化项
    3. 损失函数：Logit_loss + gamma * W2_distance
    4. 适配 1D 特征向量到原型的传输
    
    FlowOT 方法：
    - 使用流网络学习特征 -> 原型的传输映射
    - 通过积分 ODE 求解器（Euler/RK4）计算流轨迹
    - 计算传输路径的 W2 距离作为正则化
    """
    def __init__(self, dim=2, n_potentials=5, epsilon=1, is_diagonal=True,
                 sampling_batch_size=1, S_diagonal_init=0.1, r_scale=1,
                 hidden_dim=256, num_blocks=4, T0=0.25, int_mtd='RK4', num_int_pts=3):
        super().__init__()
        self.is_diagonal = is_diagonal
        self.dim = dim
        self.n_potentials = n_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.sampling_batch_size = sampling_batch_size
        
        # FlowOT: 流网络参数
        self.num_blocks = num_blocks
        self.T0 = T0
        self.int_mtd = int_mtd  # 'RK4' 或 'Euler'
        self.num_int_pts = num_int_pts
        self.hidden_dim = hidden_dim
        
        # 时间区间：每个 block 的时间范围
        self.T_ls = [T0] * num_blocks
        
        # FlowOT: 原型位置和权重
        self.log_alpha = nn.Parameter(torch.log(torch.ones(n_potentials)/n_potentials))
        self.r = nn.Parameter(torch.randn(n_potentials, dim))  # 原型位置
        
        # FlowOT: 流网络（速度场网络）
        # 每个 block 一个流网络，学习特征到原型的传输
        self.flow_nets = nn.ModuleList([
            self._build_flow_net(dim, hidden_dim) for _ in range(num_blocks)
        ])
        
        # 协方差矩阵参数（用于计算 logit loss）
        self.S_log_diagonal_matrix = nn.Parameter(torch.log(S_diagonal_init*torch.ones(n_potentials, self.dim)))
        if not is_diagonal:
            self.S_rotation_matrix = nn.Parameter(
                torch.randn(n_potentials, self.dim, self.dim)
            )
            geotorch.orthogonal(self, "S_rotation_matrix")
    
    def _build_flow_net(self, dim, hidden_dim):
        """构建流网络（速度场网络）"""
        # 简单的 MLP 作为速度场
        # 输入: (x, t)，输出: velocity
        return nn.Sequential(
            nn.Linear(dim + 1, hidden_dim),  # +1 for time
            nn.Softplus(beta=1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(beta=1),
            nn.Linear(hidden_dim, dim)
        )
        
    def init_r_by_samples(self, samples):
        """初始化原型位置"""
        assert samples.shape[0] == self.r.shape[0]
        self.r.data = torch.clone(samples.to(self.r.device))
    
    def get_S(self):
        """获取协方差矩阵"""
        if self.is_diagonal:
            S = torch.exp(self.S_log_diagonal_matrix)
        else:
            S = (self.S_rotation_matrix*(torch.exp(self.S_log_diagonal_matrix))[:, None, :])@torch.permute(self.S_rotation_matrix, (0, 2, 1))
        return S
    
    def get_r(self):
        """获取原型位置"""
        return self.r
    
    def _reparam_t(self, block_idx):
        """重新参数化时间区间（参考 FlowOT）"""
        # 设置当前 block 的时间范围
        if block_idx == 0:
            self.Tk_1 = 0.0
        else:
            self.Tk_1 = sum(self.T_ls[:block_idx])
        self.Tk = sum(self.T_ls[:block_idx+1])
    
    def _l2_norm_sqr(self, x):
        """计算 L2 范数的平方（参考 FlowOT utils.py）"""
        if len(x.shape) > 2:
            return x.view(x.shape[0], -1).pow(2).sum(dim=1)
        else:
            return x.pow(2).sum(dim=1)
    
    def _ode_solver(self, flow_net, x, t_start, t_end, reverse=False, return_trajectory=False):
        """
        ODE 求解器（参考 FlowOT nets.py）
        使用 Euler 或 RK4 方法积分流网络
        
        优化：默认不保存轨迹，只返回最终结果，大幅提升速度
        """
        num_steps = self.num_int_pts + 1
        integration_times = torch.linspace(t_start, t_end, num_steps, device=x.device)
        if reverse:
            integration_times = torch.flip(integration_times, [0])
        
        x_current = x
        x_start = x  # 保存起始点用于 W2 计算
        
        if return_trajectory:
            trajectory = [x]
        
        for i in range(len(integration_times) - 1):
            t = integration_times[i]
            dt = integration_times[i+1] - integration_times[i]
            
            # 计算速度场
            # 将时间 t 广播到 batch 维度
            t_broadcast = t.expand(x_current.shape[0], 1)
            x_with_t = torch.cat([x_current, t_broadcast], dim=1)
            velocity = flow_net(x_with_t)
            
            if self.int_mtd == 'RK4':
                # RK4 方法（计算量大，但精度高）
                k1 = velocity
                k2 = flow_net(torch.cat([x_current + 0.5*dt*k1, (t+0.5*dt).expand(x_current.shape[0], 1)], dim=1))
                k3 = flow_net(torch.cat([x_current + 0.5*dt*k2, (t+0.5*dt).expand(x_current.shape[0], 1)], dim=1))
                k4 = flow_net(torch.cat([x_current + dt*k3, (t+dt).expand(x_current.shape[0], 1)], dim=1))
                x_current = x_current + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                # Euler 方法（更快）
                x_current = x_current + dt * velocity
            
            if return_trajectory:
                trajectory.append(x_current)
        
        if return_trajectory:
            return trajectory
        else:
            # 只返回最终结果和起始点（用于 W2 计算）
            return x_current, x_start
    
    def flow_features_to_protos(self, features, protos, return_W2=False):
        """
        FlowOT 核心方法：计算特征到原型的流传输
        参考: FlowOT-main/src/refinement.py 的 flow_P_Q
        
        优化版本：减少不必要的计算，提升训练速度
        
        Args:
            features: (bs, dim) - 输入特征
            protos: (n_protos, dim) - 目标原型
            return_W2: 是否返回 W2 距离
        
        Returns:
            如果 return_W2=True: (W2_distance, transported_features)
            如果 return_W2=False: transported_features
        """
        x_input = features  # 不需要 clone，直接使用
        W2_over_blocks = 0.0
        
        # 通过所有 blocks 进行流传输
        for block_idx in range(self.num_blocks):
            self._reparam_t(block_idx)
            flow_net = self.flow_nets[block_idx]
            
            # 前向流传输（不保存轨迹，只返回最终结果）
            x_start = x_input  # 保存起始点用于 W2 计算
            x_input, _ = self._ode_solver(flow_net, x_input, self.Tk_1, self.Tk, reverse=False, return_trajectory=False)
            
            # 计算 W2 距离（参考 FlowOT）
            # W2^2 = ||x_end - x_start||^2 / (T_end - T_start)
            if return_W2:
                W2_block = self._l2_norm_sqr(x_input - x_start).mean() / (self.Tk - self.Tk_1 + 1e-8)
                W2_over_blocks += W2_block
        
        # 将特征传输到最近的原型
        # 计算每个特征到所有原型的距离，选择最近的原型
        distances = torch.cdist(x_input, protos)  # (bs, n_protos)
        nearest_proto_idx = distances.argmin(dim=1)  # (bs,)
        nearest_protos = protos[nearest_proto_idx]  # (bs, dim)
        
        # 从传输后的特征到最近原型的 W2 距离
        if return_W2:
            W2_to_proto = self._l2_norm_sqr(x_input - nearest_protos).mean()
            W2_over_blocks += W2_to_proto
        
        if return_W2:
            return W2_over_blocks, nearest_protos
        else:
            return nearest_protos
    
    def _to_gmflow_format(self, r, S, log_alpha, epsilon, batch_size=1):
        """
        将参数转换为 GMFlow 的字典格式
        参考: GMFlow-main/lib/models/diffusions/gmflow.py
        
        GMFlow 格式: 
        - 'means': (bs, num_gaussians, out_channels, h, w)
        - 'logstds': (bs, 1, 1, 1, 1) 或 (bs, num_gaussians, 1, h, w)
        - 'logweights': (bs, num_gaussians, 1, h, w)
        
        对于 1D 特征向量，我们适配为: out_channels=dim, h=1, w=1
        """
        # 计算 logstds (从协方差矩阵)
        # GMFlow 使用 logstds = 0.5 * log(variance)
        if self.is_diagonal:
            # 对角协方差：每个维度独立，使用平均方差
            variances = epsilon * S  # (n_potentials, dim)
            avg_variance = variances.mean(dim=-1)  # (n_potentials,)
            logstds = 0.5 * torch.log(avg_variance + 1e-8)  # (n_potentials,)
        else:
            # 非对角：使用平均特征值（主成分的方差）
            S_eigenvals = torch.linalg.eigvalsh(S).mean(dim=-1)  # (n_potentials,)
            logstds = 0.5 * torch.log(epsilon * S_eigenvals + 1e-8)  # (n_potentials,)
        
        # 扩展维度以匹配 GMFlow 格式: (bs, num_gaussians, out_channels, h, w)
        # 对于 1D 特征: out_channels=dim, h=1, w=1
        # means: (n_potentials, dim) -> (1, n_potentials, dim, 1, 1)
        means = r.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, n_potentials, dim, 1, 1)
        
        # logstds: (n_potentials,) -> (1, 1, 1, 1, 1) [共享所有高斯]
        logstds = logstds.mean().unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, 1, 1)
        
        # logweights: (n_potentials,) -> (1, n_potentials, 1, 1, 1)
        logweights = log_alpha.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (1, n_potentials, 1, 1, 1)
        
        # 扩展到 batch_size
        if batch_size > 1:
            means = means.expand(batch_size, -1, -1, -1, -1)
            logweights = logweights.expand(batch_size, -1, -1, -1, -1)
            logstds = logstds.expand(batch_size, -1, -1, -1, -1)
        
        return {
            'means': means,  # (bs, num_gaussians, dim, 1, 1)
            'logstds': logstds,  # (bs, 1, 1, 1, 1)
            'logweights': logweights  # (bs, num_gaussians, 1, 1, 1)
        }
    
    def _from_gmflow_format(self, gm_dict):
        """
        从 GMFlow 格式提取均值（用于 1D 特征）
        参考: GMFlow-main/lib/ops/gmflow_ops/gmflow_ops.py 的 gm_to_mean
        """
        # gm_dict['means']: (bs, num_gaussians, dim, 1, 1)
        # 返回: (bs, dim) - 使用 gm_to_mean 或直接加权平均
        if self.has_gmflow_ops:
            try:
                # 使用 GMFlow 的 gm_to_mean
                # gm_to_mean 返回: (bs, dim, 1, 1)
                mean = gm_to_mean(gm_dict)  # (bs, dim, 1, 1)
                return mean.squeeze(-1).squeeze(-1)  # (bs, dim)
            except Exception:
                # Fallback
                pass
        
        # PyTorch fallback: 加权平均（实现 gm_to_mean 的逻辑）
        means = gm_dict['means']  # (bs, num_gaussians, dim, 1, 1)
        logweights = gm_dict['logweights']  # (bs, num_gaussians, 1, 1, 1)
        # 实现 gm_to_mean_jit 的逻辑: ((logweights * power).softmax(dim=-4) * means).sum(dim=-4)
        weights = logweights.softmax(dim=1)  # (bs, num_gaussians, 1, 1, 1)
        mean = (weights * means).sum(dim=1)  # (bs, dim, 1, 1)
        return mean.squeeze(-1).squeeze(-1)  # (bs, dim)
    
    def _compute_gmm_parameters(self, x, r, S, log_alpha, epsilon):
        """计算混合高斯模型的参数（GMFlow 核心方法）"""
        if self.is_diagonal:
            x_S_x = (x[:, None, :]*S[None, :, :]*x[:, None, :]).sum(dim=-1)
            x_r = (x[:, None, :]*r[None, :, :]).sum(dim=-1)
            r_x = r[None, :, :] + S[None, :]*x[:, None, :]
        else:
            x_S_x = (x[:, None, None, :]@(S[None, :, :, :]@x[:, None, :, None]))[:, :, 0, 0]
            x_r = (x[:, None, :]*r[None, :, :]).sum(dim=-1)
            r_x = r[None, :, :] + (S[None, :, : , :]@x[:, None, :, None])[:, :, :, 0]
        
        exp_argument = (x_S_x + 2*x_r)/(2*epsilon) + log_alpha[None, :]
        return exp_argument, r_x
        
    @torch.no_grad()
    def forward(self, x):
        """
        使用 GMFlow 方法进行采样
        参考: GMFlow-main/lib/models/diffusions/gmflow.py 的 gm_sample
        """
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        log_alpha = self.log_alpha
        
        samples = []
        batch_size = x.shape[0]
        sampling_batch_size = self.sampling_batch_size

        num_sampling_iterations = (
            batch_size//sampling_batch_size if batch_size % sampling_batch_size == 0 else (batch_size//sampling_batch_size) + 1
        )

        for i in range(num_sampling_iterations):
            sub_batch_x = x[sampling_batch_size*i:sampling_batch_size*(i+1)]
            sub_batch_size = sub_batch_x.shape[0]
            
            # 转换为 GMFlow 格式
            gm_dict = self._to_gmflow_format(r, S, log_alpha, epsilon, batch_size=sub_batch_size)
            
            # 使用 GMFlow 的采样方法
            # 参考: GMFlow-main/lib/models/diffusions/gmflow.py 的 gm_sample
            if self.has_gmflow_ops:
                try:
                    # 使用 gm_to_sample
                    # gm_to_sample 返回: (bs, n_samples, dim, 1, 1)
                    samples_gm = gm_to_sample(gm_dict, n_samples=1)  # (bs, 1, dim, 1, 1)
                    sample = samples_gm.squeeze(1).squeeze(-1).squeeze(-1)  # (bs, dim)
                    samples.append(sample)
                except Exception as e:
                    # Fallback to PyTorch
                    exp_argument, r_x = self._compute_gmm_parameters(sub_batch_x, r, S, log_alpha, epsilon)
                    if self.is_diagonal:
                        mix = Categorical(logits=exp_argument)
                        comp = Independent(Normal(loc=r_x, scale=torch.sqrt(epsilon*S)[None, :, :]), 1)
                        gmm = MixtureSameFamily(mix, comp)
                    else:
                        mix = Categorical(logits=exp_argument)
                        comp = MultivariateNormal(loc=r_x, covariance_matrix=epsilon*S)
                        gmm = MixtureSameFamily(mix, comp)
                    samples.append(gmm.sample())
            else:
                # PyTorch fallback
                exp_argument, r_x = self._compute_gmm_parameters(sub_batch_x, r, S, log_alpha, epsilon)
                if self.is_diagonal:
                    mix = Categorical(logits=exp_argument)
                    comp = Independent(Normal(loc=r_x, scale=torch.sqrt(epsilon*S)[None, :, :]), 1)
                    gmm = MixtureSameFamily(mix, comp)
                else:
                    mix = Categorical(logits=exp_argument)
                    comp = MultivariateNormal(loc=r_x, covariance_matrix=epsilon*S)
                    gmm = MixtureSameFamily(mix, comp)
                samples.append(gmm.sample())

        samples = torch.cat(samples, dim=0)
        return samples
    
    @torch.no_grad()
    def forward_with_targets(self, x, target_r, target_log_alpha=None):#这个是有问题的权重不均匀后续再修改
        S = self.get_S()
        epsilon = self.epsilon
        
        # 处理目标原型，确保形状兼容
        n_protos = target_r.shape[0]
        if n_protos < self.r.shape[0]:
            # 填充到n_potentials大小
            r = torch.zeros_like(self.r)
            r[:n_protos] = target_r
            r[n_protos:] = target_r[0:1].expand(self.r.shape[0] - n_protos, -1)
            S = S  # 使用完整的S
        elif n_protos > self.r.shape[0]:
            # 截断
            r = target_r[:self.r.shape[0]]
            S = S[:self.r.shape[0]]
        else:
            r = target_r
            S = S[:n_protos]
        
        if target_log_alpha is not None:
            if target_log_alpha.shape[0] < self.log_alpha.shape[0]:
                log_alpha = torch.cat([
                    target_log_alpha,
                    torch.ones(self.log_alpha.shape[0] - target_log_alpha.shape[0], 
                             device=target_log_alpha.device) * torch.log(torch.tensor(1e-8))
                ])
            elif target_log_alpha.shape[0] > self.log_alpha.shape[0]:
                log_alpha = target_log_alpha[:self.log_alpha.shape[0]]
            else:
                log_alpha = target_log_alpha
        else:
            uniform_weights = torch.ones(n_protos, device=target_r.device) / n_protos
            if n_protos < self.log_alpha.shape[0]:
                full_weights = torch.cat([
                    uniform_weights,
                    torch.ones(self.log_alpha.shape[0] - n_protos, device=target_r.device) * 1e-8
                ])
            else:
                full_weights = uniform_weights[:self.log_alpha.shape[0]]
            log_alpha = torch.log(full_weights + 1e-8)
        
        # 执行映射（与forward()相同的逻辑）
        samples = []
        batch_size = x.shape[0]
        sampling_batch_size = self.sampling_batch_size

        num_sampling_iterations = (
            batch_size//sampling_batch_size if batch_size % sampling_batch_size == 0 else (batch_size//sampling_batch_size) + 1
        )

        for i in range(num_sampling_iterations):
            sub_batch_x = x[sampling_batch_size*i:sampling_batch_size*(i+1)]
            
            # 使用 GMMFlow 方法计算参数
            exp_argument, r_x = self._compute_gmm_parameters(sub_batch_x, r, S, log_alpha, epsilon)
            
            # 从混合高斯分布中采样（GMMFlow 核心）
            if self.is_diagonal:
                mix = Categorical(logits=exp_argument)
                comp = Independent(Normal(loc=r_x, scale=torch.sqrt(epsilon*S)[None, :, :]), 1)
                gmm = MixtureSameFamily(mix, comp)
            else:
                mix = Categorical(logits=exp_argument)
                comp = MultivariateNormal(loc=r_x, covariance_matrix=epsilon*S)
                gmm = MixtureSameFamily(mix, comp)

            samples.append(gmm.sample())

        samples = torch.cat(samples, dim=0)
        return samples
    
    def get_drift(self, x, t):
        """
        GMMFlow 的漂移函数：计算流模型的漂移项
        使用流模型的连续时间动力学
        """
        x = torch.clone(x)
        x = torch.tensor(x, requires_grad=True)
        
        epsilon = self.epsilon
        r = self.get_r()
        
        S_diagonal = torch.exp(self.S_log_diagonal_matrix)
        # GMMFlow 的时间相关协方差矩阵
        A_diagonal = (t/(epsilon*(1-t)))[:, None, None] + 1/(epsilon*S_diagonal)[None, :, :] 
        
        S_log_det = torch.sum(self.S_log_diagonal_matrix, dim=-1) 
        A_log_det = torch.sum(torch.log(A_diagonal), dim=-1) 
        
        log_alpha = self.log_alpha 
        
        if self.is_diagonal:
            S = S_diagonal 
            A = A_diagonal 
            
            S_inv = 1/S 
            A_inv = 1/A 
            
            # GMMFlow 的流模型参数
            c = ((1/(epsilon*(1-t)))[:, None]*x)[:, None, :] + (r/(epsilon*S_diagonal))[None, :, :] 
            
            exp_arg = (
                log_alpha[None, :] - 0.5*S_log_det[None, :] - 0.5*A_log_det
                - 0.5*((r*S_inv*r)/epsilon).sum(dim=-1)[None, :] + 0.5*(c*A_inv*c).sum(dim=-1)
            )
            
        else:
            S = (self.S_rotation_matrix*S_diagonal[:, None, :])@torch.permute(self.S_rotation_matrix, (0, 2, 1))
            A = (self.S_rotation_matrix[None, :, :, :]*A_diagonal[:, :, None, :])@torch.permute(self.S_rotation_matrix, (0, 2, 1))[None, :, :, :]
            
            S_inv = (self.S_rotation_matrix*(1/S_diagonal[:, None, :]))@torch.permute(self.S_rotation_matrix, (0, 2, 1))
            A_inv = (self.S_rotation_matrix[None, :, :, :]*(1/A_diagonal[:, :, None, :]))@torch.permute(self.S_rotation_matrix, (0, 2, 1))[None, :, :, :]
            
            c = ((1/(epsilon*(1-t)))[:, None]*x)[:, None, :] + (S_inv@(r[:, :, None]))[None, :, :, 0]/epsilon 
            
            c_A_inv_c = (c[:, :, None, :]@A_inv@c[:, :, :, None])[:, :, 0, 0]
            r_S_inv_r = (r[:, None, :]@S_inv@r[:, :, None])[None, :, 0, 0]
            
            exp_arg = (
                log_alpha[None, :] - 0.5*S_log_det[None, :] - 0.5*A_log_det - 0.5*r_S_inv_r/epsilon + 0.5*c_A_inv_c
            )

        # GMMFlow 的漂移计算：使用对数求和指数和对数势函数的梯度
        lse = torch.logsumexp(exp_arg, dim=-1)
        drift = (-x/(1-t[:, None]) + epsilon*torch.autograd.grad(lse, x, grad_outputs=torch.ones_like(lse, device=lse.device), create_graph=True)[0]) 
        
        return drift
    
    def sample_at_time_moment(self, x, t):
        t = t.to(x.device)
        y = self(x)
        
        return t*y + (1-t)*x + torch.sqrt(t*(1-t)*self.epsilon)*torch.randn_like(x)
    
    def get_log_potential(self, x):
        S = self.get_S()
        r = self.get_r()
        
        # may compute on CPU
        if self.is_diagonal:
            mix = Categorical(logits=self.log_alpha)
            comp = Independent(Normal(loc=r, scale=torch.sqrt(self.epsilon*S)), 1)
            gmm = MixtureSameFamily(mix, comp)

            potential = gmm.log_prob(x) + torch.logsumexp(self.log_alpha, dim=-1)
        else:
            mix = Categorical(logits=self.log_alpha)
            comp = MultivariateNormal(loc=r, covariance_matrix=self.epsilon*S)
            gmm = MixtureSameFamily(mix, comp)
            
            potential = gmm.log_prob(x) + torch.logsumexp(self.log_alpha, dim=-1)
        
        return potential
    
    def get_log_C(self, x):
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        log_alpha = self.log_alpha
                
        if self.is_diagonal:
            x_S_x = (x[:, None, :]*S[None, :, :]*x[:, None, :]).sum(dim=-1)
            x_r = (x[:, None, :]*r[None, :, :]).sum(dim=-1)
        else:
            x_S_x = (x[:, None, None, :]@(S[None, :, :, :]@x[:, None, :, None]))[:, :, 0, 0]
            x_r = (x[:, None, :]*r[None, :, :]).sum(dim=-1)
            
        exp_argument = (x_S_x + 2*x_r)/(2*epsilon) + log_alpha[None, :]
        log_norm_const = torch.logsumexp(exp_argument, dim=-1)
        
        return log_norm_const
    
    def sample_euler_maruyama(self, x, n_steps):
        epsilon = self.epsilon
        t = torch.zeros(x.shape[0], device=x.device)
        dt = 1/n_steps
        trajectory = [x]
        
        for i in range(n_steps):
            x = x + self.get_drift(x, t)*dt + math.sqrt(dt)*torch.sqrt(epsilon)*torch.randn_like(x, device=x.device)
            t += dt
            trajectory.append(x)
            
        return torch.stack(trajectory, dim=1)
    
class DPDL_ETF_Classifier(nn.Module):
    def __init__(self, train_loader=None, proto_allocation_save_path=None, 
                 proto_allocation_load_path=None):
        super(DPDL_ETF_Classifier, self).__init__()
        self.feat_dim = config_dpdl.feature_dim
        self.num_classes = config_dpdl.num_classes
        #ETF
        P = self.generate_random_orthogonal_matrix(config_dpdl.feature_dim, config_dpdl.num_classes)
        I = torch.eye(config_dpdl.num_classes)
        one = torch.ones(config_dpdl.num_classes, config_dpdl.num_classes)
        scaling_factor = np.sqrt(config_dpdl.num_classes / (config_dpdl.num_classes - 1))
        self.ori_M = scaling_factor * torch.matmul(P, I - (1.0 / config_dpdl.num_classes) * one)
        self.ori_M = self.ori_M.to(device)
        self.ori_M.requires_grad_(False)
        #General ETF
        self.protos_per_class, self.class_sample_counts, self.radius_per_class = \
            self._compute_class_aware_prototypes(
                train_loader, 
                save_path=proto_allocation_save_path,
                load_path=proto_allocation_load_path
            )
        
        self.n_protos = sum(self.protos_per_class)
        
        # 初始化ETF原型
        self.etf_protos = self.initialize_etf_prototypes_dynamic(
            self.ori_M, self.protos_per_class, self.radius_per_class)
        self.etf_protos.requires_grad_(False)
        
        print(f"DPDL总原型数: {self.n_protos}")

    def _get_orthogonal_vector(self, vector, exclude_vector=None):
        feat_dim = vector.shape[0]
        if feat_dim == 1:
            return torch.ones(1, device=vector.device)
        
        random_vec = torch.randn(feat_dim, device=vector.device)
        random_vec = random_vec - torch.dot(random_vec, vector) * vector
        
        if exclude_vector is not None:
            random_vec = random_vec - torch.dot(random_vec, exclude_vector) * exclude_vector
        
        # 归一化
        norm = torch.norm(random_vec)
        return random_vec / norm if norm > 1e-8 else random_vec
    
    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        return P
    
    def _compute_class_aware_prototypes(self, train_loader, save_path=None, load_path=None):
        if load_path and os.path.exists(load_path):
            try:
                with open(load_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    protos_per_class = saved_data['protos_per_class']
                    class_counts = saved_data['class_counts']
                    radius_per_class = saved_data['radius_per_class']
                    print(f"加载原型分配: {protos_per_class}")
                    return protos_per_class, class_counts, radius_per_class
            except Exception as e:
                print(f"加载失败: {e}，重新计算...")
        
        print("分析训练数据分布...")
        
        # 统计每个类别的样本数量
        class_counts = [0] * self.num_classes
        total_samples = 0
        
        for batch in train_loader:
            labels = batch['label']
            for label in labels:
                class_counts[label] += 1
                total_samples += 1
        
        print(f"总样本数: {total_samples}, 各类别: {class_counts}")
        max_samples = max(class_counts)
        
        # 根据样本数量分配原型数量和radius
        protos_per_class = []
        radius_per_class = []
        
        for count in class_counts:
            if count == 0:
                protos_per_class.append(1)
                radius_per_class.append(0.0)
            elif count == max_samples:
                protos_per_class.append(1)
                radius_per_class.append(0.0)
            else:
                ratio = count / max_samples
                proto_count = max(1, int(1.0 / np.sqrt(ratio)))
                proto_count = min(proto_count, config_dpdl.max_protos_per_class)
                protos_per_class.append(proto_count)
                radius_per_class.append(0.0)
        
        # 重新分配radius
        multi_proto_classes = []
        for i, (count, proto_count) in enumerate(zip(class_counts, protos_per_class)):
            if proto_count > 1 and count > 0:
                ratio = count / max_samples
                multi_proto_classes.append((i, count, ratio))
        
        multi_proto_classes.sort(key=lambda x: x[1])
        
        max_radius_rad = config_dpdl.max_radius_degrees * math.pi / 180
        for rank, (class_idx, count, ratio) in enumerate(multi_proto_classes):
            if len(multi_proto_classes) > 1:
                rank_ratio = rank / (len(multi_proto_classes) - 1)
                adaptive_radius = max_radius_rad * (1.0 - rank_ratio)
            else:
                adaptive_radius = max_radius_rad
            
            adaptive_radius = max(0.0, min(adaptive_radius, max_radius_rad))
            radius_per_class[class_idx] = adaptive_radius
        
        print(f"原型分配: {protos_per_class}, 总数: {sum(protos_per_class)}")
        
        # 保存分配结果
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_data = {
                    'protos_per_class': protos_per_class,
                    'class_counts': class_counts,
                    'radius_per_class': radius_per_class,
                }
                with open(save_path, 'wb') as f:
                    pickle.dump(save_data, f)
                print(f"保存到: {save_path}")
            except Exception as e:
                print(f"保存失败: {e}")
        
        return protos_per_class, class_counts, radius_per_class

    def initialize_etf_prototypes_dynamic(self, etf_weights, protos_per_class, radius_per_class):
        num_classes = config_dpdl.num_classes
        feat_dim = config_dpdl.feature_dim
        n_protos = sum(protos_per_class)
        protos = torch.zeros((n_protos, feat_dim), device=etf_weights.device)
        
        etf_weights = etf_weights.T
        
        current_idx = 0
        for class_idx in range(num_classes):
            etf_weight = etf_weights[class_idx]
            class_proto_count = protos_per_class[class_idx]
            class_radius = radius_per_class[class_idx]
            
            if class_proto_count == 1:
                protos[current_idx] = etf_weight
            elif class_proto_count == 2:
                random_vec = self._get_orthogonal_vector(etf_weight)
                cos_radius = math.cos(class_radius)
                sin_radius = math.sin(class_radius)
                protos[current_idx] = cos_radius * etf_weight + sin_radius * random_vec
                protos[current_idx + 1] = cos_radius * etf_weight - sin_radius * random_vec
            else:
                basis1 = self._get_orthogonal_vector(etf_weight)
                basis2 = self._get_orthogonal_vector(etf_weight, basis1)
                
                angle_step = 2 * math.pi / class_proto_count
                cos_radius = math.cos(class_radius)
                sin_radius = math.sin(class_radius)
                
                for proto_idx in range(class_proto_count):
                    angle = proto_idx * angle_step
                    offset = sin_radius * (math.cos(angle) * basis1 + math.sin(angle) * basis2)
                    protos[current_idx + proto_idx] = cos_radius * etf_weight + offset
            
            current_idx += class_proto_count
        
        return protos
    
    def analyze_prototype_distribution(self):
        distances = []
        
        for class_idx in range(self.num_classes):
            sample_count = self.class_sample_counts[class_idx]
            proto_count = self.protos_per_class[class_idx]
            radius_deg = self.radius_per_class[class_idx] * 180 / math.pi
            samples_per_proto = sample_count / proto_count if proto_count > 0 else 0
            
            print(f"Class {class_idx}: {sample_count} samples, {proto_count} protos, {radius_deg:.2f}°, {samples_per_proto:.1f} samples/proto")
            
            # 计算原型与ETF权重的角度距离
            etf_weight = self.ori_M.T[class_idx]
            start_idx = sum(self.protos_per_class[:class_idx])
            end_idx = start_idx + proto_count
            class_protos = self.etf_protos[start_idx:end_idx]
            
            for proto in class_protos:
                cos_sim = torch.clip(torch.dot(proto, etf_weight), -1, 1)
                angle = torch.acos(cos_sim) * 180 / math.pi
                distances.append(angle.item())
        
        # 统计信息
        max_samples = max(self.class_sample_counts)
        min_samples = min(self.class_sample_counts)
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        print(f"\n统计: 角度距离 {np.mean(distances):.2f}±{np.std(distances):.2f}°, "
              f"不平衡比例 {imbalance_ratio:.2f}, 原型数 {min(self.protos_per_class)}-{max(self.protos_per_class)}")
        
        return {
            'distances': distances,
            'mean_distance': np.mean(distances),
            'imbalance_ratio': imbalance_ratio,
            'proto_count_range': (min(self.protos_per_class), max(self.protos_per_class))
        }
    
    def orthogonal_complement(self):
        feat_dim = self.etf_protos.shape[1]
        class_dim = self.etf_protos.shape[0]
        
        # 首先获取权重矩阵的列向量作为初始子空间的基
        weight_basis = self.ori_M.T.cpu()  # 转置使每行成为一个基向量
        
        # 创建一组完整的基向量（包括单位向量）
        full_basis = torch.eye(feat_dim)
        
        # 使用施密特正交化过程找到正交互补子空间
        complement_basis = []
        
        for i in range(feat_dim):
            v = full_basis[i]
            
            # 检查v是否与权重子空间线性无关
            # 通过计算v与权重基向量的投影，然后检查剩余部分是否显著
            v_copy = v.clone()
            
            # 从v中减去它在权重基向量上的投影
            for w in weight_basis:
                v_copy = v_copy - torch.dot(v_copy, w) * w / torch.dot(w, w)
            
            # 如果剩余向量的范数足够大，则它是互补子空间的一部分
            if torch.norm(v_copy) > 1e-6:
                # 归一化
                v_copy = v_copy / torch.norm(v_copy)
                complement_basis.append(v_copy)
        
        # 将互补基向量堆叠成矩阵
        if complement_basis:
            orthogonal_complement = torch.stack(complement_basis)
        else:
            # 如果没有找到互补基向量，返回空矩阵
            orthogonal_complement = torch.zeros((0, feat_dim))
        
        return orthogonal_complement
    
    def project_to_complement(self, features):
        orthogonal_comp = self.orthogonal_complement().cuda()
        if orthogonal_comp.shape[0] == 0:
            return torch.zeros_like(features)
        
        # 计算特征在正交补空间上的投影
        projection = torch.matmul(features, orthogonal_comp.T)
        projected_features = torch.matmul(projection, orthogonal_comp)
        return projected_features

class DPDL_PALMNet(nn.Module):
    def __init__(self, train_loader=None, proto_allocation_save_path=None, 
                 proto_allocation_load_path=None):
        super(DPDL_PALMNet, self).__init__()
        self.backbone = ResNet18_32x32(num_classes=config_dpdl.num_classes)
        self.dpdl_classifier = DPDL_ETF_Classifier(
            train_loader=train_loader,
            proto_allocation_save_path=proto_allocation_save_path,
            proto_allocation_load_path=proto_allocation_load_path
        )
        
        self.n_protos = self.dpdl_classifier.n_protos
        self.protos_per_class = self.dpdl_classifier.protos_per_class
        self.class_sample_counts = self.dpdl_classifier.class_sample_counts
        self.etf_protos = self.dpdl_classifier.etf_protos

        self.flowot = FlowOT(
            dim=config_dpdl.feature_dim,
            n_potentials=self.n_protos,
            epsilon=config_dpdl.sb_epsilon,
            is_diagonal=config_dpdl.sb_is_diagonal,
            sampling_batch_size=config_dpdl.sb_sampling_batch_size,
            S_diagonal_init=config_dpdl.sb_s_diagonal_init,
            hidden_dim=256,
            num_blocks=2,  # 减少 blocks 数量以提升速度（从 4 改为 2）
            T0=0.25,
            int_mtd='Euler',  # 使用 Euler 方法更快（从 RK4 改为 Euler）
            num_int_pts=2  # 减少积分点数以提升速度（从 3 改为 2）
        )
        
        self._initialize_flowot_with_etf_protos()
    
    def _initialize_flowot_with_etf_protos(self):
        """使用ETF原型初始化FlowOT"""
        with torch.no_grad():
            self.flowot.init_r_by_samples(self.etf_protos)
            class_weights = []
            for i, count in enumerate(self.class_sample_counts):
                proto_count = self.protos_per_class[i]
                proto_weight = count / proto_count if proto_count > 0 else 1.0
                class_weights.extend([proto_weight] * proto_count)
            class_weights = torch.tensor(class_weights, device=self.flowot.r.device)
            class_weights = class_weights / torch.sum(class_weights)
            self.flowot.log_alpha.data = torch.log(class_weights + 1e-8)
            print(f"FlowOT初始化: {self.n_protos}个原型, 权重范围 {class_weights.min():.4f}-{class_weights.max():.4f}")
    

    
    def _logit_loss(self, rnet, x, y):
        """
        FlowOT 的 Logit loss（密度比损失）
        参考: FlowOT-main/src/refinement.py 的 logit_loss
        
        Args:
            rnet: 密度比网络 R^d -> R
            x: 源分布样本 (bs, dim)
            y: 目标分布样本 (bs, dim)
        Returns:
            loss: 标量
        """
        softplus = torch.nn.Softplus(beta=1)
        loss_X = softplus(rnet(x)).mean()
        loss_Y = softplus(-rnet(y)).mean()
        return loss_X + loss_Y
    
    def _compute_flowot_loss(self, features, class_protos, class_id, gamma=0.5):
        """
        计算 FlowOT 损失：Logit_loss + gamma * W2_distance
        参考: FlowOT-main/src/refinement.py 的 alternate_training_ode_loss
        
        Args:
            features: (bs, dim) - 类别特征
            class_protos: (n_class_protos, dim) - 该类别的原型
            class_id: 类别ID
            gamma: W2 正则化权重
        Returns:
            (logit_loss, w2_loss, total_loss)
        """
        if len(class_protos) == 0 or len(features) == 0:
            return (torch.tensor(0.0, device=features.device), 
                   torch.tensor(0.0, device=features.device),
                   torch.tensor(0.0, device=features.device))
        
        # FlowOT: 计算特征到原型的流传输和 W2 距离
        W2_loss, transported_features = self.flowot.flow_features_to_protos(
            features, class_protos, return_W2=True
        )
        
        # FlowOT: Logit loss（简化版：使用特征到原型的距离作为 logit）
        # 这里我们使用简单的距离作为 logit，而不是训练单独的 ratio network
        # 计算传输后的特征到原型的距离
        distances = torch.cdist(transported_features, class_protos)  # (bs, n_protos)
        min_distances = distances.min(dim=1)[0]  # (bs,)
        
        # Logit loss: 最小化传输后特征到原型的距离
        logit_loss = min_distances.mean()
        
        # 总损失：Logit_loss + gamma * W2
        total_loss = logit_loss + gamma * W2_loss
        
        return logit_loss, W2_loss, total_loss
    
    def get_dpdl_loss(self, normal_features, labels=None, gamma=0.5, use_fast_mode=True):
        """
        计算 DPDL 损失，使用 FlowOT 方法（优化版本）
        严格参考: FlowOT-main/src/refinement.py 的 alternate_training_ode_loss
        
        FlowOT 损失：Logit_loss + gamma * W2_distance
        - Logit_loss: 确保传输后的特征接近原型
        - W2_distance: 最优传输的正则化项，确保传输路径最优
        
        优化选项：
        - use_fast_mode=True: 使用简化的快速模式（不进行完整流传输，直接计算距离）
        - use_fast_mode=False: 使用完整的 FlowOT 流传输（较慢但更准确）
        
        Args:
            normal_features: (bs, dim) - 输入特征
            labels: (bs,) - 类别标签
            gamma: W2 正则化权重（默认 0.5，参考 FlowOT）
            use_fast_mode: 是否使用快速模式（默认 True）
        """
        batch_size = normal_features.shape[0]
        
        if use_fast_mode:
            # 快速模式：直接计算特征到原型的距离，不进行完整的流传输
            # 这大幅提升速度，但损失了流传输的精确性
            all_protos = self.etf_protos  # (n_protos, dim)
            
            # 计算每个特征到所有原型的距离
            distances = torch.cdist(normal_features, all_protos)  # (bs, n_protos)
            
            # 找到每个特征对应的类别原型
            nearest_proto_indices = []
            for i, label in enumerate(labels):
                start_idx = sum(self.protos_per_class[:label])
                end_idx = start_idx + self.protos_per_class[label]
                # 在该类别的原型范围内找到最近的原型
                class_distances = distances[i, start_idx:end_idx]
                nearest_idx = start_idx + class_distances.argmin()
                nearest_proto_indices.append(nearest_idx)
            
            nearest_proto_indices = torch.tensor(nearest_proto_indices, device=normal_features.device)
            nearest_protos = all_protos[nearest_proto_indices]  # (bs, dim)
            
            # Logit loss: 特征到最近原型的距离
            logit_loss = torch.norm(normal_features - nearest_protos, dim=1).mean()
            
            # W2 loss: 简化为特征到原型的 L2 距离的平方（近似 W2）
            w2_loss = torch.norm(normal_features - nearest_protos, dim=1).pow(2).mean()
        else:
            # 完整模式：使用 FlowOT 流传输（较慢但更准确）
            accumulated_logit_loss = 0.0
            accumulated_w2_loss = 0.0
            
            # 按类别计算损失
            for class_id in torch.unique(labels):
                class_mask = (labels == class_id)
                if class_mask.sum() == 0:
                    continue
                    
                class_features = normal_features[class_mask] 
                
                start_idx = sum(self.protos_per_class[:class_id])
                end_idx = start_idx + self.protos_per_class[class_id]
                class_protos = self.etf_protos[start_idx:end_idx]
                
                # 计算 FlowOT 损失：Logit_loss + gamma * W2
                logit_loss, w2_loss, _ = self._compute_flowot_loss(
                    class_features, class_protos, class_id, gamma=gamma
                )
                
                accumulated_logit_loss += logit_loss * class_mask.sum().float()
                accumulated_w2_loss += w2_loss * class_mask.sum().float()

            logit_loss = accumulated_logit_loss / batch_size
            w2_loss = accumulated_w2_loss / batch_size
        
        # FlowOT 总损失：Logit_loss + gamma * W2
        total_loss = logit_loss + gamma * w2_loss
        
        return {
            'total_loss': total_loss,
            'logit': logit_loss,  # FlowOT Logit loss
            'w2': w2_loss  # FlowOT W2 distance
        }
    
    def forward(self, x):
        _, features = self.backbone(x, return_feature=True)
        features = F.normalize(features, dim=1)
        return features
        
class DPDLTrainer:
    def __init__(self, net, train_loader):
        self.net = net
        self.train_loader = train_loader
        
        all_params = []
        all_params.extend(self.net.backbone.parameters())
        all_params.extend(self.net.flowot.parameters())
        self.optimizer = torch.optim.Adam(
            all_params,
            lr=config_dpdl.lr,
            weight_decay=1e-4
        )
        
        self.optimizer = torch.optim.SGD(
            all_params,
            lr=config_dpdl.lr,
            momentum=config_dpdl.momentum,
            weight_decay=config_dpdl.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config_dpdl.epochs * len(train_loader),
                1,
                1e-6 / config_dpdl.lr,
            ),
        )

    def train_epoch(self, epoch_idx):
        self.net.train()
        total_loss_avg = 0.0
        logit_avg = 0.0
        w2_avg = 0.0
        
        for batch in tqdm(self.train_loader, desc=f'DPDL FlowOT Epoch {epoch_idx:03d}'):
            data = batch['data'].to(device)
            labels = batch['label'].to(device)
            
            features = self.net(data)
            
            loss_dict = self.net.get_dpdl_loss(features, labels, gamma=0.5)
            total_loss = loss_dict['total_loss'] 
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss_avg = total_loss_avg * 0.9 + float(total_loss) * 0.1
            logit_avg = logit_avg * 0.9 + float(loss_dict.get('logit', 0.0)) * 0.1
            w2_avg = w2_avg * 0.9 + float(loss_dict.get('w2', 0.0)) * 0.1
        
        self.scheduler.step()
        
        
        return {
            'epoch': epoch_idx,
            'total_loss': total_loss_avg,
            'logit': logit_avg,  # FlowOT Logit loss
            'w2': w2_avg,  # FlowOT W2 distance
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
class DPLTrainingManager:
    def __init__(self, config_dpdl):
        self.config_dpdl = config_dpdl
        self.device = device
        self.best_loss = float('inf')
        self.best_epoch_idx = 0
        self.best_acc = 0.0
        self.best_acc_epoch_idx = 0
        self.train_losses = []
        self.val_losses = []
        
    def train(self, train_loader, val_loader):
        model = DPDL_PALMNet(
            train_loader=train_loader,
            proto_allocation_save_path=self.config_dpdl.proto_allocation_save_path,
            proto_allocation_load_path=self.config_dpdl.proto_allocation_load_path
        )
        trainer = DPDLTrainer(model, train_loader)
        model = model.to(self.device)
        
        for epoch in range(1, self.config_dpdl.epochs + 1):
            train_metrics = trainer.train_epoch(epoch)
            val_metrics = self.validate(model, val_loader)
            
            self.train_losses.append(train_metrics['total_loss'])
            self.val_losses.append(val_metrics['loss'])
            
            self.report(epoch, train_metrics, val_metrics)
            self._save_model(model, epoch, val_metrics, self.config_dpdl.epochs)
             
        return model, trainer
    
    def validate(self, model, val_loader):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
    
        # 直接使用ETF原型，每个类别一个原型
        etf_protos = model.dpdl_classifier.ori_M.T  # [num_classes, feat_dim]
        
        for batch in tqdm(val_loader, desc='Validating'):
            data = batch['data'].to(self.device)
            labels = batch.get('label', None)
            
            with torch.no_grad():
                features = model(data)
            
            # 计算损失时需要梯度（用于get_drift中的autograd.grad）
            loss_dict = model.get_dpdl_loss(features, labels)
            total_loss += loss_dict['total_loss'].item()
            
            # 释放梯度以节省内存（验证阶段不需要反向传播，不会更新参数）
            features = features.detach()

            with torch.no_grad():
                labels = labels.to(self.device)
                # 计算特征与ETF原型的距离，直接得到类别预测
                distances = torch.cdist(features, etf_protos)  # [batch_size, num_classes]
                predicted = distances.argmin(dim=1)  # [batch_size]
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total if total > 0 else 0.0
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _save_model(self, model, epoch, val_metrics, num_epochs):
        output_dir = self.config_dpdl.save_dir
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        
        state = {
            'model_state_dict': state_dict,
            'epoch': epoch,
            'loss': val_metrics['loss'],
            'accuracy': val_metrics.get('accuracy', 0.0),
            'protos_per_class': model.dpdl_classifier.protos_per_class,
            'etf_protos': model.dpdl_classifier.etf_protos,
        }
        

        if val_metrics['loss'] <= self.best_loss:
            old_pattern = os.path.join(output_dir, 'best_epoch*_loss*.pth')
            for old_file in glob.glob(old_pattern):
                try:
                    os.remove(old_file)
                except Exception as e:
                    print(f"Warning: Failed to remove {old_file}: {e}")
            
            self.best_epoch_idx = epoch
            self.best_loss = val_metrics['loss']
            
            torch.save(state, os.path.join(output_dir, 'best_dpdl_etf_model.pth'))
            save_fname = f'best_epoch{self.best_epoch_idx}_loss{self.best_loss:.4f}.pth'
            save_pth = os.path.join(output_dir, save_fname)
            torch.save(state, save_pth)

        val_acc = val_metrics.get('accuracy', 0.0)
        if val_acc >= self.best_acc:
            old_acc_pattern = os.path.join(output_dir, 'best_acc_epoch*_acc*.pth')
            for old_file in glob.glob(old_acc_pattern):
                try:
                    os.remove(old_file)
                except Exception as e:
                    print(f"Warning: Failed to remove {old_file}: {e}")

            self.best_acc_epoch_idx = epoch
            self.best_acc = val_acc

            torch.save(state, os.path.join(output_dir, 'best_acc_dpdl_etf_model.pth'))
            acc_save_fname = f'best_acc_epoch{self.best_acc_epoch_idx}_acc{self.best_acc:.2f}.pth'
            acc_save_pth = os.path.join(output_dir, acc_save_fname)
            torch.save(state, acc_save_pth)
        
        if epoch == num_epochs:
            old_last_pattern = os.path.join(output_dir, 'last_epoch*.pth')
            for old_file in glob.glob(old_last_pattern):
                try:
                    os.remove(old_file)
                except Exception as e:
                    print(f"Warning: Failed to remove {old_file}: {e}")
            save_fname = f'last_epoch{epoch}_loss{val_metrics["loss"]:.4f}.pth'
            save_pth = os.path.join(output_dir, save_fname)
            torch.save(state, save_pth)
    
    def report(self, epoch, train_metrics, val_metrics):
        val_loss = val_metrics['loss'] if isinstance(val_metrics, dict) else val_metrics
        val_acc = val_metrics.get('accuracy', 0.0) if isinstance(val_metrics, dict) else 0.0
        
        print(f'Epoch {epoch:03d} | '
              f'LR: {train_metrics["lr"]:.6f} | '
              f'Train Loss: {train_metrics["total_loss"]:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.2f}% | '
              f'Logit: {train_metrics.get("logit", 0.0):.4f} | '
              f'W2: {train_metrics.get("w2", 0.0):.4f}')

class PALMRMDSPostprocessor:
    def __init__(self, num_classes=16):
        self.num_classes = num_classes
        self.setup_flag = False
        self.class_mean = None
        self.precision = None
        self.whole_mean = None
        self.whole_precision = None

    def setup(self, model, train_loader):
        """使用训练数据设置RMDS统计量"""
        if not self.setup_flag:
            print('\n Estimating mean and variance from training set for RMDS...')
            all_feats = []
            all_labels = []
            all_preds = []
            
            model.eval()
            with torch.no_grad():
                for batch in tqdm(train_loader, desc='RMDS Setup: ', position=0, leave=True):
                    data = batch['data'].cuda()
                    labels = batch['label']
                    
                    # 获取特征
                    features = model(data)
                    features = F.normalize(features, dim=-1)
                    
                    all_feats.append(features.cpu())
                    all_labels.append(deepcopy(labels))
                    
                    # 简单的预测（基于特征与原型的距离）
                    # 这里我们使用一个简单的分类方法
                    pred = torch.randint(0, self.num_classes, (len(data),))
                    all_preds.append(pred)

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            
            # 计算类别条件统计量
            self.class_mean = []
            centered_data = []
            for c in range(self.num_classes):
                class_samples = all_feats[all_labels.eq(c)].data
                if len(class_samples) > 0:
                    self.class_mean.append(class_samples.mean(0))
                    centered_data.append(class_samples - self.class_mean[c].view(1, -1))
                else:
                    # 如果某个类别没有样本，使用零向量
                    self.class_mean.append(torch.zeros(all_feats.shape[1]))
                    centered_data.append(torch.zeros(1, all_feats.shape[1]))

            self.class_mean = torch.stack(self.class_mean)  # shape [#classes, feature dim]

            # 计算类别协方差矩阵的逆
            if len(torch.cat(centered_data)) > 0:
                group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
                group_lasso.fit(torch.cat(centered_data).cpu().numpy().astype(np.float32))
                self.precision = torch.from_numpy(group_lasso.precision_).float()
            else:
                self.precision = torch.eye(all_feats.shape[1])

            # 计算整体统计量
            self.whole_mean = all_feats.mean(0)
            centered_data = all_feats - self.whole_mean.view(1, -1)
            group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
            group_lasso.fit(centered_data.cpu().numpy().astype(np.float32))
            self.whole_precision = torch.from_numpy(group_lasso.precision_).float()
            
            self.setup_flag = True
            print('RMDS setup completed.')
        else:
            pass

    @torch.no_grad()
    def postprocess(self, model, data):
        """使用RMDS方法计算置信度"""
        features = model(data)
        features = F.normalize(features, dim=-1)
        
        # 计算背景分数（相对于整体分布）
        tensor1 = features.cpu() - self.whole_mean.view(1, -1)
        background_scores = -torch.matmul(
            torch.matmul(tensor1, self.whole_precision), tensor1.t()).diag()

        # 计算每个类别的分数
        class_scores = torch.zeros((features.shape[0], self.num_classes))
        for c in range(self.num_classes):
            tensor = features.cpu() - self.class_mean[c].view(1, -1)
            class_scores[:, c] = -torch.matmul(
                torch.matmul(tensor, self.precision), tensor.t()).diag()
            class_scores[:, c] = class_scores[:, c] - background_scores

        # 获取最高分数作为置信度
        conf, pred = torch.max(class_scores, dim=1)
        return pred, conf

class PALMMDSPostprocessor:
    def __init__(self, num_classes=16):
        self.num_classes = num_classes
        self.setup_flag = False
        self.class_mean = None
        self.precision = None

    def setup(self, model, train_loader):
        """使用训练数据设置MDS统计量"""
        if not self.setup_flag:
            print('\n Estimating mean and variance from training set for MDS...')
            all_feats = []
            all_labels = []
            all_preds = []
            
            model.eval()
            with torch.no_grad():
                for batch in tqdm(train_loader, desc='MDS Setup: ', position=0, leave=True):
                    data = batch['data'].cuda()
                    labels = batch['label']
                    
                    # 获取特征
                    features = model(data)
                    # features = F.normalize(features, dim=-1)
                    
                    all_feats.append(features.cpu())
                    all_labels.append(deepcopy(labels))
                    
                    # 简单的预测（基于特征与原型的距离）
                    # 这里我们使用一个简单的分类方法
                    pred = torch.randint(0, self.num_classes, (len(data),))
                    all_preds.append(pred)

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            
            # 计算类别条件统计量
            self.class_mean = []
            centered_data = []
            for c in range(self.num_classes):
                class_samples = all_feats[all_labels.eq(c)].data
                self.class_mean.append(class_samples.mean(0))
                centered_data.append(class_samples - self.class_mean[c].view(1, -1))


            self.class_mean = torch.stack(self.class_mean)  # shape [#classes, feature dim]
            group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
            group_lasso.fit(torch.cat(centered_data).cpu().numpy().astype(np.float32))
            self.precision = torch.from_numpy(group_lasso.precision_).float()
            self.setup_flag = True
        else:
            pass

    @torch.no_grad()
    def postprocess(self, model, data):
        """使用MDS方法计算置信度"""
        features = model(data)
        # features = F.normalize(features, dim=-1)
        
        # 计算每个类别的分数
        class_scores = torch.zeros((features.shape[0], self.num_classes))
        for c in range(self.num_classes):
            tensor = features.cpu() - self.class_mean[c].view(1, -1)
            class_scores[:, c] = -torch.matmul(
                torch.matmul(tensor, self.precision), tensor.t()).diag()

        # 获取最高分数作为置信度
        conf, pred = torch.max(class_scores, dim=1)
        return pred, conf

def ood_evaluate_palm_rmds(model, test_loader, ood_test_loader, train_loader, trainer=None, epoch_idx=101):

    model.eval()
    
    # 创建RMDS后处理器
    rmds_processor = PALMRMDSPostprocessor(num_classes=26)
    
    # 设置RMDS统计量
    rmds_processor.setup(model, train_loader)
    
    id_pred_list, id_conf_list, id_label_list = [], [], []
    ood_pred_list, ood_conf_list, ood_label_list = [], [], []
    
    # 评估ID数据
    with torch.no_grad():
        for batch in tqdm(iter(test_loader), desc='Eval_id: '):
            data = batch['data'].to(device)
            label = batch['label'].to(device)
            
            # 使用RMDS方法计算预测和置信度
            pred, conf = rmds_processor.postprocess(model, data)
            
            id_pred_list.append(pred.cpu())
            id_conf_list.append(conf.cpu())
            id_label_list.append(label.cpu())
    
    # 评估OOD数据
    with torch.no_grad():
        for batch in tqdm(iter(ood_test_loader), desc='Eval_ood: '):
            data = batch['data'].to(device)
            label = batch['label'].to(device)
            
            # 使用RMDS方法计算预测和置信度
            pred, conf = rmds_processor.postprocess(model, data)
            
            ood_pred_list.append(pred.cpu())
            ood_conf_list.append(conf.cpu())
            ood_label_list.append(label.cpu())
    
    # 转换为numpy数组
    id_pred_list = torch.cat(id_pred_list).numpy()
    id_conf_list = torch.cat(id_conf_list).numpy()
    id_label_list = torch.cat(id_label_list).numpy()
    
    ood_pred_list = torch.cat(ood_pred_list).numpy()
    ood_conf_list = torch.cat(ood_conf_list).numpy()
    ood_label_list = -1 * np.ones(len(ood_pred_list))  # OOD标签为-1
    
    # 计算评估指标
    pred = np.concatenate([id_pred_list, ood_pred_list])
    conf = np.concatenate([id_conf_list, ood_conf_list])
    label = np.concatenate([id_label_list, ood_label_list])
    
    metrics = compute_all_metrics(conf, label, pred)
    print(
        f"RMDS - FPR: {metrics[0]:.4f}, AUROC: {metrics[1]:.4f}, "
        f"AUPR_IN: {metrics[2]:.4f}, AUPR_OUT: {metrics[3]:.4f}, "
        f"ACC: {metrics[4]:.4f}"
    )
    
    return metrics

def ood_evaluate_palm_mds(model, test_loader, ood_test_loader, train_loader, trainer=None, epoch_idx=101):

    model.eval()
    
    # 创建MDS后处理器
    mds_processor = PALMMDSPostprocessor(num_classes=26)
    
    # 设置MDS统计量
    mds_processor.setup(model, train_loader)
    
    id_pred_list, id_conf_list, id_label_list = [], [], []
    ood_pred_list, ood_conf_list, ood_label_list = [], [], []
    
    # 评估ID数据
    with torch.no_grad():
        for batch in tqdm(iter(test_loader), desc='Eval_id: '):
            data = batch['data'].to(device)
            label = batch['label'].to(device)
            
            # 使用MDS方法计算预测和置信度
            pred, conf = mds_processor.postprocess(model, data)
            
            id_pred_list.append(pred.cpu())
            id_conf_list.append(conf.cpu())
            id_label_list.append(label.cpu())
    
    # 评估OOD数据
    with torch.no_grad():
        for batch in tqdm(iter(ood_test_loader), desc='Eval_ood: '):
            data = batch['data'].to(device)
            label = batch['label'].to(device)
            
            # 使用MDS方法计算预测和置信度
            pred, conf = mds_processor.postprocess(model, data)
            
            ood_pred_list.append(pred.cpu())
            ood_conf_list.append(conf.cpu())
            ood_label_list.append(label.cpu())
    
    # 转换为numpy数组
    id_pred_list = torch.cat(id_pred_list).numpy()
    id_conf_list = torch.cat(id_conf_list).numpy()
    id_label_list = torch.cat(id_label_list).numpy()
    
    ood_pred_list = torch.cat(ood_pred_list).numpy()
    ood_conf_list = torch.cat(ood_conf_list).numpy()
    ood_label_list = -1 * np.ones(len(ood_pred_list))  
    
    # 计算评估指标
    pred = np.concatenate([id_pred_list, ood_pred_list])
    conf = np.concatenate([id_conf_list, ood_conf_list])
    label = np.concatenate([id_label_list, ood_label_list])
    
    metrics = compute_all_metrics(conf, label, pred)
    print(
        f"MDS - FPR: {metrics[0]:.4f}, AUROC: {metrics[1]:.4f}, "
        f"AUPR_IN: {metrics[2]:.4f}, AUPR_OUT: {metrics[3]:.4f}, "
        f"ACC: {metrics[4]:.4f}"
    )
    
    return metrics

def ood_evaluate_orthogonal(model, test_loader, ood_test_loader, epoch_idx=101):
    """使用正交投影比进行OOD检测"""
    model.eval()
    id_pred_list, id_conf_list, id_label_list = [], [], []
    ood_pred_list, ood_conf_list, ood_label_list = [], [], []
    
    # 获取ETF权重矩阵和正交补空间
    etf_weights = model.dpdl_classifier.etf_protos.T
    orthogonal_comp = model.dpdl_classifier.orthogonal_complement().cuda()
    
    print(f"ETF weight shape: {etf_weights.shape}")
    print(f"Orthogonal complement shape: {orthogonal_comp.shape}")
    
    # 验证正交性
    ortho_check = torch.abs(torch.mm(orthogonal_comp, etf_weights)).max()
    print(f"Orthogonality check (should be close to 0): {ortho_check:.6f}")
    
    # 评估ID数据
    with torch.no_grad():
        for batch in tqdm(iter(test_loader), desc='Eval_id: '):
            data = batch['data'].to(device)
            label = batch['label'].to(device)
            
            features = model(data)
            
            # 计算logits（特征投影到ETF权重）
            logits = features @ etf_weights
            
            # 计算在正交补空间上的投影
            ortho_proj = model.dpdl_classifier.project_to_complement(features)
            etf_proj = features - ortho_proj
            score = torch.norm(etf_proj, dim=1)
            
            # 获取预测类别
            pred = torch.argmax(logits, dim=1)
            
            id_pred_list.append(pred.cpu())
            id_conf_list.append(score.cpu())
            id_label_list.append(label.cpu())
            
        id_pred_list = torch.cat(id_pred_list).numpy()
        id_conf_list = torch.cat(id_conf_list).numpy()
        id_label_list = torch.cat(id_label_list).numpy()
    
    # 评估OOD数据
    with torch.no_grad():
        for batch in tqdm(iter(ood_test_loader), desc='Eval_ood: '):
            data = batch['data'].to(device)
            label = batch['label'].to(device)
            
            features = model(data)
            
            # 计算logits（特征投影到ETF权重）
            logits = features @ etf_weights
            
            # 计算在正交补空间上的投影
            ortho_proj = model.dpdl_classifier.project_to_complement(features)
            etf_proj = features - ortho_proj
            score = torch.norm(etf_proj, dim=1)
            
            # 获取预测类别
            pred = torch.argmax(logits, dim=1)
            
            ood_pred_list.append(pred.cpu())
            ood_conf_list.append(score.cpu())
            ood_label_list.append(label.cpu())
            
        ood_pred_list = torch.cat(ood_pred_list).numpy()
        ood_conf_list = torch.cat(ood_conf_list).numpy()
        ood_label_list = torch.cat(ood_label_list).numpy()
        ood_label_list = -1 * np.ones_like(ood_label_list)
    
    # 绘制分数分布图


    pred = np.concatenate([id_pred_list, ood_pred_list])
    conf = np.concatenate([id_conf_list, ood_conf_list])
    label = np.concatenate([id_label_list, ood_label_list])
    
    metrics = compute_all_metrics(conf, label, pred)
    print(
        f"FPR: {metrics[0]:.4f}, AUROC: {metrics[1]:.4f}, "
        f"AUPR_IN: {metrics[2]:.4f}, AUPR_OUT: {metrics[3]:.4f}, "
        f"ACC: {metrics[4]:.4f}"
    )

if __name__ == "__main__":
    """主训练函数"""
    config_dpdl = Config_DPDL()
    configopenood = config.Config(*config_dpdl.config_files)
    os.makedirs(config_dpdl.save_dir, exist_ok=True)
    
    # 打印配置信息
    print(f"=== DPDL配置信息 ===")
    print(f"FlowOT epsilon: {config_dpdl.sb_epsilon}")
    print(f"原型最大数量: {config_dpdl.max_protos_per_class}")
    print("专注于FlowOT最优传输和OOD检测")
    
    # 获取数据加载器
    loader_dict = get_dataloader(configopenood)
    train_loader = loader_dict['train']
    val_loader = loader_dict['val']
    test_loader = loader_dict['test']
    ood_loader_dict = get_ood_dataloader(configopenood)
    ood_test_loader = ood_loader_dict['val']
    
    training_manager = DPLTrainingManager(config_dpdl)
    model, trainer = training_manager.train(train_loader, val_loader)
    print("训练完成！")
    
    # OOD检测评估
    print("\n=== OOD检测评估 ===")
    # ood_evaluate_orthogonal(model, test_loader, ood_test_loader, epoch_idx=101)   
    ood_evaluate_palm_rmds(model, test_loader, ood_test_loader, train_loader, trainer=None, epoch_idx=101)
    # ood_evaluate_palm_mds(model, test_loader, ood_test_loader, train_loader, trainer=None, epoch_idx=101)