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

class GMMFlow(nn.Module):
    """
    GMMFlow: 基于 GMFlow-main 的实际实现
    
    本实现严格参考了 GMFlow-main 的代码：
    - 参考文件: GMFlow-main/lib/models/diffusions/gmflow.py
    - 参考文件: GMFlow-main/lib/ops/gmflow_ops/gmflow_ops.py
    
    核心特性：
    1. 使用 GMFlow 的字典格式: {'means', 'logstds', 'logweights'}
    2. 使用 gm_to_mean() 和 gm_to_sample() 函数（如果可用）
    3. 适配 1D 特征向量（GMFlow 原为 2D 图像设计）
    4. 如果 CUDA 扩展不可用，自动回退到 PyTorch 实现
    
    数据格式适配：
    - GMFlow 原始格式: (bs, num_gaussians, out_channels, h, w) for 2D images
    - 本实现适配: (bs, num_gaussians, dim, 1, 1) for 1D features
    """
    def __init__(self, dim=2, n_potentials=5, epsilon=1, is_diagonal=True,
                 sampling_batch_size=1, S_diagonal_init=0.1, r_scale=1):
        super().__init__()
        self.is_diagonal = is_diagonal
        self.dim = dim
        self.n_potentials = n_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.sampling_batch_size = sampling_batch_size
        self.has_gmflow_ops = HAS_GMFLOW_OPS
        
        # GMFlow 格式：混合高斯的权重、均值和协方差
        # 使用 log_alpha (logweights), r (means), logstds
        self.log_alpha = nn.Parameter(torch.log(torch.ones(n_potentials)/n_potentials))
        self.r = nn.Parameter(torch.randn(n_potentials, dim))  # means
        self.r_scale = nn.Parameter(torch.ones(n_potentials, 1))
        
        # 协方差矩阵参数（使用对数空间以确保正定性）
        # 对于对角协方差，使用 logstds
        self.S_log_diagonal_matrix = nn.Parameter(torch.log(S_diagonal_init*torch.ones(n_potentials, self.dim)))
        self.S_rotation_matrix = nn.Parameter(
            torch.randn(n_potentials, self.dim, self.dim)
        )
        
        if not is_diagonal:
            geotorch.orthogonal(self, "S_rotation_matrix")
        
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

        self.gmmflow = GMMFlow(
            dim=config_dpdl.feature_dim,
            n_potentials=self.n_protos,
            epsilon=config_dpdl.sb_epsilon,
            is_diagonal=config_dpdl.sb_is_diagonal,
            sampling_batch_size=config_dpdl.sb_sampling_batch_size,
            S_diagonal_init=config_dpdl.sb_s_diagonal_init
        )
        
        self._initialize_gmmflow_with_etf_protos()
    
    def _initialize_gmmflow_with_etf_protos(self):
        """使用ETF原型初始化GMMFlow"""
        with torch.no_grad():
            self.gmmflow.init_r_by_samples(self.etf_protos)
            class_weights = []
            for i, count in enumerate(self.class_sample_counts):
                proto_count = self.protos_per_class[i]
                proto_weight = count / proto_count if proto_count > 0 else 1.0
                class_weights.extend([proto_weight] * proto_count)
            class_weights = torch.tensor(class_weights, device=self.gmmflow.r.device)
            class_weights = class_weights / torch.sum(class_weights)
            self.gmmflow.log_alpha.data = torch.log(class_weights + 1e-8)
            print(f"GMMFlow初始化: {self.n_protos}个原型, 权重范围 {class_weights.min():.4f}-{class_weights.max():.4f}")
    

    
    def _gaussian_mixture_nll_loss_1d(self, pred_means, target, pred_logstds, pred_logweights, eps=1e-4):
        """
        GMFlow 的混合高斯负对数似然损失（适配 1D 特征向量）
        严格参考: GMFlow-main/lib/models/losses/diffusion_loss.py 的 gaussian_mixture_nll_loss
        
        官方实现逻辑：
        - 2D: pred_means (bs, *, num_gaussians, c, h, w), target (bs, *, c, h, w)
        - 2D: gaussian_ll = (-0.5 * diff_weighted.square() - pred_logstds).sum(dim=-3)  # 对 c 维度求和
        - 1D适配: pred_means (bs, num_gaussians, dim), target (bs, dim)
        - 1D适配: gaussian_ll = (-0.5 * diff_weighted.square() - pred_logstds).sum(dim=-1)  # 对 dim 维度求和
        
        Args:
            pred_means: (bs, num_gaussians, dim) - 预测的均值
            target: (bs, dim) - 目标特征
            pred_logstds: (bs, num_gaussians, 1) - 预测的对数标准差（每个高斯一个logstd）
            pred_logweights: (bs, num_gaussians) - 预测的对数权重
        Returns:
            loss: (bs,) - 每个样本的损失
        """
        # 扩展 target 维度以匹配 pred_means: (bs, dim) -> (bs, 1, dim)
        # 对应官方: target.unsqueeze(-4) 将 (bs, *, c, h, w) -> (bs, *, 1, c, h, w)
        target = target.unsqueeze(1)  # (bs, 1, dim)
        
        # 计算损失（严格参考官方实现）
        # inverse_std: (bs, num_gaussians, 1) - 广播到 (bs, num_gaussians, dim)
        inverse_std = torch.exp(-pred_logstds).clamp(max=1 / eps)  # (bs, num_gaussians, 1)
        
        # diff_weighted: (bs, num_gaussians, dim)
        # 对应官方: (pred_means - target.unsqueeze(-4)) * inverse_std
        diff_weighted = (pred_means - target) * inverse_std  # (bs, num_gaussians, dim)
        
        # gaussian_ll: (bs, num_gaussians)
        # 严格对应官方: (-0.5 * diff_weighted.square() - pred_logstds).sum(dim=-3)
        # 官方对 dim=-3 求和（即 c 维度），1D适配对 dim=-1 求和（即 dim 维度）
        # pred_logstds 会自动广播到 (bs, num_gaussians, dim)，然后对 dim 维度求和
        gaussian_ll = (-0.5 * diff_weighted.square() - pred_logstds).sum(dim=-1)  # (bs, num_gaussians)
        
        # loss: (bs,) - logsumexp over num_gaussians
        # 对应官方: -torch.logsumexp(gaussian_ll + pred_logweights.squeeze(-3), dim=-3)
        loss = -torch.logsumexp(gaussian_ll + pred_logweights, dim=-1)  # (bs,)
        
        return loss
    
    def _compute_gmflow_transition_loss(self, features, class_protos, class_logweights, class_id):
        """
        计算 GMFlow transition loss（混合高斯负对数似然损失）
        参考: GMFlow-main/lib/models/diffusions/gmflow.py 的 loss 方法
        """
        if len(class_protos) == 0 or len(features) == 0:
            return torch.tensor(0.0, device=features.device)
        
        batch_size = features.shape[0]
        n_class_protos = class_protos.shape[0]
        epsilon = self.gmmflow.epsilon
        S = self.gmmflow.get_S()
        
        # 获取类特定的协方差
        start_idx = sum(self.protos_per_class[:class_id])
        S_class = S[start_idx:start_idx + n_class_protos]  # (n_class_protos, dim)
        
        # 计算 logstds（从协方差矩阵）
        if self.gmmflow.is_diagonal:
            # 对角协方差：每个维度独立，使用平均方差
            variances = epsilon * S_class  # (n_class_protos, dim)
            avg_variance = variances.mean(dim=-1)  # (n_class_protos,)
            logstds = 0.5 * torch.log(avg_variance + 1e-8)  # (n_class_protos,)
        else:
            # 非对角：使用平均特征值
            S_eigenvals = torch.linalg.eigvalsh(S_class).mean(dim=-1)  # (n_class_protos,)
            logstds = 0.5 * torch.log(epsilon * S_eigenvals + 1e-8)  # (n_class_protos,)
        
        # 扩展维度以匹配 GMFlow 格式
        # pred_means: (batch_size, n_class_protos, dim)
        pred_means = class_protos.unsqueeze(0).expand(batch_size, -1, -1)  # (bs, n_class_protos, dim)
        
        # pred_logstds: (batch_size, n_class_protos, 1) 或 (batch_size, 1, 1)
        pred_logstds = logstds.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # (bs, n_class_protos, 1)
        
        # pred_logweights: (batch_size, n_class_protos)
        pred_logweights = class_logweights.unsqueeze(0).expand(batch_size, -1)  # (bs, n_class_protos)
        
        # 计算 GMFlowNLLLoss
        loss = self._gaussian_mixture_nll_loss_1d(
            pred_means, features, pred_logstds, pred_logweights
        )
        
        return loss.mean()
    
    def _compute_proto_alignment_loss(self, features, class_protos):
        """
        计算原型对齐损失，确保特征明确映射到该类别的最近原型
        这对于多原型类别很重要，确保特征被分配到正确的原型
        
        Args:
            features: (bs, dim) - 类别特征
            class_protos: (n_class_protos, dim) - 该类别的原型
        Returns:
            loss: 标量 - 原型对齐损失（到最近原型的距离）
        """
        if len(class_protos) == 0 or len(features) == 0:
            return torch.tensor(0.0, device=features.device)
        
        # 计算特征到该类别的所有原型的距离
        distances = torch.cdist(features, class_protos)  # (bs, n_class_protos)
        
        # 找到每个特征到最近原型的距离
        min_distances = distances.min(dim=1)[0]  # (bs,) - 到最近原型的距离
        
        # 对齐损失：最小化到最近原型的距离
        # 这确保特征明确映射到该类别的某个原型
        alignment_loss = min_distances.mean()
        
        return alignment_loss
    
    def _get_gmflow_velocity(self, x_t, t, class_protos, class_logweights, class_id):
        """
        基于 GMFlow 官方实现计算 velocity field (drift)
        参考: GMFlow-main/lib/models/diffusions/gmflow.py 的 u_to_x_0 和 reverse_transition
        
        在 GMFlow 中，velocity u 的计算公式为：
        - 如果 prediction_type='u': x_0 = x_t - sigma * u
        - 所以 u = (x_t - x_0) / sigma
        
        对于 velocity matching，我们需要计算从 x_t 到目标原型的 velocity
        
        Args:
            x_t: (bs, dim) - 时间 t 的特征
            t: (bs,) - 时间步
            class_protos: (n_class_protos, dim) - 该类别的原型
            class_logweights: (n_class_protos,) - 该类别的权重
            class_id: 类别ID
            
        Returns:
            velocity: (bs, dim) - velocity field (drift)
        """
        batch_size = x_t.shape[0]
        n_class_protos = class_protos.shape[0]
        epsilon = self.gmmflow.epsilon
        num_timesteps = 1000  # GMFlow 默认时间步数，可以根据需要调整
        
        # 计算 sigma (参考 GMFlow 的 reverse_transition)
        t_expanded = t.reshape(-1, 1)  # (bs, 1)
        sigma = t_expanded / num_timesteps  # (bs, 1)
        sigma = sigma.clamp(min=1e-6)
        
        # 获取类特定的协方差
        start_idx = sum(self.protos_per_class[:class_id])
        S = self.gmmflow.get_S()
        S_class = S[start_idx:start_idx + n_class_protos]  # (n_class_protos, dim)
        
        # 计算 logstds（从协方差矩阵）
        if self.gmmflow.is_diagonal:
            variances = epsilon * S_class  # (n_class_protos, dim)
            avg_variance = variances.mean(dim=-1)  # (n_class_protos,)
            logstds = 0.5 * torch.log(avg_variance + 1e-8)  # (n_class_protos,)
        else:
            S_eigenvals = torch.linalg.eigvalsh(S_class).mean(dim=-1)  # (n_class_protos,)
            logstds = 0.5 * torch.log(epsilon * S_eigenvals + 1e-8)  # (n_class_protos,)
        
        # 构建 GMFlow 格式的字典
        # means: (bs, n_class_protos, dim)
        means = class_protos.unsqueeze(0).expand(batch_size, -1, -1)  # (bs, n_class_protos, dim)
        logweights = class_logweights.unsqueeze(0).expand(batch_size, -1)  # (bs, n_class_protos)
        logstds_expanded = logstds.unsqueeze(0).expand(batch_size, -1)  # (bs, n_class_protos)
        
        # 计算每个高斯分量的 x_0 预测
        # 对于每个原型，x_0 就是原型本身（在 velocity matching 中）
        # 但我们需要考虑混合高斯的加权平均
        # 参考 GMFlow 的 u_to_x_0: x_0 = x_t - sigma * u
        # 所以 u = (x_t - x_0) / sigma
        
        # 计算到每个原型的距离权重（基于 logweights）
        weights = torch.softmax(logweights, dim=1)  # (bs, n_class_protos)
        
        # 对于 velocity matching，目标 x_0 是加权平均的原型
        x_0_pred = (weights.unsqueeze(-1) * means).sum(dim=1)  # (bs, dim)
        
        # 计算 velocity: u = (x_t - x_0) / sigma
        # 参考 GMFlow 的 u_to_x_0 方法
        velocity = (x_t - x_0_pred) / sigma  # (bs, dim)
        
        return velocity
    
    def _compute_velocity_matching_loss(self, features, protos, class_logweights, class_id):
        """
        计算 Velocity Matching 损失，参考 LightSB 的 bridge matching 实现
        但使用 GMFlow 官方的 velocity/drift 计算方法
        
        参考:
        - LightSB: adpdl_longtail_lightsb_unbalance.py 的 _compute_bridge_matching_loss
        - GMFlow: GMFlow-main/lib/models/diffusions/gmflow.py 的 velocity 计算
        
        Args:
            features: (bs, dim) - ID 特征
            protos: (n_protos, dim) - 该类别的原型
            class_logweights: (n_protos,) - 该类别的权重
            class_id: 类别ID
            
        Returns:
            velocity_matching_loss: 标量损失
        """
        if len(protos) == 0 or len(features) == 0:
            return torch.tensor(0.0, device=features.device)
        
        batch_size = features.shape[0]
        epsilon = self.gmmflow.epsilon
        safe_t = 1e-2  # 避免 t=1 的数值问题
        
        # 找到每个特征对应的最近原型（作为目标）
        proto_distances = torch.cdist(features, protos)  # (bs, n_protos)
        proto_indices = proto_distances.argmin(dim=1)  # (bs,)
        target_protos = protos[proto_indices]  # (bs, dim)
        
        # 随机采样时间步 t
        t = torch.rand([batch_size], device=features.device) * (1 - safe_t)  # (bs,)
        
        # 构建中间状态 x_t（参考 LightSB 的实现）
        # x_t = target_protos * t + features * (1-t) + sqrt(epsilon * t * (1-t)) * noise
        noise = torch.randn_like(features)
        x_t = (target_protos * t[:, None] + 
               features * (1 - t[:, None]) + 
               torch.sqrt(epsilon * t[:, None] * (1 - t[:, None])) * noise)
        
        # 使用 GMFlow 方法计算 predicted velocity (drift)
        x_t_grad = x_t.requires_grad_(True)
        predicted_velocity = self._get_gmflow_velocity(
            x_t_grad, t, protos, class_logweights, class_id
        )
        
        # 计算 target velocity（参考 LightSB）
        # target_drift = (target_protos - x_t) / (1 - t)
        target_velocity = (target_protos - x_t_grad) / (1 - t[:, None] + 1e-8)
        
        # Velocity Matching Loss: MSE between predicted and target velocity
        velocity_matching_loss = F.mse_loss(predicted_velocity, target_velocity)
        
        return velocity_matching_loss
    
    def get_dpdl_loss(self, normal_features, labels=None):
        """
        计算 DPDL 损失，使用 GMFlow 的 transition loss (GMFlowNLLLoss)
        严格参考官方实现，使用 transition loss（官方标准损失）
        同时添加 velocity matching 损失
        
        参考: 
        - GMFlow-main/lib/models/diffusions/gmflow.py 的 loss 方法
        - LightSB: adpdl_longtail_lightsb_unbalance.py 的 bridge matching loss (velocity matching)
        """
        batch_size = normal_features.shape[0]
        accumulated_transition_loss = 0.0
        accumulated_velocity_matching_loss = 0.0
        
        # 按类别计算损失
        for class_id in torch.unique(labels):
            class_mask = (labels == class_id)
            if class_mask.sum() == 0:
                continue
                
            class_features = normal_features[class_mask] 
            
            start_idx = sum(self.protos_per_class[:class_id])
            end_idx = start_idx + self.protos_per_class[class_id]
            class_protos = self.etf_protos[start_idx:end_idx]  
            class_log_alpha = self.gmmflow.log_alpha[start_idx:end_idx]  
            class_logweights = class_log_alpha  # 使用 log_alpha 作为 logweights
            
            # 计算 GMFlow transition loss (GMFlowNLLLoss) - 官方标准损失
            transition_loss = self._compute_gmflow_transition_loss(
                class_features, class_protos, class_logweights, class_id
            )
            accumulated_transition_loss += transition_loss * class_mask.sum().float()
            
            # Velocity Matching Loss（参考 LightSB 的 bridge matching，但使用 GMFlow 的 velocity 计算）
            velocity_matching_loss = self._compute_velocity_matching_loss(
                class_features, class_protos, class_logweights, class_id
            )
            accumulated_velocity_matching_loss += velocity_matching_loss * class_mask.sum().float()

        loss_transition = accumulated_transition_loss / batch_size
        loss_velocity_matching = accumulated_velocity_matching_loss / batch_size
        
        # 主要使用 transition loss（官方标准损失）
        # velocity matching 损失作为辅助
        velocity_matching_weight = 0.1  # Velocity Matching 损失权重，可以调整
        total_loss = loss_transition + velocity_matching_weight * loss_velocity_matching
        
        return {
            'total_loss': total_loss,
            'transition': loss_transition,  # GMFlow transition loss (GMFlowNLLLoss)
            'velocity_matching': loss_velocity_matching  # Velocity Matching 损失（使用 GMFlow velocity）
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
        all_params.extend(self.net.gmmflow.parameters())
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
        transition_avg = 0.0
        velocity_matching_avg = 0.0
        
        for batch in tqdm(self.train_loader, desc=f'DPDL GMFlow Epoch {epoch_idx:03d}'):
            data = batch['data'].to(device)
            labels = batch['label'].to(device)
            
            features = self.net(data)
            
            loss_dict = self.net.get_dpdl_loss(features, labels)
            total_loss = loss_dict['total_loss'] 
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss_avg = total_loss_avg * 0.9 + float(total_loss) * 0.1
            transition_avg = transition_avg * 0.9 + float(loss_dict.get('transition', 0.0)) * 0.1
            velocity_matching_avg = velocity_matching_avg * 0.9 + float(loss_dict.get('velocity_matching', 0.0)) * 0.1
        
        self.scheduler.step()
        
        
        return {
            'epoch': epoch_idx,
            'total_loss': total_loss_avg,
            'transition': transition_avg,  # GMFlow transition loss (GMFlowNLLLoss)
            'velocity_matching': velocity_matching_avg,  # Velocity Matching 损失
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
              f'Transition: {train_metrics.get("transition", 0.0):.4f} | '
              f'Velocity Matching: {train_metrics.get("velocity_matching", 0.0):.4f}')

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
    print(f"GMMFlow epsilon: {config_dpdl.sb_epsilon}")
    print(f"原型最大数量: {config_dpdl.max_protos_per_class}")
    print("专注于GMMFlow最优传输和OOD检测")
    
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