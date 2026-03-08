import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import DropPath

class Mlp(nn.Module):
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


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        B, N, C = x.shape
        num_heads = self.num_heads
        head_dim = C // num_heads

        # 生成Q/K
        qk = self.qk(x).reshape(B, N, 2, C).permute(2, 0, 1, 3)
        q, k = qk[0], qk[1]  # 各 [B, N, C]

        # 激活处理
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0

        # 分割多头
        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        v = x.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]

        # 注意力计算
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)  # 归一化因子
        kv = k.transpose(-2, -1) @ v  # [B, H, D, D]
        x = (q @ kv) * z  # [B, H, N, D]

        # 输出投影
        x = x.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        v = v.transpose(1, 2).reshape(B, N, C).permute(0, 2, 1)  # [B, C, N]
        x = x + self.lepe(v).permute(0, 2, 1)  # 加入局部增强
        return x


class MLLABlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4., qkv_bias=True,
                 drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # 卷积位置增强
        self.cpe1 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)

        # 注意力分支
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 二次位置增强
        self.cpe2 = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, C, N = x.shape  # 输入格式 [B, C, N]
        x = x.permute(0, 2, 1)  # 转换为 [B, N, C]

        # 第一层位置增强
        x = x + self.cpe1(x.permute(0, 2, 1)).permute(0, 2, 1)
        shortcut = x

        # 注意力分支
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).permute(0, 2, 1)
        x = self.act(self.dwc(x)).permute(0, 2, 1)
        x = self.attn(x)
        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)

        # 第二层位置增强
        x = x + self.cpe2(x.permute(0, 2, 1)).permute(0, 2, 1)

        # MLP分支
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x.permute(0, 2, 1)


class DimensionalAdapter(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64):
        super().__init__()
        # 升维层（修正维度顺序）
        self.up_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        # 核心处理模块（添加通道对齐）
        self.mlla = MLLABlock(
            dim=hidden_dim,  # 必须等于hidden_dim
            num_heads=4,
        )
        # 降维层（保持维度一致）
        self.down_proj = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim)
        )

    def forward(self, x):
        # 原始输入: (B, 48, 11)
        x = self.up_proj(x)  # → (B,48,64)

        # 关键修正：调整维度顺序
        x = x.permute(0, 2, 1)  # → (B,64,48) [通道优先]
        x = self.mlla(x)  # → (B,64,48)
        x = x.permute(0, 2, 1)  # → (B,48,64) [恢复顺序]

        return self.down_proj(x)  # → (B,48,11)

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class CCU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class MAU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        #print(up.shape)
        Y1 = self.GWC(up) + self.PWC1(up)
        #print(Y1.shape)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class MAU_NoAttn(nn.Module):
    '''
    去除注意力机制的MAU模块（消融实验用）
    alpha: 0<alpha<1，用于分割高/低通道
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        # 1. 通道分割参数（与原MAU完全一致，保证变量定义不变）
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel

        # 2. 通道挤压卷积（与原MAU完全一致，保留特征维度压缩逻辑）
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)

        # 3. 高通道变换分支（GWC+PWC1，与原MAU完全一致）
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)

        # 4. 低通道变换分支（PWC2+特征拼接，与原MAU完全一致）
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)

        # -------------------------- 关键修改：删除注意力相关层（advavg） --------------------------
        # 原MAU中的self.advavg = nn.AdaptiveAvgPool2d(1) 在此处删除，不再保留注意力计算组件

    def forward(self, x):
        # 1. 通道分割（与原MAU完全一致：按up_channel/low_channel分割输入x）
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)

        # 2. 通道挤压（与原MAU完全一致：压缩高/低通道维度）
        up, low = self.squeeze1(up), self.squeeze2(low)

        # 3. 特征变换（与原MAU完全一致：高通道GWC+PWC1，低通道PWC2+拼接）
        Y1 = self.GWC(up) + self.PWC1(up)  # 高通道变换结果
        Y2 = torch.cat([self.PWC2(low), low], dim=1)  # 低通道变换结果

        # 4. 特征融合（-------------------------- 关键修改：删除注意力加权步骤 --------------------------）
        # 原MAU中「out = F.softmax(self.advavg(out), dim=1) * out」的注意力加权逻辑全部删除
        out = torch.cat([Y1, Y2], dim=1)  # 仅保留特征拼接，不做注意力权重分配

        # 5. 输出合并（与原MAU完全一致：分割拼接后的特征并求和）
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2

class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.CCU = CCU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.MAU = MAU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)
        self.MAU_NoAttn = MAU_NoAttn(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.CCU(x)
        x = self.MAU(x)
        return x






if __name__ == '__main__':
    x = torch.randn(3, 32, 64, 64) # 输入 B C H W
    model = ScConv(32)
    print(model(x).shape)