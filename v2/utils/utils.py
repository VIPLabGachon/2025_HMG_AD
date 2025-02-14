import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List

def generate_grid(height: int, width: int):
    xs = torch.linspace(0, 1, width)
    ys = torch.linspace(0, 1, height)

    indices = torch.stack(torch.meshgrid((xs, ys), indexing='xy'), 0)       # 2 h w
    indices = F.pad(indices, (0, 0, 0, 0, 0, 1), value=1)                   # 3 h w
    indices = indices[None]                                                 # 1 3 h w

    return indices


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    """
    copied from ..data.common but want to keep models standalone
    """
    sh = h / h_meters
    sw = w / w_meters

    return [
        [ 0., -sw,          w/2.],
        [-sh,  0., h*offset+h/2.],
        [ 0.,  0.,            1.]
    ]


class Normalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()

        self.register_buffer('mean', torch.tensor(mean)[None, :, None, None], persistent=False)
        self.register_buffer('std', torch.tensor(std)[None, :, None, None], persistent=False)

    def forward(self, x):
        return (x - self.mean) / self.std


class RandomCos(nn.Module):
    def __init__(self, *args, stride=1, padding=0, **kwargs):
        super().__init__()

        linear = nn.Conv2d(*args, **kwargs)

        self.register_buffer('weight', linear.weight)
        self.register_buffer('bias', linear.bias)
        self.kwargs = {
            'stride': stride,
            'padding': padding,
        }

    def forward(self, x):
        return torch.cos(F.conv2d(x, self.weight, self.bias, **self.kwargs))


class BEVEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        sigma: int,
        bev_height: int,
        bev_width: int,
        h_meters: int,
        w_meters: int,
        offset: int,
        decoder_blocks: list,
    ):
        """
        Only real arguments are:

        dim: embedding size
        sigma: scale for initializing embedding

        The rest of the arguments are used for constructing the view matrix.

        In hindsight we should have just specified the view matrix in config
        and passed in the view matrix...
        """
        super().__init__()

        # each decoder block upsamples the bev embedding by a factor of 2
        h = bev_height // (2 ** len(decoder_blocks))
        w = bev_width // (2 ** len(decoder_blocks))

        # bev coordinates
        grid = generate_grid(h, w).squeeze(0)
        grid[0] = bev_width * grid[0]
        grid[1] = bev_height * grid[1]

        # map from bev coordinates to ego frame
        V = get_view_matrix(bev_height, bev_width, h_meters, w_meters, offset)  # 3 3
        V_inv = torch.FloatTensor(V).inverse()                                  # 3 3
        grid = V_inv @ rearrange(grid, 'd h w -> d (h w)')                      # 3 (h w)
        grid = rearrange(grid, 'd (h w) -> d h w', h=h, w=w)                    # 3 h w

        # egocentric frame
        self.register_buffer('grid', grid, persistent=False)                    # 3 h w
        self.learned_features = nn.Parameter(sigma * torch.randn(dim, h, w))    # d h w

    def get_prior(self):
        return self.learned_features
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dropout_rate=0.1):
        super(ConvBlock, self).__init__()
        self.conv3x3_1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv1x1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3x3_2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv3x3_1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv1x1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.conv3x3_2(x)))
        return x


class BEV_Ego_Fusion(nn.Module):
    """
    BEV feature와 Ego feature를 각각 입력받아,
    BEV는 Conv+Flatten+Linear로 [B,T, bev_dim]으로 변환하고,
    Ego와 concat([B,T, bev_dim + ego_dim])한 결과를 반환.
    """
    def __init__(
        self,
        bev_in_channels=128,  # BEV feature의 입력 채널
        bev_mid_channels=64,
        bev_out_channels=128,
        bev_dim=112,          # BEV를 최종적으로 flatten+linear 했을 때의 차원
        ego_dim=112,          # ego feature의 차원
        H=25, W=25            # BEV feature의 공간 크기
    ):
        super(BEV_Ego_Fusion, self).__init__()
        
        # 1) BEV feature를 위한 conv 블록
        self.conv_block = ConvBlock(bev_in_channels, bev_mid_channels, bev_out_channels)
        
        # 2) Conv 블록의 출력 shape: [B*T, bev_out_channels, H, W]
        #    flatten할 차원 = bev_out_channels * H * W
        self.flatten_dim = bev_out_channels * H * W
        
        # 3) flatten 후 원하는 차원(bev_dim)으로 줄이기 위한 FC
        self.fc = nn.Linear(self.flatten_dim, bev_dim)
        
        # 4) ego_dim은 그대로 사용. (별도의 처리 없이 concat)
        self.ego_dim = ego_dim
        self.bev_dim = bev_dim

    def forward(self, bev_feature, ego_feature):
        """
        Args:
            bev_feature: [B, T, bev_in_channels, H, W]
            ego_feature: [B, T, ego_dim]
        Returns:
            concat_vector: [B, T, bev_dim + ego_dim]
        """
        B, T, C, H, W = bev_feature.shape
        
        # (A) BEV feature 처리
        # 시간 차원(T)을 배치 차원에 합침: [B*T, C, H, W]
        x = bev_feature.view(B * T, C, H, W)
        # Conv block
        x = self.conv_block(x)  # [B*T, bev_out_channels, H, W]
        # Flatten
        x = x.reshape(B * T, -1)   # [B*T, flatten_dim]
        # Linear로 bev_dim으로 매핑
        bev_vec = self.fc(x)    # [B*T, bev_dim]
        # 다시 (B, T) 형태로 복원
        bev_vec = bev_vec.view(B, T, self.bev_dim)  # [B, T, bev_dim]
        
        # Concatenate
        concat_vector = torch.cat([bev_vec, ego_feature], dim=-1)  # [B, T, bev_dim + ego_dim]
        
        return concat_vector

