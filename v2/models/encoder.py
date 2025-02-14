import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from torchvision import models
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from typing import List
from utils.attention import CrossAttention, CrossViewAttention
from utils.utils import generate_grid, get_view_matrix, Normalize, RandomCos, BEVEmbedding

ResNetBottleNeck = lambda c: Bottleneck(c, c // 4)

class Encoder(nn.Module):
    def __init__(
            self,
            backbone,
            cross_view: dict,
            bev_embedding: dict,
            dim: int = 128,
            middle: List[int] = [2, 2],
            scale: float = 1.0,
    ):
        super().__init__()

        self.norm = Normalize()
        self.backbone = backbone

        if scale < 1.0:
            self.down = lambda x: F.interpolate(x, scale_factor=scale, recompute_scale_factor=False)
        else:
            self.down = lambda x: x

        assert len(self.backbone.output_shapes) == len(middle)

        cross_views = list()
        layers = list()

        for feat_shape, num_layers in zip(self.backbone.output_shapes, middle):
            _, feat_dim, feat_height, feat_width = self.down(torch.zeros(feat_shape)).shape

            cva = CrossViewAttention(feat_height, feat_width, feat_dim, dim, **cross_view)
            cross_views.append(cva)

            layer = nn.Sequential(*[ResNetBottleNeck(dim) for _ in range(num_layers)])
            layers.append(layer)

        self.bev_embedding = BEVEmbedding(dim, **bev_embedding)
        self.cross_views = nn.ModuleList(cross_views)
        self.layers = nn.ModuleList(layers)

    def forward(self, batch):
        b, t, n, c, h, w = batch['image'].shape  # Batch, Time, Cameras, Channels, Height, Width

        # Combine batch, time, and cameras into one dimension for backbone processing
        image = batch['image'].view(b * t * n, c, h, w)  # (b*t*n, c, h, w)
        I_inv = batch['intrinsics'].view(b * t * n, *batch['intrinsics'].shape[3:]).inverse()  # (b*t*n, 3, 3)
        E_inv = batch['extrinsics'].view(b * t * n, *batch['extrinsics'].shape[3:]).inverse()  # (b*t*n, 4, 4)

        I_inv = rearrange(I_inv, '(bt n) ... -> bt n ...', bt=b*t, n=n)
        # print(I_inv.shape) => torch.Size([4, 5, 3, 3])
        E_inv = rearrange(E_inv, '(bt n) ... -> bt n ...', bt=b*t, n=n)
        # print(E_inv.shape) => torch.Size([4, 5, 4, 4])

        # Normalize and process image features using backbone
        features = [self.down(y) for y in self.backbone(self.norm(image))]  # Backbone features
        # Initialize BEV embedding
        x = self.bev_embedding.get_prior()  # (d, H, W)
        x = repeat(x, '... -> b ...', b=b * t)  # (b*t, d, H, W)

        for cross_view, feature, layer in zip(self.cross_views, features, self.layers):
            # Rearrange feature to split batch, time, and cameras
            feature = rearrange(feature, '(b t n) ... -> b t n ...', b=b, t=t, n=n)  # (b, t, n, ..., h, w)
            # print(feature.shape) => torch.Size([1, 4, 5, 32, 68, 120])
            
            # Combine batch and time for cross-view attention
            feature = feature.view(b * t, n, *feature.shape[3:])  # (b*t, n, ..., h, w)
            # print(feature.shape) => torch.Size([4, 5, 32, 68, 120])
            x = cross_view(x, self.bev_embedding, feature, I_inv, E_inv)

            # Apply convolutional layers
            x = layer(x)

        # Separate batch and time dimensions
        x = x.view(b, t, *x.shape[1:])  # (b, t, d, H, W)
        # print(x.shape) => torch.Size([1, 4, 128, 32, 32])
        return x

class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmbeddingMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class FeatureEmbedding(nn.Module):
    def __init__(self, hidden_dim=32, output_dim=16):
        super(FeatureEmbedding, self).__init__()
        # 각 입력 그룹에 대해 MLP 정의
        self.position_mlp = EmbeddingMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=output_dim)
        self.orientation_mlp = EmbeddingMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=output_dim)
        self.velocity_mlp = EmbeddingMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=output_dim)
        self.scalar_mlp = EmbeddingMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=output_dim)  # accel, brake, steer
    
    def forward(self, data):

        batch_size, time_steps, _ = data.shape
        data = data.view(batch_size * time_steps, -1)  # Flatten batch and time for consistent processing

        position = data[:, :3]
        orientation = data[:, 3:6]
        velocity = data[:, 6:9]
        scalars = data[:, 9:12]  # Ensure correct slicing
        
        # 각 그룹별 MLP 처리
        position_embed = self.position_mlp(position)
        orientation_embed = self.orientation_mlp(orientation)
        velocity_embed = self.velocity_mlp(velocity)
        scalar_embed = self.scalar_mlp(scalars)
        
        # 임베딩 결합
        combined = torch.cat((
            position_embed,
            orientation_embed,
            velocity_embed,
            scalar_embed
        ), dim=1)
        
        combined = combined.view(batch_size, time_steps, -1) 
        
        return combined

class TrafficLightEncoder(nn.Module):
    def __init__(self, feature_dim=256, pretrained=True):
        super(TrafficLightEncoder, self).__init__()
        self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Remove final classification layer
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(num_features, feature_dim)

    def forward(self, x):
        # Select the front camera (index 0 along camera dimension) and current time (index 2 along time dimension)
        current_time_frame = x[:, 2, 0, :, :, :]  # Shape: [batch, channels, height, width]

        # Extract features using ResNet-50
        features = self.backbone(current_time_frame)  # Shape: [batch, num_features]

        # Reduce dimensionality
        features = self.dropout(self.fc(features))  # Shape: [batch, feature_dim]

        return features

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet 기반의 HD Map Feature Extractor
    """
    def __init__(self, input_channels=6, output_channels=2048):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # 첫 번째 Conv 레이어 수정
        resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # ResNet Feature Extractor

    def forward(self, x):
        return self.feature_extractor(x)


class FeatureAdjuster(nn.Module):
    """
    ResNet 출력 크기 조정을 위한 모듈
    - 채널 수 축소: 2048 -> 128
    - 해상도 조정: 8x8 -> 32x32
    """
    def __init__(self, input_channels=2048, output_channels=128, output_size=(32, 32)):
        super(FeatureAdjuster, self).__init__()
        self.channel_match = nn.Conv2d(input_channels, output_channels, kernel_size=1)  # 채널 축소
        self.output_size = output_size

    def forward(self, x):
        # 채널 축소
        x = self.channel_match(x)
        # 해상도 조정
        x = F.interpolate(x, size=self.output_size, mode='bilinear', align_corners=False)
        return x


class HDMapFeaturePipeline(nn.Module):
    """
    HD Map 데이터로부터 최종 Feature 추출
    """
    def __init__(self, input_channels=6, resnet_output_channels=2048, final_channels=128, final_size=(25, 25)):
        super(HDMapFeaturePipeline, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor(input_channels, resnet_output_channels)
        self.feature_adjuster = FeatureAdjuster(resnet_output_channels, final_channels, final_size)

    def forward(self, x):
        batch_size, time_steps, channels, height, width = x.shape  # 입력 텐서의 크기 추출
        x = x.flatten(0, 1)  # Batch와 Time 차원을 결합 -> [batch*time, channels, height, width]

        # ResNet으로 Feature 추출
        x = self.feature_extractor(x)  # [batch*time, 2048, H', W']

        # Feature 크기 조정
        x = self.feature_adjuster(x)  # [batch*time, 128, 32, 32]

        # 배치와 시간 차원을 다시 분리
        x = x.view(batch_size, time_steps, 128, 25, 25)  # [batch, time, channels, height, width]
        return x