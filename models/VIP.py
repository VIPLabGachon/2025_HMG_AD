import torch
import torch.nn as nn
from torchvision.models import resnet18
from models.backbones.efficientnet import EfficientNetExtractor
from models.encoder import SequenceEncoder, SingleFrameEncoder, TemporalTransformer
from models.BEVHDmapFusionNet import BEVHDMapFusionNet
from models.decoder import BEVDecoderWithStackedGRU
from torchvision.models import ResNet18_Weights

class FrontCameraProcessorWithAttention(nn.Module):
    def __init__(self, output_dim: int = 256, pretrained: bool = True):
        super().__init__()
        # ResNet 백본
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        # Feature map 채널 크기 조정
        self.conv = nn.Conv2d(512, output_dim, kernel_size=1)

        # Spatial Attention Layer
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(output_dim, 1, kernel_size=3, padding=1),  # 채널 축소
            nn.Sigmoid()  # 중요도 맵 생성
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)  # (B*T, C, H, W)

        # ResNet feature map 추출
        features = self.feature_extractor(x)  # (B*T, 512, H_out, W_out)

        # 채널 크기 조정
        features = self.conv(features)  # (B*T, output_dim, H_out, W_out)

        # Spatial Attention 적용
        attention_map = self.spatial_attention(features)  # (B*T, 1, H_out, W_out)
        features = features * attention_map  # (B*T, output_dim, H_out, W_out)

        # (B, T, output_dim, H_out, W_out)로 복원
        feature_map = features.view(B, T, features.size(1), features.size(2), features.size(3))

        return feature_map


class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # (1) Backbone & Config
        backbone = EfficientNetExtractor(
            layer_names=['reduction_2', 'reduction_4'],
            image_height=224,
            image_width=480,
            model_name='efficientnet-b4'
        )

        cross_view_config = {
            "heads": 4,
            "dim_head": 32,
            "qkv_bias": True,
            "skip": True,
            "no_image_features": False,
            "image_height": 224,
            "image_width": 480,
        }

        bev_embedding_config = {
            "sigma": 1.0,
            "bev_height": 200,
            "bev_width": 200,
            "h_meters": 100.0,
            "w_meters": 100.0,
            "offset": 0.0,
            "decoder_blocks": [2, 2],
        }

        # (2) Encoders
        single_frame_encoder = SingleFrameEncoder(
            backbone=backbone,
            cross_view=cross_view_config,
            bev_embedding=bev_embedding_config,
            dim=128
        )
        temporal_transformer = TemporalTransformer(d_model=128, nhead=8, num_layers=2)

        # 시퀀스 인코더 (BEV Feature 추출)
        self.sequence_encoder = SequenceEncoder(
            base_encoder=single_frame_encoder,
            temporal_module=temporal_transformer
        )

        # (3) Fusion & Decoder
        self.bev_fusion_net = BEVHDMapFusionNet(
            bev_dim=128,
            hd_map_dim=6,
            front_view_dim=256,
            ego_dim=28,
            fused_dim=64,
            output_dim=32
        )
        self.bev_decoder = BEVDecoderWithStackedGRU(
            input_channels=32,  # Fusion 후의 output_dim
            hidden_size=128,
            seq_len=2,         # T 길이에 맞춰 조정
            spatial_dim=50,     # fusion_output의 H, W에 맞춰 조정
            num_layers=3,
            control_dim=3      
        )
        
        # 전면 카메라 프로세서
        self.front_camera_processor = FrontCameraProcessorWithAttention(output_dim=256)

    def forward(self,
                camera_images: torch.Tensor,
                intrinsics: torch.Tensor,
                extrinsics: torch.Tensor,
                hd_map_tensors: torch.Tensor,
                ego_inputs: torch.Tensor):
        """
        Args:
            camera_images :  (B, T, N_cam, C, H, W)
            intrinsics    :  (B, T, N_cam, 3, 3)
            extrinsics    :  (B, T, N_cam, 4, 4)
            hd_map_tensors:  (B, T, C_map, H_map, W_map)
            ego_inputs    :  (B, T, ego_dim)

        Returns:
            output: (B, T, Control)
        """
        # 전면 카메라 이미지 추출 (N_cam의 첫 번째 카메라를 전면 카메라로 가정)
        front_camera_images = camera_images[:, :, 0]  # (B, T, C, H, W)

        # BEV Feature 추출
        batch_dict = {
            "image": camera_images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics
        }
        bev_features = self.sequence_encoder(batch_dict)  # (B, T, D, H, W)

        # 전면 카메라 feature map 추출
        front_camera_features = self.front_camera_processor(front_camera_images)
        # (B, T, 256, H_out, W_out)

        # Fusion (BEV Feature + HD Map + Ego Info + Front Camera Feature)
        # (B, T, out_dim=32, H, W) 리턴
        fusion_output = self.bev_fusion_net(bev_features, hd_map_tensors, ego_inputs, front_camera_features)

        # 디코더는 (B, out_dim=32, T, H, W) 형태를 기대하므로 permute
        fusion_output = fusion_output.permute(0, 2, 1, 3, 4)  # (B, 32, T, H, W)

        # (B, T, Control) 형태로 예측
        output = self.bev_decoder(fusion_output)

        return output
