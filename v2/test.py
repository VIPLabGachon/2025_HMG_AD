import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.encoder import Encoder, HDMapFeaturePipeline, FeatureEmbedding, TrafficLightEncoder
from models.decoder import TrafficSignClassificationHead, EgoStateHead, Decoder
from models.GRU import BEVGRU, EgoStateGRU 
from models.backbones.efficientnet import EfficientNetExtractor
from models.control import FutureControlMLP, ControlMLP
from utils.attention import FeatureFusionAttention
from utils.utils import BEV_Ego_Fusion
from dataloader.dataloader import camDataLoader


class EndToEndModel(nn.Module):
    """
    End-to-End 모델 클래스.
    입력 배치에서 이미지, intrinsics, extrinsics, HD map, ego_info 등을 받아
    각 서브모듈(HD map 파이프라인, Ego GRU, BEV Encoder, Front-view Encoder,
    Fusion Attention, 분류 및 제어 헤드)을 거쳐 최종 제어 및 분류 출력을 생성합니다.
    """
    def __init__(self, config):
        """
        config: dict 형태의 설정값. (이미지 크기, BEV 크기, decoder_blocks 등)
        """
        super(EndToEndModel, self).__init__()
        # 기본 설정
        image_h = config.get("image_h", 270)
        image_w = config.get("image_w", 480)
        bev_h = config.get("bev_h", 200)
        bev_w = config.get("bev_w", 200)
        bev_h_meters = config.get("bev_h_meters", 50)
        bev_w_meters = config.get("bev_w_meters", 50)
        bev_offset = config.get("bev_offset", 0)
        decoder_blocks = config.get("decoder_blocks", [128, 128, 64])

        # Backbone 초기화 (EfficientNetExtractor) / 1634MB
        self.backbone = EfficientNetExtractor(
            model_name="efficientnet-b4",
            layer_names=["reduction_2", "reduction_4"],
            image_height=image_h,
            image_width=image_w,
        )
        
        cross_view_config = {
            "heads": 4,
            "dim_head": 32,
            "qkv_bias": True,
            "skip": True,
            "no_image_features": False,
            "image_height": image_h,
            "image_width": image_w,
        }
        
        # BEVEmbedding 관련 설정
        bev_embedding_config = {
            "sigma": 1.0,
            "bev_height": bev_h,
            "bev_width": bev_w,
            "h_meters": bev_h_meters,
            "w_meters": bev_w_meters,
            "offset": bev_offset,
            "decoder_blocks": decoder_blocks,
        }
        
        # Encoder 초기화 / 1636MB(+2MB)
        self.encoder = Encoder(
            backbone=self.backbone,
            cross_view=cross_view_config,
            bev_embedding=bev_embedding_config,
            dim=128,
            scale=1.0,
            middle=[2, 2],
        )
        
        # BEV GRU 모델 초기화 / 1980MB(+344MB)
        # 입력 채널 수: 256 (hd map feature 128 + encoder output 128)
        # 출력 채널 수: 128
        input_dim = 256
        hidden_dim = 256
        output_dim = 256
        height, width = 25, 25
        self.bev_gru = BEVGRU(input_dim, hidden_dim, output_dim, height, width)
        
        # # Ego GRU 모델 및 Feature Embedding 초기화 / 1980MB(+0MB)
        self.feature_embedding = FeatureEmbedding(hidden_dim=32, output_dim=16)
        self.ego_gru = EgoStateGRU(input_dim=176, hidden_dim=256, output_dim=256, num_layers=1)
        
        # Ego + BEV Fusion 
        self.ego_fusion = BEV_Ego_Fusion()

        # HD Map Feature Pipeline 초기화 / 2076MB(+96MB)
        self.hd_map_pipeline = HDMapFeaturePipeline(input_channels=7, final_channels=128, final_size=(25, 25))
        
        # Front-view (Traffic) Encoder 초기화 / 2176MB(+100MB)
        self.traffic_encoder = TrafficLightEncoder(feature_dim=128, pretrained=True)
        
        # # Feature Fusion Attention 초기화
        # self.fusion_model = FeatureFusionAttention(feature_dim=128, bev_dim=128, time_steps=2, spatial_dim=32)
        
        # Traffic Sign Classification Head 초기화 / 2176MB(+0MB)
        self.classification_head = TrafficSignClassificationHead(input_dim=128, num_classes=10)
        
        # Future Control Head (MLP) 초기화 / 2176MB(+0MB)
        self.control = ControlMLP(future_steps=2, control_dim=3)
        
        # Future Ego Head 초기화 / 2176MB(+0MB)
        self.ego_header = EgoStateHead(input_dim=256, hidden_dim=128, output_dim=21)
        
        # BEV decoder 초기화 / 2176MB(+0MB)
        self.bev_decoder = Decoder(dim=128, blocks=decoder_blocks, residual=True, factor=2)
    
    def forward(self, batch):
        """
        batch: dict with keys
            - "image": [B, num_views, C, H, W]
            - "intrinsics": [B, num_views, 3, 3]
            - "extrinsics": [B, num_views, 4, 4]
            - "hd_map": [B, time_steps, ...]
            - "ego_info": [B, ...]
        """
        # HD Map Encoding
        hd_features = self.hd_map_pipeline(batch["hd_map"])  
        # hd_features shape: [B, time_steps, 128, 18, 18]

        # Ego Encoding
        ego_embedding = self.feature_embedding(batch["ego_info"])  
        # ego_embedding shape: [B, seq_len, 64]

        # BEV Encoding
        bev_output = self.encoder(batch)  
        # bev_output shape: [B, time_steps, 128, 25, 25]

        # fusion ego + bev
        fusion_ego = self.ego_fusion(bev_output, ego_embedding)
        # fusion_ego shape = [B, time_steps, 224]

        ego_gru_output, ego_gru_output_2 = self.ego_gru(fusion_ego)  
        # ego_gru_output shape: [B, present_step + future_steps, 256]

        # Ego decoding
        future_ego = self.ego_header(ego_gru_output)
        # future_ego shape = [B, future_steps, 21]
        
        # BEV Decoding
        bev_decoding = self.bev_decoder(bev_output)
        # bev_decoding shape: [B, time_steps, 8, 200, 200]

        # HD map feature와 BEV feature를 채널 차원에서 concat (256 채널)
        concat_bev = torch.cat([hd_features, bev_output], dim=2)  
        # concat_bev shape: [B, time_steps, 256, 18, 18]
        
        # GRU를 통해 과거, 현재, 미래 정보를 추출
        _, gru_bev = self.bev_gru(concat_bev)  
        # gru_bev shape: [B, future_steps, 256, 25, 25]

        # Front-view Encoding (Traffic Light 관련 특징 추출)
        front_feature = self.traffic_encoder(batch["image"])  
        # front_feature shape: [B, 128]

        # Traffic Sign Classification Head (front feature 사용)
        classification_output = self.classification_head(front_feature)  
        # classification_output shape: [B, num_classes]

        # Future Control 예측 (fusion된 feature 사용)
        control_output = self.control(front_feature, gru_bev, ego_gru_output_2, batch["ego_info"])
        # control_output shape: [B, 3]

        return {
            "control": control_output,                      # control_output shape: [B, 3]
            "classification": classification_output,        # classification_output shape: [B, num_classes]
            "future_ego": future_ego,                       # future_ego shape = [B, future_steps, 21]
            "bev_seg": bev_decoding                         # bev_decoding shape: [B, time_steps, 64, 144, 144]
        }


def get_dataloader(root_dir, num_timesteps=3, batch_size=1):
    """
    데이터셋과 DataLoader 초기화 함수.
    """
    dataset = camDataLoader(root_dir, num_timesteps=num_timesteps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def main():
    # device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 설정 값
    config = {
        "image_h": 270,
        "image_w": 480,
        "bev_h": 200,
        "bev_w": 200,
        "bev_h_meters": 50,
        "bev_w_meters": 50,
        "bev_offset": 0,
        "decoder_blocks": [128, 128, 64],
    }
    
    # End-to-End 모델 초기화
    model = EndToEndModel(config).to(device)

    # 데이터 로더 초기화 (root_dir 경로는 실제 데이터셋 경로로 수정)
    root_dir = "/home/vip/hd/Dataset"
    dataloader = get_dataloader(root_dir, num_timesteps=3, batch_size=1)
    
    # 배치 단위로 forward pass 수행
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # 데이터의 각 항목을 device로 이동
            batch = {
                "image": data["images_input"].to(device),           # [B, num_views, C, H, W]
                "intrinsics": data["intrinsic_input"].to(device),   # [B, num_views, 3, 3]
                "extrinsics": data["extrinsic_input"].to(device),   # [B, num_views, 4, 4]
                "hd_map": data["hd_map_input"].to(device),
                "ego_info": data["ego_info"].to(device),            # [B, 3, 12]
            }

            ego_info_future_gt = data["ego_info_future"].to(device) # [B, 2, 12]
            bev_seg_gt = data["gt_hd_map_input"].to(device)         # [B, T, 8, 144, 144]
            traffic_gt = data["traffic"].to(device)                 # [B, C]
            control_gt = data["control"].to(device)                 # [B, 3]
            outputs = model(batch)
            future_ego = outputs["future_ego"]                      # [B, 2, 21]
            bev_seg = outputs["bev_seg"]                            # [B, T, 8, 144, 144]
            traffic = outputs["classification"]                     # [B, C]
            control = outputs["control"]                            # [B, 3]       
            print(f"Batch {batch_idx}: control shape = {control.shape}, classification shape = {traffic.shape}")

if __name__ == "__main__":
    main()