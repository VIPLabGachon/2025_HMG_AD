#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rospy
import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
matplotlib.use('Agg')
import torch
from sensor_msgs.msg import CompressedImage
from morai_msgs.msg import CtrlCmd, EgoNoisyStatus, EgoVehicleStatus
from collections import deque
from io import BytesIO
from PIL import Image


import csv
import time
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from MORAI_UDP_NetworkModule.lib.network.UDP import Receiver
from MORAI_UDP_NetworkModule.lib.define.Camera import Camera
import threading
from MORAI_UDP_NetworkModule.lib.network.UDP import Sender
from MORAI_UDP_NetworkModule.lib.define.EgoCtrlCmd import EgoCtrlCmd
from utils.inference import generate_hd_map


import os
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

# (기존의 import 구문 유지)
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
    각 서브모듈을 거쳐 최종 제어 및 분류 출력을 생성합니다.
    """
    def __init__(self, config):
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

        # Backbone 초기화
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
        
        bev_embedding_config = {
            "sigma": 1.0,
            "bev_height": bev_h,
            "bev_width": bev_w,
            "h_meters": bev_h_meters,
            "w_meters": bev_w_meters,
            "offset": bev_offset,
            "decoder_blocks": decoder_blocks,
        }
        
        self.encoder = Encoder(
            backbone=self.backbone,
            cross_view=cross_view_config,
            bev_embedding=bev_embedding_config,
            dim=128,
            scale=1.0,
            middle=[2, 2],
        )
        
        input_dim = 256
        hidden_dim = 256
        output_dim = 256
        height, width = 25, 25
        self.bev_gru = BEVGRU(input_dim, hidden_dim, output_dim, height, width)
        
        self.feature_embedding = FeatureEmbedding(hidden_dim=32, output_dim=16)
        self.ego_gru = EgoStateGRU(input_dim=176, hidden_dim=256, output_dim=256, num_layers=1)
        self.ego_fusion = BEV_Ego_Fusion()

        self.hd_map_pipeline = HDMapFeaturePipeline(input_channels=7, final_channels=128, final_size=(25, 25))
        self.traffic_encoder = TrafficLightEncoder(feature_dim=128, pretrained=True)
        self.classification_head = TrafficSignClassificationHead(input_dim=128, num_classes=10)
        self.control = ControlMLP(future_steps=2, control_dim=3)
        self.ego_header = EgoStateHead(input_dim=256, hidden_dim=128, output_dim=12)
        self.bev_decoder = Decoder(dim=128, blocks=decoder_blocks, residual=True, factor=2)
    
    def forward(self, batch):
        hd_features = self.hd_map_pipeline(batch["hd_map"])
        ego_embedding = self.feature_embedding(batch["ego_info"])
        bev_output = self.encoder(batch)
        fusion_ego = self.ego_fusion(bev_output, ego_embedding)
        ego_gru_output, ego_gru_output_2 = self.ego_gru(fusion_ego)
        future_ego = self.ego_header(ego_gru_output)
        bev_decoding = self.bev_decoder(bev_output)
        concat_bev = torch.cat([hd_features, bev_output], dim=2)
        _, gru_bev = self.bev_gru(concat_bev)
        front_feature = self.traffic_encoder(batch["image"])
        classification_output = self.classification_head(front_feature)
        control_output = self.control(front_feature, gru_bev, ego_gru_output_2, batch["ego_info"])
        return {
            "control": control_output,
            "classification": classification_output,
            "future_ego": future_ego,
            "bev_seg": bev_decoding
        }
    
# ==============================
#      HD Map 및 JSON 파일 로드
# ==============================
global_path_file = "/home/v/vip_ws/src/hmg_2025/src/2025_HMG_AD/hd_sample/dataset/R_KR_PG_KATRI__HMG_Scenario_0/global_path.csv"
calibration_folder = "/home/v/vip_ws/src/hmg_2025/src/2025_HMG_AD/hd_sample/Calibration"
model_path = "/home/v/vip_ws/src/hmg_2025/src/2025_HMG_AD/end_to_end_model_test.pth"

ego_data_path = "/home/v/vip_ws/visualizations/ego_data.csv"

if not os.path.exists(ego_data_path):
    with open(ego_data_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = [
            "timestamp", "pos_x", "pos_y", "pos_z",
            "roll", "pitch", "yaw",
            "enu_vel_x", "enu_vel_y", "enu_vel_z",
            "vel_x", "vel_y", "vel_z",
            "ang_vel_roll", "ang_vel_pitch", "ang_vel_yaw",
            "acc_x", "acc_y", "acc_z",
            "rel_path_1", "rel_path_2", "rel_path_3", "rel_path_4", "rel_path_5",
            "rel_path_6", "rel_path_7", "rel_path_8", "rel_path_9", "rel_path_10"
        ]
        writer.writerow(header)

def save_ego_info(ego_info):
    """
    Ego 정보를 CSV 파일로 저장하는 함수
    """
    with open(ego_data_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        timestamp = time.time()
        writer.writerow([timestamp] + ego_info)
    # rospy.loginfo("[EGO] Saved ego info to CSV.")


# =============== 시퀀스 길이 T (여기서는 T=3) ===============
T = 3
sequence_queue = deque(maxlen=T)  # 각 프레임 데이터 저장

def load_calibration_data(calibration_folder, T=3):
    """
    캘리브레이션 데이터를 불러와 모델 입력 형식으로 변환합니다.
    
    Args:
        calibration_folder (str): 캘리브레이션 파일이 저장된 폴더 경로.
        T (int): 시계열 길이 (여기서는 T=3).
    
    Returns:
        dict: 모델 입력 딕셔너리 (intrinsic: (1, T, 5, 3, 3), extrinsic: (1, T, 5, 4, 4))
    """
    camera_count = 5
    intrinsic_keyword = "intrinsic"
    extrinsic_keyword = "extrinsic"

    intrinsic_queue = []
    extrinsic_queue = []

    for t in range(T):
        intrinsic_per_timestep = []
        extrinsic_per_timestep = []
        for i in range(1, camera_count + 1):  # CAMERA_1 ~ CAMERA_5
            intrinsic_file = os.path.join(calibration_folder, f"CAMERA_{i}__{intrinsic_keyword}.npy")
            extrinsic_file = os.path.join(calibration_folder, f"CAMERA_{i}__{extrinsic_keyword}.npy")

            if not os.path.exists(intrinsic_file) or not os.path.exists(extrinsic_file):
                raise FileNotFoundError(f"Missing calibration files for CAMERA_{i}: {intrinsic_file}, {extrinsic_file}")

            intrinsic_np = np.load(intrinsic_file)  # (3, 3)
            extrinsic_np = np.load(extrinsic_file)  # (4, 4)

            intrinsic_per_timestep.append(torch.tensor(intrinsic_np, dtype=torch.float32))
            extrinsic_per_timestep.append(torch.tensor(extrinsic_np, dtype=torch.float32))

        intrinsic_queue.append(torch.stack(intrinsic_per_timestep, dim=0))  # (5, 3, 3)
        extrinsic_queue.append(torch.stack(extrinsic_per_timestep, dim=0))  # (5, 4, 4)

    intrinsic_tensor = torch.stack(intrinsic_queue, dim=0)  # (T, 5, 3, 3)
    extrinsic_tensor = torch.stack(extrinsic_queue, dim=0)  # (T, 5, 4, 4)

    intrinsic_tensor = intrinsic_tensor.unsqueeze(0)  # (1, T, 5, 3, 3)
    extrinsic_tensor = extrinsic_tensor.unsqueeze(0)  # (1, T, 5, 4, 4)

    return {
        "intrinsic": intrinsic_tensor,
        "extrinsic": extrinsic_tensor
    }

def build_model_input_from_sequence(seq):
    """
    주어진 시퀀스(seq)는 길이 T=3의 리스트입니다.
      seq[i] = {"ego_info": (28,), "camera_data": (5,3,224,480), "hd_map": (6,150,150)}
    이를 배치 차원(B=1)을 포함한 모델 입력 딕셔너리로 묶어 반환합니다.
    """
    ego_list = []
    cam_list = []
    hd_map_list = []
    for frame in seq:
        ego_list.append(frame["ego_info"])      # (28,)
        cam_list.append(frame["camera_data"])     # (5,3,224,480)
        hd_map_list.append(frame["hd_map"])         # (6,150,150)

    ego_np    = np.stack(ego_list, axis=0)    # (T, 28)
    cam_np    = np.stack(cam_list, axis=0)    # (T, 5,3,224,480)
    hd_map_np = np.stack(hd_map_list, axis=0)  # (T, 6,150,150)

    ego_np    = np.expand_dims(ego_np, axis=0)    # (1, T, 28)
    cam_np    = np.expand_dims(cam_np, axis=0)    # (1, T, 5,3,224,480)
    hd_map_np = np.expand_dims(hd_map_np, axis=0)  # (1, T, 6,150,150)

    return {
        "ego_info": ego_np,
        "camera":   cam_np,
        "hd_map":   hd_map_np,
    }

# ========== UDP 설정 ==========
IP = "127.0.0.1"
PORTS = {
    "front": 1233,
    "left":  1234,
    "right": 1235,
    "rearL": 1236,
    "rearR": 1237
}
EGO_CTRL_PORT = 9093  # Ego Control UDP 포트

# UDP Receiver 객체 생성
udp_cameras = {name: Receiver(IP, port, Camera()) for name, port in PORTS.items()}
ego_ctrl_sender = Sender(IP, EGO_CTRL_PORT)

# 최신 카메라 데이터 저장 (초기값: None)
latest_camera_data = {name: None for name in PORTS.keys()}

# 전역 변수 (마지막 처리한 초(sec) 및 모델 객체)
last_processed_sec = None
model = None  # 이후 main()에서 초기화

def receive_udp_images():
    """
    5개 카메라의 최신 이미지를 UDP로 지속적으로 수신하여 latest_camera_data에 저장합니다.
    """
    global last_processed_sec

    while True:
        try:
            temp_data = {}  # 현재 sec에 해당하는 데이터를 임시 저장
            sec_value = None
            
            for name, receiver in udp_cameras.items():
                data = receiver.get_data()
                sec = data.image.sec

                # 이미 처리한 초(sec)는 건너뜁니다.
                if last_processed_sec is not None and sec == last_processed_sec:
                    continue

                if sec_value is None:
                    sec_value = sec
                if sec != sec_value:
                    continue

                # JPEG 압축 해제 및 이미지 리사이즈 (모델 입력에 맞춰 480x224)
                image = cv2.imdecode(np.frombuffer(data.image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    continue
                
                save_path = f"/home/v/vip_ws/visualizations/Camera/camera_{name}_{sec}.jpg"
                cv2.imwrite(save_path, image)

                image = cv2.resize(image, (480, 270))  # (width, height)
                cv2.imshow(f"MORAI Cam {name}", image)
                cv2.waitKey(1)

                image = image.astype(np.float32) / 255.0
                image = image.transpose(2, 0, 1)  # (3,224,480)

                temp_data[name] = image

            if len(temp_data) == 5:
                latest_camera_data.update(temp_data)
                last_processed_sec = sec_value

        except Exception as e:
            # 예외 발생 시 무시하고 계속 진행합니다.
            pass

        time.sleep(0.01)

def send_udp_control_output(output):
    """
    추론 결과(output)를 기반으로 Ego 차량 제어 명령을 UDP로 전송합니다.
    Args:
        output (torch.Tensor): (B, T, 3) 텐서 (accel, brake, steer 포함)
    """
    output_np = output.cpu().numpy()
    accel, brake, steer = output_np.squeeze()  # 마지막 시점의 제어값

    accel = np.clip(accel, 0, 1)
    brake = np.clip(brake, 0, 1)
    steer = np.clip(steer, -1, 1)
    
    ctrl_cmd = EgoCtrlCmd()
    ctrl_cmd.ctrl_mode = 2  # AutoMode
    ctrl_cmd.gear = 4       # Drive 모드
    ctrl_cmd.cmd_type = 1   # Throttle (accel, brake, steer)
    ctrl_cmd.accel = float(accel)
    ctrl_cmd.brake = float(brake)
    ctrl_cmd.steer = float(steer)

    ego_ctrl_sender.send(ctrl_cmd)
    rospy.loginfo(f"Sent UDP Control: accel={accel}, brake={brake}, steer={steer}")

def ego_callback(ego_msg):
    """
    Ego 상태 메시지가 들어올 때, 5개 카메라의 최신 이미지와 HD Map을 사용해 T=3 시퀀스를 구성,
    모델 추론 후 UDP 제어 명령을 전송합니다.
    """
    global model

    # (1) 모든 카메라 데이터가 준비되었는지 확인
    if any(latest_camera_data[cam] is None for cam in latest_camera_data):
        rospy.logwarn("[WARNING] Not all camera images are available yet.")
        return

    # (2) Ego 상태 정보 추출
    ego_position = [
        ego_msg.noisy_position.east,
        ego_msg.noisy_position.north,
        ego_msg.noisy_position.up
    ]
    ego_orientation = [
        ego_msg.noisy_orientation.roll,
        ego_msg.noisy_orientation.pitch,
        ego_msg.noisy_orientation.yaw
    ]
    ego_velocity = [
        ego_msg.noisy_velocity.x,
        ego_msg.noisy_velocity.y,
        ego_msg.noisy_velocity.z
    ]
    ego_control = [
        ego_msg.accel,
        ego_msg.brake,
        ego_msg.steer
    ]
    ego_position_hd = {
        "position" : [ego_msg.noisy_position.east,ego_msg.noisy_position.north],
        "heading"  : ego_msg.noisy_orientation.yaw
    }

    # rospy.loginfo(f"Ego position: {ego_position}")

    # 최종 ego 벡터 (길이=28)
    ego_info = (
        ego_position + ego_orientation +
        ego_velocity + ego_control
    )

    save_ego_info(ego_info)

    # (3) HD Map 생성 (Ego 기준)
    hd_map_tensor = generate_hd_map(global_path_file, ego_position_hd)  # [7, 200, 200]

    # (4) 카메라 5개 데이터 묶기, shape=(5,3,224,480)
    camera_data = np.array([
        latest_camera_data["front"],
        latest_camera_data["left"],
        latest_camera_data["right"],
        latest_camera_data["rearL"],
        latest_camera_data["rearR"]
    ])

    # (5) 현재 시점 frame_data 구성
    frame_data = {
        "ego_info":   np.array(ego_info, dtype=np.float32),  # (28,)
        "camera_data": camera_data,                           # (5,3,224,480)
        "hd_map":      hd_map_tensor                          # (6,150,150)
    }

    # (6) 시퀀스에 추가 (총 T=3 프레임)
    sequence_queue.append(frame_data)

    # (7) 시퀀스가 T=3이면 모델 입력 생성 및 추론 실행
    if len(sequence_queue) == T:
        seq_list = list(sequence_queue)  # 길이 T
        model_input_sequence = build_model_input_from_sequence(seq_list)
        model_input_calibration = load_calibration_data(calibration_folder, T=T)

        # 두 데이터 결합 (모델 입력 형식)
        model_input = {
            **model_input_sequence,       # 시퀀스 데이터: ego_info, camera, hd_map
            **model_input_calibration     # calibration 데이터: intrinsic, extrinsic
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # numpy 데이터를 torch.Tensor로 변환 후 device 이동
        camera_images = torch.tensor(model_input["camera"], dtype=torch.float32).to(device)
        intrinsics    = model_input["intrinsic"].to(device)
        extrinsics    = model_input["extrinsic"].to(device)
        hd_map_tensors= torch.tensor(model_input["hd_map"], dtype=torch.float32).to(device).squeeze(-1)
        ego_inputs    = torch.tensor(model_input["ego_info"], dtype=torch.float32).to(device)
        # print(camera_images.shape)
        # print(intrinsics.shape)
        # print(extrinsics.shape)
        # print(hd_map_tensors.shape)
        # print(ego_inputs.shape)
        # torch.Size([1, 3, 5, 3, 224, 480])
        # torch.Size([1, 3, 5, 3, 3])
        # torch.Size([1, 3, 5, 4, 4])
        # torch.Size([1, 3, 7, 200, 200])
        # torch.Size([1, 3, 12])
        batch = {
                "image": camera_images.to(device),           # [B, num_views, C, H, W]
                "intrinsics": intrinsics.to(device),   # [B, num_views, 3, 3]
                "extrinsics": extrinsics.to(device),   # [B, num_views, 4, 4]
                "hd_map": hd_map_tensors.to(device),
                "ego_info": ego_inputs.to(device),            # [B, 3, 12]
            }

        with torch.no_grad():
            output = model(
                batch
            )
        sequence_queue.clear()
        control = output["control"]     

        if output is not None:
            send_udp_control_output(control)

# =============== 메인 함수 ===============
def main():

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

    global model, cmd_pub

    rospy.init_node("faster_ego_method1_example", anonymous=True)

    # ROS 퍼블리셔 (필요시 사용)
    # cmd_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=10)

    # 모델을 한 번만 초기화 (노드 시작 시)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_local = EndToEndModel(config).to(device)
    model_local.load_state_dict(torch.load(model_path, map_location=device))
    model_local.eval()
    model = model_local  # 전역 변수에 저장

    # UDP 카메라 수신 스레드 시작
    udp_thread = threading.Thread(target=receive_udp_images, daemon=True)
    udp_thread.start()
    
    # Ego 상태 토픽 구독 (더 빠른 주기로 수신)
    rospy.Subscriber("/Ego_noisy_topic", EgoNoisyStatus, ego_callback)

    rospy.loginfo("Initialized. Using 'latest camera' method. Waiting for messages...")
    rospy.spin()

if __name__ == "__main__":
    main()
