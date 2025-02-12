#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
(1) 정적 HD Map 데이터 로드
(2) 카메라 5개 + Ego 토픽 수신
(3) 각 카메라의 '가장 최신' 이미지를 보관
(4) Ego 콜백에서 HD Map 시각화, T=2 시퀀스로 묶어 모델 입력 생성
"""
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
from models.VIP import CombinedModel

# ==============================
#   HD Map 및 JSON 파일 로드
# ==============================
road_mesh_file = "/home/v/vip_ws/src/hmg_2025/src/2025_HMG_AD/hd_sample/Map_Data/R_KR_PG_KATRI/road_mesh_out_line_polygon_set.json"
lane_boundary_file = "/home/v/vip_ws/src/hmg_2025/src/2025_HMG_AD/hd_sample/Map_Data/R_KR_PG_KATRI/lane_boundary_set.json"
crosswalk_file = "/home/v/vip_ws/src/hmg_2025/src/2025_HMG_AD/hd_sample/Map_Data/R_KR_PG_KATRI/singlecrosswalk_set.json"
traffic_light_file = "/home/v/vip_ws/src/hmg_2025/src/2025_HMG_AD/hd_sample/Map_Data/R_KR_PG_KATRI/traffic_light_set.json"
global_path_file = "/home/v/vip_ws/src/hmg_2025/src/2025_HMG_AD/hd_sample/dataset/R_KR_PG_KATRI__HMG_Scenario_0/global_path.csv"
calibration_folder = "/home/v/vip_ws/src/hmg_2025/src/2025_HMG_AD/hd_sample/Calibration"
model_path = "/home/v/vip_ws/src/hmg_2025/src/2025_HMG_AD/model_epoch_1.pth"

with open(road_mesh_file, "r") as f:
    road_mesh_data = json.load(f)
with open(lane_boundary_file, "r") as f:
    lane_data = json.load(f)
with open(crosswalk_file, "r") as f:
    crosswalk_data = json.load(f)
with open(traffic_light_file, "r") as f:
    traffic_light_data = json.load(f)

# csv
global_path_data = pd.read_csv(global_path_file)
global_path_points = global_path_data[["PositionX (m)", "PositionY (m)"]].values

# txt
# global_path_points = []
# with open(global_path_file, "r") as file:
#     for line in file:
#         # Split each line by space
#         parts = line.strip().split()
#         if len(parts) >= 4:  # Ensure there are enough columns (ID, X, Y, Yaw)
#             # Extract PositionX and PositionY (columns 2 and 3)
#             position_x = float(parts[1])
#             position_y = float(parts[2])
#             global_path_points.append([position_x, position_y])

# Convert to a NumPy array for further processing
global_path_points = np.array(global_path_points)

# 차선 JSON 분류
lane_boundary_points_by_type = {}
for entry in lane_data:
    lane_type = str(entry.get("lane_type", "unknown"))
    if lane_type not in lane_boundary_points_by_type:
        lane_boundary_points_by_type[lane_type] = []
    lane_boundary_points_by_type[lane_type].append(entry["points"])

# 횡단보도
crosswalk_points = [entry["points"] for entry in crosswalk_data]

# 신호등
traffic_light_points_by_subtype = {}
for entry in traffic_light_data:
    st = str(entry.get("sub_type", "unknown"))
    if st not in traffic_light_points_by_subtype:
        traffic_light_points_by_subtype[st] = []
    traffic_light_points_by_subtype[st].append(entry["point"][:2])

# ==============================
#         유틸 함수
# ==============================
def calculate_distance(point, ego_pos):
    return np.sqrt((point[0] - ego_pos[0])**2 + (point[1] - ego_pos[1])**2)

def filter_points_within_range(points, ego_pos, max_dist):
    """
    polygon 형태의 points(2D 좌표 리스트) 중 ego_pos와의 거리가 max_dist 이내인 것만 필터링
    """
    filtered = []
    for polygon in points:
        temp = [p for p in polygon if calculate_distance(p, ego_pos) <= max_dist]
        if len(temp) > 0:
            filtered.append(temp)
    return filtered

def rotate_point(pt, theta):
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    x, y = pt
    return (x*cos_t - y*sin_t, x*sin_t + y*cos_t)

def global_to_local_north_up(pt_global, ego_pos, ego_yaw_deg):
    """
    ENU 기반 글로벌 -> 차량 기준 Local North-Up 좌표 변환
    차량 위치를 (0,0), 차량 진행 방향을 상(y축 양의 방향)으로 두기
    """
    ego_yaw_rad = np.deg2rad(ego_yaw_deg)
    # 차량 yaw가 90도일 때가 전진 방향(북쪽) → theta = ego_yaw_rad - pi/2
    theta = ego_yaw_rad - np.pi/2
    shift = (pt_global[0] - ego_pos[0], pt_global[1] - ego_pos[1])
    return rotate_point(shift, -theta)

def classify_and_extract_points(data):
    """
    road_mesh_data(또는 유사 구조)에 대해,
    'class' 필드별로 폴리곤 points를 모아 반환
    """
    result = {}
    for elem in data:
        cls = elem.get("class", "unknown")
        if cls not in result:
            result[cls] = []
        if "points" in elem:
            result[cls].append(elem["points"])
    return result

def polygon_global_to_local_north_up(polygons, ego_pos, ego_yaw_deg):
    local_polygons = []
    for polygon in polygons:
        local_poly = [global_to_local_north_up(p, ego_pos, ego_yaw_deg) for p in polygon]
        local_polygons.append(local_poly)
    return local_polygons

def plot_polygon_local(polygons, color="blue"):
    """
    matplotlib을 사용해 로컬 좌표계에서 폴리곤(또는 라인)을 표시
    """
    for poly in polygons:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        plt.plot(xs, ys, color=color)

def compute_relative_path(global_path, ego_position, n_points=5):
    """
    Ego 위치에서 가장 가까운 지점부터 n_points개를 가져와
    ego_position 기준으로 상대좌표를 구함
    """
    dist = np.linalg.norm(global_path - ego_position[:2], axis=1)
    idx_min = np.argmin(dist)
    pts = global_path[idx_min: idx_min + n_points]
    if len(pts) < n_points:
        pad = np.zeros((n_points - len(pts), 2))
        pts = np.vstack([pts, pad])
    return pts - ego_position[:2]

def _get_next_path_points(global_path, ego_pos, n_points=5):
    """
    global_path에서 ego_pos와 가장 가까운 지점부터 n_points개 추출
    """
    dist = np.linalg.norm(global_path - np.array(ego_pos), axis=1)
    idx_min = np.argmin(dist)
    pts = global_path[idx_min: idx_min + n_points]
    if len(pts) < n_points:
        pad = np.zeros((n_points - len(pts), global_path.shape[1]))
        pts = np.vstack([pts, pad])
    return pts

def _compute_relative_path(next_points, ego_pos):
    return (next_points - np.array(ego_pos)).flatten()

def assign_colors(categories):
    cmap = cm.get_cmap("tab20", len(categories))
    return {cat: cmap(i) for i, cat in enumerate(categories)}

classified_exterior_points = classify_and_extract_points(road_mesh_data)
classified_interior_points = classify_and_extract_points(road_mesh_data)

# 사전 색상 할당(맵 데이터용)
lane_boundary_colors = assign_colors(lane_boundary_points_by_type.keys())
traffic_light_colors = assign_colors(traffic_light_points_by_subtype.keys())
exterior_colors = assign_colors(classified_exterior_points.keys())
interior_colors = assign_colors(classified_interior_points.keys())

# ==============================
#    HD Map to Tensor
# ==============================
def _process_hd_map(hd_map_frame):
    """
    150x150 RGB 이미지를 여러 레이어로 분리해 (6,150,150) 텐서로 만든다.
    - 빨강(>200) : Exterior
    - 초록(>200) : Interior
    - 파랑(>200) : 차선
    - 노랑(빨강+초록) : 횡단보도
    - 청록(초록+파랑) : 신호등
    - 자주(빨강+파랑) : Ego
    """
    exterior      = (hd_map_frame[:, :, 0] > 200).astype(np.float32)
    interior      = (hd_map_frame[:, :, 1] > 200).astype(np.float32)
    lane          = (hd_map_frame[:, :, 2] > 200).astype(np.float32)
    crosswalk     = ((hd_map_frame[:, :, 0] > 200) & (hd_map_frame[:, :, 1] > 200)).astype(np.float32)
    traffic_light = ((hd_map_frame[:, :, 1] > 200) & (hd_map_frame[:, :, 2] > 200)).astype(np.float32)
    ego_vehicle   = ((hd_map_frame[:, :, 0] > 200) & (hd_map_frame[:, :, 2] > 200)).astype(np.float32)

    return np.stack([exterior, interior, lane, crosswalk, traffic_light, ego_vehicle], axis=0)

def save_hd_map_as_tensor(ego_pos, ego_yaw_deg):
    """
    (Ego_pos, Ego_yaw) 기준으로 HD Map 시각화를 한 뒤, 150x150 이미지를 (6,150,150) 텐서로 변환
    """
    plt.figure(figsize=(10,10))

    # global path 로드 - csv
    global_path_data = pd.read_csv(global_path_file)
    global_path_points = global_path_data[['PositionX (m)', 'PositionY (m)']].values
    
    # global path load - txt
    # global global_path_points
    relative_path = compute_relative_path(global_path_points, ego_pos, n_points=5)

    # Road mesh 분류
    exterior_pts = classify_and_extract_points(road_mesh_data)
    interior_pts = classify_and_extract_points(road_mesh_data)

    # 시야 범위 30m 내 필터
    max_dist = 30
    filtered_exterior = {
        c: filter_points_within_range(ps, ego_pos[:2], max_dist)
        for c, ps in exterior_pts.items()
    }
    filtered_interior = {
        c: filter_points_within_range(ps, ego_pos[:2], max_dist)
        for c, ps in interior_pts.items()
    }
    filtered_lane = {
        lt: filter_points_within_range(ps, ego_pos[:2], max_dist)
        for lt, ps in lane_boundary_points_by_type.items()
    }
    filtered_crosswalk = filter_points_within_range(crosswalk_points, ego_pos[:2], max_dist)
    filtered_traffic_light = {
        st: [p for p in pts if calculate_distance(p, ego_pos[:2]) <= max_dist]
        for st, pts in traffic_light_points_by_subtype.items()
    }

    # (7-1) Exterior (Global -> Local 변환 후 플롯)
    for cls, polygons in filtered_exterior.items():
        local_polygons = polygon_global_to_local_north_up(polygons, ego_pos[:2], ego_yaw_deg)
        plot_polygon_local(local_polygons, color=exterior_colors[cls])
        
    # (7-2) Interior
    for cls, polygons in filtered_interior.items():
        local_polygons = polygon_global_to_local_north_up(polygons, ego_pos[:2], ego_yaw_deg)
        plot_polygon_local(local_polygons, color=interior_colors[cls])
        
    # (7-3) Lane Boundaries
    for lt, polygons in filtered_lane.items():
        local_polygons = polygon_global_to_local_north_up(polygons, ego_pos[:2], ego_yaw_deg)
        plot_polygon_local(local_polygons, color=lane_boundary_colors[lt])
        
    # (7-4) Traffic Lights (점)
    for st, pts in filtered_traffic_light.items():
        for pt in pts:
            local_pt = global_to_local_north_up(pt, ego_pos[:2], ego_yaw_deg)
            plt.scatter(local_pt[0], local_pt[1], color=traffic_light_colors[st], s=80, zorder=5)
        
    # (7-5) Crosswalk
    local_crosswalks = polygon_global_to_local_north_up(filtered_crosswalk, ego_pos[:2], ego_yaw_deg)
    plot_polygon_local(local_crosswalks, color='orange')
        
    # (7-6) Global Path 일부 점 (Ego 기준 표시)
    #       relative_path 자체는 단순히 (global_path - ego_pos)지만,
    #       실제 회전을 적용해야 정확히 '차량이 북쪽을 보게' 표현됨
    for r_pt in relative_path:
        global_pt = (r_pt[0] + ego_pos[0], r_pt[1] + ego_pos[1])
        local_pt = global_to_local_north_up(global_pt, ego_pos[:2], ego_yaw_deg)
        plt.scatter(local_pt[0], local_pt[1], color='cyan', s=50, zorder=5)
        
    # (7-7) Ego Vehicle 위치 (로컬 좌표계에선 항상 (0, 0))
    plt.scatter(0, 0, color='green', s=100, zorder=5)

    plt.xlim(-30,30)
    plt.ylim(-30,30)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")

    # save_path = "/home/v/vip_ws/visualizations"
    # plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0)

    # 이미지 메모리로 저장
    import sys
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close()

    # PIL로 열어 150x150으로 리사이즈 → numpy
    img_pil = Image.open(buf).resize((150,150))
    hd_map_frame = np.array(img_pil)[..., :3]  # RGB만 추출 (150,150,3)
    buf.close()

    # (6,150,150) 채널 텐서로 변환
    return _process_hd_map(hd_map_frame)

# =============== T=2 시퀀스 ===============
T = 3
sequence_queue = deque(maxlen=T)  # 각 시점(frame_data)를 보관

def load_calibration_data(calibration_folder, T=2):
    """
    캘리브레이션 데이터를 불러와 모델 입력 형식으로 변환합니다.

    Args:
        calibration_folder (str): 캘리브레이션 데이터가 저장된 폴더 경로.
        T (int): 시계열 길이. 기본값은 2.

    Returns:
        dict: model_input 형태의 딕셔너리.
            - "intrinsic": Shape (1, T, 5, 3, 3)
            - "extrinsic": Shape (1, T, 5, 4, 4)
    """
    # 카메라 번호 및 파일 키워드
    camera_count = 5
    intrinsic_keyword = "intrinsic"
    extrinsic_keyword = "extrinsic"

    # 시계열 데이터 큐 초기화
    intrinsic_queue = []
    extrinsic_queue = []

    # 시계열 반복
    for t in range(T):
        intrinsic_per_timestep = []
        extrinsic_per_timestep = []

        # 각 카메라별 Intrinsic 및 Extrinsic 파일 처리
        for i in range(1, camera_count + 1):  # CAMERA_1 ~ CAMERA_5
            # 파일 경로 생성
            intrinsic_file = os.path.join(calibration_folder, f"CAMERA_{i}__{intrinsic_keyword}.npy")
            extrinsic_file = os.path.join(calibration_folder, f"CAMERA_{i}__{extrinsic_keyword}.npy")

            # 파일 로드
            if not os.path.exists(intrinsic_file) or not os.path.exists(extrinsic_file):
                raise FileNotFoundError(f"Missing calibration files for CAMERA_{i}: {intrinsic_file}, {extrinsic_file}")

            intrinsic_np = np.load(intrinsic_file)  # Intrinsic: (3, 3)
            extrinsic_np = np.load(extrinsic_file)  # Extrinsic: (4, 4)

            # Tensor 변환
            intrinsic_per_timestep.append(torch.tensor(intrinsic_np, dtype=torch.float32))
            extrinsic_per_timestep.append(torch.tensor(extrinsic_np, dtype=torch.float32))

        # 현재 시점 데이터를 큐에 추가
        intrinsic_queue.append(torch.stack(intrinsic_per_timestep, dim=0))  # Shape: (5, 3, 3)
        extrinsic_queue.append(torch.stack(extrinsic_per_timestep, dim=0))  # Shape: (5, 4, 4)

    # 시계열 데이터를 통합하여 최종 텐서 생성
    intrinsic_tensor = torch.stack(intrinsic_queue, dim=0)  # Shape: (T, 5, 3, 3)
    extrinsic_tensor = torch.stack(extrinsic_queue, dim=0)  # Shape: (T, 5, 4, 4)

    # Batch 차원 추가 (B=1)
    intrinsic_tensor = intrinsic_tensor.unsqueeze(0)  # Shape: (1, T, 5, 3, 3)
    extrinsic_tensor = extrinsic_tensor.unsqueeze(0)  # Shape: (1, T, 5, 4, 4)

    # Model Input 형태로 저장
    model_input = {
        "intrinsic": intrinsic_tensor,  # (1, T, 5, 3, 3)
        "extrinsic": extrinsic_tensor   # (1, T, 5, 4, 4)
    }

    return model_input

def build_model_input_from_sequence(seq):
    """
    seq: 길이가 T=2인 리스트
         seq[0] -> (ego, cam, hd_map) at t_{k-1}
         seq[1] -> (ego, cam, hd_map) at t_{k}
    원하는 형식의 numpy array로 묶어서 반환
    """
    # 예: ego_info -> shape=(T, 28)
    ego_list = []
    cam_list = []
    hd_map_list = []
    for frame in seq:
        ego_list.append(frame["ego_info"])   # (28,) 형태
        cam_list.append(frame["camera_data"])# (5,3,224,480)
        hd_map_list.append(frame["hd_map"])  # (6,150,150)

    # numpy로 묶기
    ego_np    = np.stack(ego_list, axis=0)    # (T=2, 28)
    cam_np    = np.stack(cam_list, axis=0)    # (T=2, 5, 3, 224, 480)
    hd_map_np = np.stack(hd_map_list, axis=0) # (T=2, 6, 150, 150)

    # 배치 차원 B=1 추가 → (1, T, ...)
    ego_np    = np.expand_dims(ego_np, axis=0)
    cam_np    = np.expand_dims(cam_np, axis=0)
    hd_map_np = np.expand_dims(hd_map_np, axis=0)

    model_input = {
        "ego_info": ego_np,    # (1, 2, 28)
        "camera":   cam_np,    # (1, 2, 5, 3, 224, 480)
        "hd_map":   hd_map_np, # (1, 2, 6, 150, 150)
    }
    return model_input


# =============== 전역 변수: 카메라 최신 이미지 ===============
latest_camera_data = {
    "front": None,
    "left":  None,
    "right": None,
    "rearL": None,
    "rearR": None
}

# =============== 카메라 콜백 (가장 최신 이미지 저장) ===============
def _decode_compressed_image(msg_compressed):
    np_arr = np.frombuffer(msg_compressed.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR
    # (Width=480, Height=224) → cv2.resize의 인자는 (width, height)
    img = cv2.resize(img, (480, 224))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)  # (3,224,480)
    return img

def front_callback(msg):
    global latest_camera_data
    latest_camera_data["front"] = _decode_compressed_image(msg)

def left_callback(msg):
    global latest_camera_data
    latest_camera_data["left"] = _decode_compressed_image(msg)

def rearL_callback(msg):
    global latest_camera_data
    latest_camera_data["rearL"] = _decode_compressed_image(msg)

def rearR_callback(msg):
    global latest_camera_data
    latest_camera_data["rearR"] = _decode_compressed_image(msg)

def right_callback(msg):
    global latest_camera_data
    latest_camera_data["right"] = _decode_compressed_image(msg)

def publish_control_output(output):
    """
    Args:
        output: 추론 결과 텐서, shape=(B, T, 3)로 accel, brake, steer 포함
    """
    # 출력 텐서를 CPU로 이동하고 numpy 배열로 변환
    output_np = output.cpu().numpy()

    # 가장 최근 시점의 accel, brake, steer 추출
    # 예: output_np[0, -1, :]에서 B=0(첫 번째 배치), T=-1(가장 최신 시점)
    accel, brake, steer = output_np[0, -1, :]

    # ROS 메시지로 퍼블리시
    ctrl_msg = CtrlCmd()
    ctrl_msg.longlCmdType = 1
    ctrl_msg.accel = float(accel)
    ctrl_msg.brake = float(brake)
    ctrl_msg.steering = float(steer)

    cmd_pub.publish(ctrl_msg)

    rospy.loginfo(f"Published: accel={accel}, brake={brake}, steer={steer}")

# =============== Ego 콜백 (가장 최신 카메라 + HD Map) ===============
def ego_callback(ego_msg):
    """
    빠른 주기로 들어오는 Ego 메시지에서,
    5개 카메라의 '가장 최신' 이미지 + HD Map을 이용하여
    T=2 시퀀스 생성
    """

    # (1) 5개 카메라가 모두 준비되어 있는지 검사
    if any(latest_camera_data[cam] is None for cam in latest_camera_data):
        rospy.logwarn_throttle(3.0, "Not all camera images are available yet.")
        return

    # (2) Ego 상태 추출
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
    ego_enu_velocity = [
        ego_msg.noisy_enu_velocity.east,
        ego_msg.noisy_enu_velocity.north,
        ego_msg.noisy_enu_velocity.up
    ]
    ego_velocity = [
        ego_msg.noisy_velocity.x,
        ego_msg.noisy_velocity.y,
        ego_msg.noisy_velocity.z
    ]
    ego_ang_vel = [
        ego_msg.noisy_angularVelocity.roll,
        ego_msg.noisy_angularVelocity.pitch,
        ego_msg.noisy_angularVelocity.yaw
    ]
    ego_accel = [
        ego_msg.noisy_acceleration.x,
        ego_msg.noisy_acceleration.y,
        ego_msg.noisy_acceleration.z
    ]
    ego_control = [
        ego_msg.accel,
        ego_msg.brake,
        ego_msg.steer
    ]

    next_pts = _get_next_path_points(global_path_points, ego_position[:2], n_points=5)
    print(next_pts)
    print(ego_position)
    rel_path = _compute_relative_path(next_pts, ego_position[:2])  # shape=(10,)

    # 최종 ego 벡터
    ego_info = (
        ego_position + ego_orientation + ego_enu_velocity
        + ego_velocity + ego_ang_vel + ego_accel
        + ego_control
    )  # 길이=21

    # (3) HD Map 생성
    ego_yaw_deg = ego_orientation[2]
    hd_map_tensor = save_hd_map_as_tensor(ego_position, ego_yaw_deg)

    # (4) 카메라 5개 묶어서 shape=(5,3,224,480)
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

    # (6) T=2 시퀀스에 추가
    sequence_queue.append(frame_data)

    # (7) 시퀀스가 2개 모이면 모델 입력 생성
    if len(sequence_queue) == T:
        seq_list = list(sequence_queue)  # 길이 2
        model_input_sequence = build_model_input_from_sequence(seq_list)
        model_input_calibration = load_calibration_data(calibration_folder, T=T)

        # 두 개의 데이터를 하나로 결합
        model_input = {
            **model_input_sequence,               # 시퀀스 데이터 추가
            **model_input_calibration             # 캘리브레이션 데이터 추가
        }
        # print("Combined Model Input Shapes:")
        # print(f"Intrinsic Tensor Shape: {model_input['intrinsic'].shape}")  # (1, 2, 5, 3, 3)
        # print(f"Extrinsic Tensor Shape: {model_input['extrinsic'].shape}")  # (1, 2, 5, 4, 4)
        # print(f"Ego Info Shape: {model_input['ego_info'].shape}")           # (1, 2, 21) (예시)
        # print(f"Camera Shape: {model_input['camera'].shape}")               # (1, 2, 5, 3, 224, 480) (예시)
        # print(f"HD Map Shape: {model_input['hd_map'].shape}")    
        
        # 모델 초기화
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CombinedModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 평가 모드로 설정

        # 모델 입력 데이터를 GPU로 이동하기 전에 numpy.ndarray를 torch.Tensor로 변환
        camera_images = torch.tensor(model_input["camera"], dtype=torch.float32).to(device)
        intrinsics = model_input["intrinsic"].to(device)
        extrinsics = model_input["extrinsic"].to(device)
        hd_map_tensors = torch.tensor(model_input["hd_map"], dtype=torch.float32).to(device)
        ego_inputs = torch.tensor(model_input["ego_info"], dtype=torch.float32).to(device)


        # 추론 실행
        with torch.no_grad():
            output = model(
                camera_images=camera_images,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                hd_map_tensors=hd_map_tensors,
                ego_inputs=ego_inputs
            )
            publish_control_output(output)

        sequence_queue.clear()


# =============== 메인 함수 ===============
cmd_pub = None
def main():
    global cmd_pub
    rospy.init_node("faster_ego_method1_example", anonymous=True)

    # ROS 퍼블리셔 초기화
    cmd_pub = rospy.Publisher('/ctrl_cmd', CtrlCmd, queue_size=10)

    # (1) 카메라 5개 구독
    rospy.Subscriber("/image_jpeg1/compressed", CompressedImage, front_callback)
    rospy.Subscriber("/image_jpeg2/compressed",  CompressedImage, left_callback)
    rospy.Subscriber("/image_jpeg3/compressed", CompressedImage, right_callback)
    rospy.Subscriber("/image_jpeg4/compressed", CompressedImage, rearL_callback)
    rospy.Subscriber("/image_jpeg5/compressed", CompressedImage, rearR_callback)

    # (2) Ego 토픽 구독 (더 빠른 주기로 들어온다고 가정)
    rospy.Subscriber("/Ego_noisy_topic", EgoNoisyStatus, ego_callback)
    # rospy.Subscriber("/Ego_topic", EgoVehicleStatus, ego_callback)

    rospy.loginfo("Initialized. Using 'latest camera' method. Waiting for messages...")
 
    

    rospy.spin()

if __name__ == "__main__":
    main()
