import cv2
import numpy as np
import math
import pandas as pd

def process_hd_map(hd_map_frame):
    """
    HD Map 프레임을 다중 채널 텐서로 전처리합니다.
    채널 순서: ego, global path, drivable area, white lane, yellow lane, crosswalk, traffic light
    """
    # 색상 정의 (BGR 순서)
    ego_color = np.array([0, 0, 255])
    global_path_color = np.array([255, 0, 0])
    drivable_area_color = np.array([200, 200, 200])
    white_lane_color = np.array([255, 255, 255])
    yellow_lane_color = np.array([0, 255, 255])
    crosswalk_color = np.array([0, 255, 0])
    traffic_light_color = np.array([0, 0, 255])
    
    # 각 채널의 픽셀값이 해당 색상과 정확히 일치하는지 확인
    ego_mask = np.all(hd_map_frame == ego_color, axis=-1).astype(np.float32)
    global_path_mask = np.all(hd_map_frame == global_path_color, axis=-1).astype(np.float32)
    drivable_area_mask = np.all(hd_map_frame == drivable_area_color, axis=-1).astype(np.float32)
    white_lane_mask = np.all(hd_map_frame == white_lane_color, axis=-1).astype(np.float32)
    yellow_lane_mask = np.all(hd_map_frame == yellow_lane_color, axis=-1).astype(np.float32)
    crosswalk_mask = np.all(hd_map_frame == crosswalk_color, axis=-1).astype(np.float32)
    traffic_light_mask = np.all(hd_map_frame == traffic_light_color, axis=-1).astype(np.float32)
    
    # 7채널 텐서로 스택
    multi_channel_tensor = np.stack([ego_mask, global_path_mask, drivable_area_mask,
                                     white_lane_mask, yellow_lane_mask, crosswalk_mask,
                                     traffic_light_mask], axis=0)
    return multi_channel_tensor

def generate_hd_map(global_path_csv, ego, 
                    hdmap_filename="/home/v/vip_ws/src/hmg_2025/src/2025_HMG_AD/v2/R_KR_PG_KATRI_DP_Full_HD.png",
                    scale_factor=10, margin=10, min_x=-162.019, min_y=953.702,
                    radius_m=25, arrow_length_world=2.0):
    """
    글로벌 경로와 실시간 ego 데이터를 받아 HD‑MAP 이미지를 생성한 후, 
    다중 채널 텐서로 전처리합니다.
    
    Parameters:
        global_path_csv (str): 글로벌 경로 CSV 파일 경로.
        ego (dict): 실시간 ego 정보. 반드시 다음 키를 포함해야 합니다.
                    - "position": [x, y] (월드 좌표)
                    - "heading": heading (도 단위, 예: 0~360)
        hdmap_filename (str): HD‑MAP 이미지 파일 경로.
        scale_factor (int): 1m 당 픽셀 수.
        margin (int): 좌표 변환 시 적용할 margin (픽셀).
        min_x (float): HD‑MAP 생성 시 사용한 최소 world x 좌표.
        min_y (float): HD‑MAP 생성 시 사용한 최소 world y 좌표.
        radius_m (float): ego를 중심으로 crop할 실제 반경 (m). (전체 영역: 2*radius_m)
        arrow_length_world (float): ego heading을 나타내는 화살표의 길이 (m).
        
    Returns:
        np.ndarray: 전처리된 7채널 텐서 (채널 순서:
                    ego, global path, drivable area, white lane, yellow lane, crosswalk, traffic light)
    """
    # 1. HD‑MAP 원본 이미지 로드
    hdmap_orig = cv2.imread(hdmap_filename)
    if hdmap_orig is None:
        raise IOError(f"HD‑MAP 이미지 '{hdmap_filename}'를 불러올 수 없습니다.")
    img_height = hdmap_orig.shape[0]

    # 2. world 좌표 → 이미지 좌표 변환 함수
    def world_to_img(pt):
        x, y = pt[0], pt[1]
        img_x = int(round((x - min_x + margin) * scale_factor))
        # 이미지 좌표계에서 y는 반전됩니다.
        img_y = img_height - int(round((y - min_y + margin) * scale_factor))
        return [img_x, img_y]

    # 3. 글로벌 경로 오버레이
    hdmap_with_path = hdmap_orig.copy()
    try:
        global_path_df = pd.read_csv(global_path_csv)
    except Exception as e:
        raise IOError(f"글로벌 경로 CSV 파일 '{global_path_csv}'를 읽어올 수 없습니다: {e}")

    global_path_points = []
    for idx, row in global_path_df.iterrows():
        try:
            x = float(row["PositionX (m)"])
            y = float(row["PositionY (m)"])
        except KeyError:
            raise KeyError("CSV 파일에 'PositionX (m)'와 'PositionY (m)' 열이 존재하지 않습니다.")
        pt = [x, y]
        pt_img = world_to_img(pt)
        global_path_points.append(pt_img)
    
    if global_path_points:
        pts = np.array(global_path_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(hdmap_with_path, [pts], isClosed=False, color=(255, 0, 0), thickness=2)

    # 4. 실시간 ego 데이터 오버레이
    ego_position = ego.get("position")
    ego_heading = ego.get("heading")
    if ego_position is None or ego_heading is None:
        raise ValueError("Ego 데이터는 'position'과 'heading' 키를 포함해야 합니다.")
    
    # ego의 월드 좌표를 이미지 좌표로 변환
    ego_img_pos = world_to_img(ego_position)
    
    # ego 표시: 빨간 원 (BGR: (0, 0, 255))
    cv2.circle(hdmap_with_path, tuple(ego_img_pos), 5, (0, 0, 255), -1)
    
    # ego heading을 나타내는 화살표
    heading_rad = math.radians(ego_heading)
    dx = arrow_length_world * math.cos(heading_rad)
    dy = arrow_length_world * math.sin(heading_rad)
    arrow_endpoint_world = [ego_position[0] + dx, ego_position[1] + dy]
    arrow_img_endpoint = world_to_img(arrow_endpoint_world)
    cv2.arrowedLine(hdmap_with_path, tuple(ego_img_pos), tuple(arrow_img_endpoint), (0, 0, 255), 2)

    # 5. BEV Crop: ego를 중심으로 실제 좌표 radius_m (즉, 2*radius_m x 2*radius_m 영역) crop
    crop_size_pixels = int(2 * radius_m * scale_factor)
    half_crop = crop_size_pixels // 2
    center_x, center_y = ego_img_pos
    x1 = max(center_x - half_crop, 0)
    y1 = max(center_y - half_crop, 0)
    x2 = min(center_x + half_crop, hdmap_orig.shape[1])
    y2 = min(center_y + half_crop, hdmap_orig.shape[0])
    cropped_img = hdmap_with_path[y1:y2, x1:x2]

    # 6. Crop된 이미지 회전: ego의 heading이 상단(위쪽)을 향하도록.
    #    회전 각도: ego_heading이 90°가 되어야 하므로 rotation_angle = 90 - ego_heading
    rotation_angle = 90 - ego_heading
    crop_center = (cropped_img.shape[1] // 2, cropped_img.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(crop_center, rotation_angle, 1.0)
    rotated_img = cv2.warpAffine(cropped_img, rot_mat, (cropped_img.shape[1], cropped_img.shape[0]))
    
    # 최종적으로 원하는 크기로 리사이즈 (예: 200x200)
    rotated_img = cv2.resize(rotated_img, (200, 200))
    
    # 7. 전처리: HD‑MAP 프레임을 다중 채널 텐서로 변환
    hd_map_tensor = process_hd_map(rotated_img)
    return hd_map_tensor

# 사용 예시
if __name__ == "__main__":
    # 예시 global path CSV 파일 경로 및 ego 데이터
    global_path_csv = "global_path.csv"  # 실제 파일 경로로 변경
    ego_data = {
        "position": [0, 0],   # 예시 좌표 (월드 좌표)
        "heading": 45         # 예시 heading (도 단위)
    }
    
    hd_map_tensor = generate_hd_map(global_path_csv, ego_data)
    print("HD Map Tensor shape:", hd_map_tensor.shape)
