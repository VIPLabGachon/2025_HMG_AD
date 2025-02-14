import cv2
import numpy as np
import glob
import os
import math
import pandas as pd

# ----------------------------
# 1. 미리 생성된 HD‑MAP 이미지 로드
# ----------------------------
hdmap_filename = "/home/vip/2025_HMG_AD/v2/R_KR_PR_Sangam_DP_Full_HD.png"
hdmap_orig = cv2.imread(hdmap_filename)
if hdmap_orig is None:
    raise IOError(f"HD‑MAP 이미지 '{hdmap_filename}'를 불러올 수 없습니다.")

# ----------------------------
# 2. HD‑MAP 생성 시 사용한 파라미터 (모든 시나리오에서 동일해야 합니다)
# ----------------------------
scale_factor = 10   # 예: 1m 당 10픽셀
margin = 10         # 원래 좌표 기준 margin (픽셀)
# HD‑MAP 생성 시 전체 좌표(all_points)로부터 계산한 최소 world 좌표
min_x = -597.991
min_y = -1887.1
# HD‑MAP 이미지의 높이 (y 좌표 변환 시 사용)
img_height = hdmap_orig.shape[0]

def world_to_img(pt):
    """
    월드 좌표(pt: [x, y, ...])를 HD‑MAP 생성 시 사용한 파라미터를 바탕으로 
    이미지 좌표로 변환합니다.
    """
    x, y = pt[0], pt[1]
    img_x = int(round((x - min_x + margin) * scale_factor))
    img_y = img_height - int(round((y - min_y + margin) * scale_factor))
    return [img_x, img_y]

def draw_object(img, center_world, yaw, length, width, color=(128, 0, 128)):
    """
    객체의 world 좌표상의 중심(center_world), yaw, length, width 정보를 이용해 
    회전 사각형을 계산하고, 해당 사각형 내부를 보라색(color)으로 채워 오버레이합니다.
    """
    half_length = length / 2.0
    half_width = width / 2.0
    angle_rad = math.radians(yaw)
    # 로컬 좌표계의 네 모서리 (l/2, w/2)
    local_corners = [( half_length,  half_width),
                     ( half_length, -half_width),
                     (-half_length, -half_width),
                     (-half_length,  half_width)]
    corners_world = []
    for dx, dy in local_corners:
        wx = center_world[0] + dx * math.cos(angle_rad) - dy * math.sin(angle_rad)
        wy = center_world[1] + dx * math.sin(angle_rad) + dy * math.cos(angle_rad)
        corners_world.append([wx, wy])
    corners_img = [world_to_img(pt) for pt in corners_world]
    corners_img = np.array(corners_img, np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(img, [corners_img], color=color)

# ----------------------------
# 3. 시나리오 디렉토리 순회 (전체 시나리오 처리)
# ----------------------------
base_scenario_pattern = "/home/vip/hd/Dataset/R_KR_PR_Sangam_DP__HMG_Scenario_*"
scenario_dirs = sorted(glob.glob(base_scenario_pattern))

# BEV crop: 실제 좌표 50m 반경 → 100m×100m 영역 (픽셀 단위)
radius_m = 25  
crop_size_pixels = int(2 * radius_m * scale_factor)

for scenario_dir in scenario_dirs:
    print(f"Processing scenario: {scenario_dir}")
    
    # ----------------------------
    # 3-1. Global Path CSV 로드 및 오버레이
    # ----------------------------
    global_path_csv = os.path.join(scenario_dir, "global_path.csv")
    if not os.path.exists(global_path_csv):
        print(f"Global path CSV not found in {scenario_dir}, skipping scenario.")
        continue
    global_path_df = pd.read_csv(global_path_csv)
    
    hdmap_with_path = hdmap_orig.copy()
    global_path_points = []
    for idx, row in global_path_df.iterrows():
        pt = [float(row["PositionX (m)"]), float(row["PositionY (m)"])]
        pt_img = world_to_img(pt)
        global_path_points.append(pt_img)
    if global_path_points:
        global_path_points = np.array(global_path_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(hdmap_with_path, [global_path_points], isClosed=False, color=(255, 0, 0), thickness=2)
    
    # ----------------------------
    # 3-2. Ego 데이터 오버레이 및 해당 index의 Object Info 오버레이
    # ----------------------------
    ego_dir = os.path.join(scenario_dir, "EGO_INFO")
    if not os.path.exists(ego_dir):
        print(f"EGO_INFO folder not found in {scenario_dir}, skipping Ego overlay.")
        continue
    ego_files = glob.glob(os.path.join(ego_dir, "*.txt"))
    ego_files = sorted(ego_files, key=lambda f: float(os.path.splitext(os.path.basename(f))[0]))
    selected_ego_files = [f for f in ego_files if float(os.path.splitext(os.path.basename(f))[0]) % 10 == 0]
    
    # 결과 저장 폴더 (예: 시나리오 내 HD_MAP 폴더)
    output_folder = os.path.join(scenario_dir, "GT_HD")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for ego_file in selected_ego_files:
        with open(ego_file, 'r') as f:
            lines = f.readlines()
        ego_position = None
        ego_heading = None
        for line in lines:
            line = line.strip()
            if line.startswith("position:"):
                parts = line[len("position:"):].strip().split()
                ego_position = [float(parts[0]), float(parts[1])]
            elif line.startswith("orientation:"):
                parts = line[len("orientation:"):].strip().split()
                if len(parts) >= 3:
                    ego_heading = float(parts[2])
        if ego_position is None or ego_heading is None:
            continue
        
        # Ego의 월드 좌표를 이미지 좌표로 변환
        ego_img_pos = world_to_img(ego_position)
        
        # HD‑MAP에 Global Path가 오버레이된 상태에서 Ego 오버레이용 이미지 복사본 생성
        ego_overlay = hdmap_with_path.copy()
        
        # Ego 표시: 빨간 원 및 화살표
        cv2.circle(ego_overlay, tuple(ego_img_pos), 5, (0, 0, 255), -1)
        arrow_length_world = 2.0  # 예: 2m 길이
        heading_rad = math.radians(ego_heading)
        dx = arrow_length_world * math.cos(heading_rad)
        dy = arrow_length_world * math.sin(heading_rad)
        arrow_endpoint_world = [ego_position[0] + dx, ego_position[1] + dy]
        arrow_img_endpoint = world_to_img(arrow_endpoint_world)
        cv2.arrowedLine(ego_overlay, tuple(ego_img_pos), tuple(arrow_img_endpoint), (0, 0, 255), 2)
        
        # ----------------------------
        # 3-2-1. 해당 index의 Object Info 오버레이 (OBJECT_INFO 폴더)
        # ----------------------------
        # Ego 파일명이 예를 들어 "10.txt"라면 object_info_10.txt만 오버레이
        label = os.path.splitext(os.path.basename(ego_file))[0]
        object_info_file = os.path.join(scenario_dir, "OBJECT_INFO", f"object_info_{label}.txt")
        if os.path.exists(object_info_file):
            with open(object_info_file, 'r') as f:
                obj_lines = f.readlines()
            for line in obj_lines:
                parts = line.strip().split()
                if len(parts) < 14:
                    continue
                # 형식: class, E, N, U, roll, pitch, yaw, length, width, height, vx, vy, vz, track id
                E = float(parts[1])
                N = float(parts[2])
                yaw_obj = float(parts[6])
                length_obj = float(parts[7])
                width_obj = float(parts[8])
                center_world = [E, N]
                draw_object(ego_overlay, center_world, yaw_obj, length_obj, width_obj,
                            color=(128, 0, 128))
        else:
            print(f"Object info file for index {label} not found in {scenario_dir}/OBJECT_INFO.")
        
        # ----------------------------
        # 3-3. BEV Crop: Ego를 중심으로 실제 좌표 50m 반경 (100m×100m 영역) crop
        # ----------------------------
        half_crop = crop_size_pixels // 2
        center_x, center_y = ego_img_pos
        x1 = max(center_x - half_crop, 0)
        y1 = max(center_y - half_crop, 0)
        x2 = min(center_x + half_crop, hdmap_orig.shape[1])
        y2 = min(center_y + half_crop, hdmap_orig.shape[0])
        cropped_img = ego_overlay[y1:y2, x1:x2]
        
        # ----------------------------
        # 3-4. Crop된 이미지 회전: Ego의 heading이 위쪽(상단)을 향하도록
        # ----------------------------
        rotation_angle = 90 - ego_heading  # (양수면 반시계 방향)
        crop_center = (cropped_img.shape[1] // 2, cropped_img.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(crop_center, rotation_angle, 1.0)
        rotated_img = cv2.warpAffine(cropped_img, rot_mat, (cropped_img.shape[1], cropped_img.shape[0]))
        
        # ----------------------------
        # 3-5. 결과 저장 (예: hd_map_10.png)
        # ----------------------------
        output_filename = os.path.join(output_folder, f"gt_hd_{label}.png")
        cv2.imwrite(output_filename, rotated_img)
        print(f"Ego BEV 오버레이 image saved: {output_filename}")
