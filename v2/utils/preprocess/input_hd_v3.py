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
    # y축은 이미지 좌측 상단이 (0,0)이므로, y값은 반전합니다.
    img_y = img_height - int(round((y - min_y + margin) * scale_factor))
    return [img_x, img_y]

# ----------------------------
# 3. 여러 시나리오 디렉토리 순회 (전체 시나리오 처리)
# ----------------------------
base_scenario_pattern = "/home/vip/hd/Dataset/R_KR_PR_Sangam_DP__HMG_Scenario_*"
scenario_dirs = sorted(glob.glob(base_scenario_pattern))

# Ego crop: 실제 좌표 50m 반경 → 100m×100m 영역 (픽셀 단위)
radius_m = 25  
crop_size_pixels = int(2 * radius_m * scale_factor)  # 2 * 50m * (픽셀/m)

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
    
    # HD‑MAP 이미지 복사본에 global path 오버레이
    hdmap_with_path = hdmap_orig.copy()
    global_path_points = []
    for idx, row in global_path_df.iterrows():
        # CSV 열 이름은 실제 CSV에 맞게 수정 (여기서는 "PositionX (m)"와 "PositionY (m)")
        pt = [float(row["PositionX (m)"]), float(row["PositionY (m)"])]
        pt_img = world_to_img(pt)
        global_path_points.append(pt_img)
    if global_path_points:
        global_path_points = np.array(global_path_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(hdmap_with_path, [global_path_points], isClosed=False, color=(255, 0, 0), thickness=2)
    
    # ----------------------------
    # 3-2. Ego 데이터 오버레이
    # ----------------------------
    ego_dir = os.path.join(scenario_dir, "EGO_INFO")
    if not os.path.exists(ego_dir):
        print(f"EGO_INFO folder not found in {scenario_dir}, skipping scenario.")
        continue
    ego_files = glob.glob(os.path.join(ego_dir, "*.txt"))
    # 파일명이 "2.txt", "3.txt", ... "10.txt", "20.txt" 등이라고 가정
    ego_files = sorted(ego_files, key=lambda f: float(os.path.splitext(os.path.basename(f))[0]))
    # 10의 배수 파일만 선택
    selected_ego_files = [f for f in ego_files if float(os.path.splitext(os.path.basename(f))[0]) % 10 == 0]
    
    # 결과 저장 폴더 (HD_MAP 폴더)
    output_folder = os.path.join(scenario_dir, "HD_MAP")
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
                # 예: "position: -7.723345 1061.216 -0.3244491"
                parts = line[len("position:"):].strip().split()
                ego_position = [float(parts[0]), float(parts[1])]
            elif line.startswith("orientation:"):
                # 예: "orientation: 0.5460205 -0.04907227 61.08992"
                parts = line[len("orientation:"):].strip().split()
                if len(parts) >= 3:
                    ego_heading = float(parts[2])
        
        if ego_position is None or ego_heading is None:
            continue
        
        # Ego의 월드 좌표를 이미지 좌표로 변환
        ego_img_pos = world_to_img(ego_position)
        
        # Ego 오버레이: 빨간 원 및 화살표
        ego_overlay = hdmap_with_path.copy()
        cv2.circle(ego_overlay, tuple(ego_img_pos), 5, (0, 0, 255), -1)
        arrow_length_world = 2.0  # 예: 2m 길이
        heading_rad = math.radians(ego_heading)
        dx = arrow_length_world * math.cos(heading_rad)
        dy = arrow_length_world * math.sin(heading_rad)
        arrow_endpoint_world = [ego_position[0] + dx, ego_position[1] + dy]
        arrow_img_endpoint = world_to_img(arrow_endpoint_world)
        cv2.arrowedLine(ego_overlay, tuple(ego_img_pos), tuple(arrow_img_endpoint), (0, 0, 255), 2)
        
        # ----------------------------
        # BEV Crop: Ego를 중심으로 실제 좌표 50m 반경 (100m×100m 영역) crop
        # ----------------------------
        half_crop = crop_size_pixels // 2
        center_x, center_y = ego_img_pos
        x1 = max(center_x - half_crop, 0)
        y1 = max(center_y - half_crop, 0)
        x2 = min(center_x + half_crop, hdmap_orig.shape[1])
        y2 = min(center_y + half_crop, hdmap_orig.shape[0])
        cropped_img = ego_overlay[y1:y2, x1:x2]
        
        # ----------------------------
        # Crop된 이미지 회전: Ego의 heading이 위쪽(상단)을 향하도록
        # ----------------------------
        # 회전 각도: Ego의 heading이 90°가 되어야 하므로, rotation_angle = 90 - ego_heading
        rotation_angle = 90 - ego_heading
        crop_center = (cropped_img.shape[1] // 2, cropped_img.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(crop_center, rotation_angle, 1.0)
        rotated_img = cv2.warpAffine(cropped_img, rot_mat, (cropped_img.shape[1], cropped_img.shape[0]))
        
        # ----------------------------
        # 결과 저장 (파일명 예: hd_map_10.png)
        # ----------------------------
        label = os.path.splitext(os.path.basename(ego_file))[0]
        output_filename = os.path.join(output_folder, f"hd_map_{label}.png")
        cv2.imwrite(output_filename, rotated_img)
        print(f"Ego BEV 오버레이 이미지 saved: {output_filename}")
