import json
import numpy as np
import cv2

# === 1. JSON 파일 불러오기 ===
# drivable area 데이터
with open('/home/vip/hd/Dataset/Map_Data/R_KR_PR_Sangam_DP/road_mesh_out_line_polygon_set.json', 'r') as f:
    drivable_data = json.load(f)

# lane 데이터
with open('/home/vip/hd/Dataset/Map_Data/R_KR_PR_Sangam_DP/lane_boundary_set.json', 'r') as f:
    lane_data = json.load(f)

# crosswalk 데이터
with open('/home/vip/hd/Dataset/Map_Data/R_KR_PR_Sangam_DP/singlecrosswalk_set.json', 'r') as f:
    crosswalk_data = json.load(f)

# traffic light 데이터
with open('/home/vip/hd/Dataset/Map_Data/R_KR_PR_Sangam_DP/traffic_light_set.json', 'r') as f:
    traffic_light_data = json.load(f)

# === 2. 전체 좌표(드라이버블 영역, lane, crosswalk, traffic light)로 이미지 범위 결정 ===
all_points = []

# drivable area: 외곽 및 interior 좌표
for obj in drivable_data:
    if "points" in obj:
        all_points.extend(obj["points"])
    for interior in obj.get("interiors", []):
        all_points.extend(interior["points"])

# lane 데이터 좌표
for lane in lane_data:
    for pt in lane["points"]:
        all_points.append(pt)

# crosswalk 데이터 좌표
for cw in crosswalk_data:
    for pt in cw["points"]:
        all_points.append(pt)

# traffic light: 중심 좌표(point)
for tl in traffic_light_data:
    all_points.append(tl["point"])

all_points = np.array(all_points)
xs = all_points[:, 0]
ys = all_points[:, 1]
min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()

print(min_x)
print(min_y)
# === 3. 해상도 및 이미지 크기 설정 ===
scale_factor = 10   # 원래보다 10배 해상도
margin = 10        # 원래 좌표 기준 margin (픽셀)
orig_width = int(np.ceil(max_x - min_x)) + 2 * margin
orig_height = int(np.ceil(max_y - min_y)) + 2 * margin
width = orig_width * scale_factor
height = orig_height * scale_factor

# 최종 이미지 (컬러, BGR) 생성
final_img = np.zeros((height, width, 3), dtype=np.uint8)

def world_to_img(pt):
    """
    월드 좌표(pt)를 고해상도 이미지 좌표로 변환합니다.
    pt: [x, y, ...] (z는 무시)
    """
    x, y = pt[0], pt[1]
    img_x = int(round((x - min_x + margin) * scale_factor))
    img_y = height-int(round((y - min_y + margin) * scale_factor))
    return [img_x, img_y]

# === 4. drivable area segmentation 처리 ===
for obj in drivable_data:
    # 외곽 다각형은 회색으로 채움
    if "points" in obj:
        outer_pts = np.array([world_to_img(pt) for pt in obj["points"]], np.int32)
        outer_pts = outer_pts.reshape((-1, 1, 2))
        cv2.fillPoly(final_img, [outer_pts], color=(200, 200, 200))
    # interior 영역은 드라이버블 영역에서 제외 (검정)
    for interior in obj.get("interiors", []):
        inner_pts = np.array([world_to_img(pt) for pt in interior["points"]], np.int32)
        inner_pts = inner_pts.reshape((-1, 1, 2))
        cv2.fillPoly(final_img, [inner_pts], color=(0, 0, 0))

# === 5. lane segmentation 처리 ===
for lane in lane_data:
    pts = np.array([world_to_img(pt) for pt in lane["points"]], np.int32).reshape((-1, 1, 2))
    # lane_width를 선의 두께로 사용 (scale_factor 적용)
    thickness = max(1, int(round(lane.get("lane_width", 0.6) * scale_factor)))
    lane_color_list = lane.get("lane_color", ["white"])
    lane_color_str = lane_color_list[0].lower() if lane_color_list else "white"
    if lane_color_str == "white":
        color = (255, 255, 255)
    elif lane_color_str == "yellow":
        color = (0, 255, 255)
    else:
        color = (255, 255, 255)
    cv2.polylines(final_img, [pts], isClosed=False, color=color, thickness=thickness)

# === 6. crosswalk segmentation 처리 ===
for cw in crosswalk_data:
    pts = np.array([world_to_img(pt) for pt in cw["points"]], np.int32).reshape((-1, 1, 2))
    # crosswalk 내부를 초록색으로 채움
    cv2.fillPoly(final_img, [pts], color=(0, 255, 0))

# === 7. traffic light segmentation 처리 ===
for tl in traffic_light_data:
    # 중심 좌표 변환 (x, y만 사용)
    center = world_to_img(tl["point"])
    # traffic light의 크기: width, length (scale_factor 적용)
    rect_size = (tl["width"] * scale_factor, tl["length"] * scale_factor)
    # heading (회전각, 단위: 도)
    angle = tl["heading"]
    rotated_rect = (tuple(center), rect_size, angle)
    # 회전 사각형의 꼭짓점 좌표 계산
    box = cv2.boxPoints(rotated_rect)
    box = np.int0(box)
    
    # sub_type에 따라 색상 결정 (예: "red"가 있으면 빨강, "yellow"가 있으면 노랑)
    sub_types = [s.lower() for s in tl.get("sub_type", [])]
    if "red" in sub_types:
        color = (0, 255, 0)  # BGR: 빨강
    elif "yellow" in sub_types:
        color = (0, 255, 0)  # 노랑
    else:
        color = (0, 255, 0)  # 기본값: 초록
    # traffic light 영역 채우기 및 경계선 그리기
    cv2.drawContours(final_img, [box], 0, color, -1)

# === 8. 최종 이미지 저장 ===
cv2.imwrite("R_KR_PR_Sangam_DP_Full_HD.png", final_img)
print("최종 segmentation 이미지가 'final_segmentation_with_trafficlight_high_res.png'로 저장되었습니다.")
