import os
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import re
import logging

# 로거 설정: 파일과 콘솔 모두에 로그를 기록
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 파일 핸들러 (로그 파일 저장)
file_handler = logging.FileHandler('data_loader_debug.log')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# -------------------------------------------------------------------
# EGO_INFO 파일 내부에서 실제 프레임 번호를 추출하는 함수
def get_ego_frame(file_path):
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # 예: "frame: 690" 혹은 "Frame: 690" 등
                if "frame:" in line.lower():
                    parts = line.split(":")
                    if len(parts) > 1:
                        frame_num = parts[1].strip()
                        return int(frame_num)
    except Exception as e:
        logger.warning(f"Error reading {file_path}: {e}")
    # 내부에서 찾지 못하면 파일 이름에서 추출 (예: "690.txt")
    basename = os.path.basename(file_path)
    match = re.search(r'(\d+)', basename)
    return int(match.group(1)) if match else -1

# -------------------------------------------------------------------
class camDataLoader(Dataset):
    def __init__(self, root_dir, num_timesteps=3, image_size=(270, 480), map_size=(200, 200), 
                 hd_map_dir="HD_MAP", gt_hd_map_dir="GT_HD", ego_info_dir="EGO_INFO", 
                 traffic_info_dir="TRAFFIC_INFO", num_traffic_classes=10):
        """
        Args:
            root_dir (str): CALIBRATION 및 시나리오 폴더들이 있는 루트 디렉토리.
            num_timesteps (int): 입력 프레임 개수 (이후 2프레임은 미래 GT로 사용).
            image_size (tuple): 출력 이미지 크기 (height, width).
            map_size (tuple): HD Map 이미지 크기 (height, width).
            hd_map_dir (str): 각 시나리오 내 HD_MAP 폴더 이름.
            gt_hd_map_dir (str): 각 시나리오 내 GT_HD_MAP 폴더 이름 (BEV segmentation GT용).
            ego_info_dir (str): 각 시나리오 내 EGO_INFO 폴더 이름.
            traffic_info_dir (str): 각 시나리오 내 TRAFFIC_INFO 폴더 이름.
            num_traffic_classes (int): traffic GT의 클래스 수 (one-hot 인코딩에 사용).
        """
        self.root_dir = root_dir
        self.num_timesteps = num_timesteps          # 입력 프레임 개수
        self.future_steps = 2                       # 미래 GT 프레임 개수 (고정)
        self.total_steps = self.num_timesteps + self.future_steps  # 총 사용 프레임 수 (예: 5)
        self.map_size = map_size
        self.hd_map_dir = hd_map_dir
        self.gt_hd_map_dir = gt_hd_map_dir           # GT_HD_MAP 폴더 이름 (BEV segmentation GT)
        self.ego_info_dir = ego_info_dir
        self.traffic_info_dir = traffic_info_dir      # TRAFFIC_INFO 폴더 이름
        self.num_traffic_classes = num_traffic_classes  # traffic GT 클래스 개수

        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),   # 이미지 크기 고정
            transforms.ToTensor()              # 텐서 변환
        ])

        # Calibration 파일 로드
        calibration_dir = os.path.join(root_dir, "Calibration")
        assert os.path.isdir(calibration_dir), f"Calibration directory not found: {calibration_dir}"
        logger.debug(f"Calibration directory found: {calibration_dir}")

        camera_files = [f for f in os.listdir(calibration_dir) if f.endswith(".npy")]
        camera_ids = sorted(set(f.split("__")[0] for f in camera_files))
        self.num_cameras = len(camera_ids)
        logger.debug(f"Found camera IDs: {camera_ids}")

        self.intrinsic_data = []
        self.extrinsic_data = []
        for camera_id in camera_ids:
            intrinsic_file = os.path.join(calibration_dir, f"{camera_id}__intrinsic.npy")
            extrinsic_file = os.path.join(calibration_dir, f"{camera_id}__extrinsic.npy")

            assert os.path.isfile(intrinsic_file), f"Intrinsic file not found: {intrinsic_file}"
            assert os.path.isfile(extrinsic_file), f"Extrinsic file not found: {extrinsic_file}"

            logger.debug(f"Loading calibration files for camera {camera_id}:")
            logger.debug(f"    Intrinsic: {intrinsic_file}")
            logger.debug(f"    Extrinsic: {extrinsic_file}")

            self.intrinsic_data.append(torch.tensor(np.load(intrinsic_file), dtype=torch.float32))
            self.extrinsic_data.append(torch.tensor(np.load(extrinsic_file), dtype=torch.float32))

        # 시나리오 폴더 로드
        def extract_scenario_number(scenario_name):
            match = re.search(r'(\d+)', scenario_name)
            return int(match.group(1)) if match else float('inf')

        all_scenario_dirs = sorted(
            [os.path.join(root_dir, d) for d in os.listdir(root_dir)
             if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("R_KR_")],
            key=extract_scenario_number
        )
        logger.debug(f"Found scenario directories: {all_scenario_dirs}")

        # 내부에서 사용할 extract_number 함수 (파일명 내 숫자 추출)
        def extract_number(file_path):
            basename = os.path.basename(file_path)
            match = re.search(r'(\d+)', basename)
            return int(match.group(1)) if match else -1

        # 각 시나리오별 데이터(카메라, EGO_INFO, TRAFFIC_INFO) 수집 및 valid index 계산
        # 전체 시나리오에서 _전체 시퀀스를 구성할 수 있는 시작 인덱스_ (즉, valid sample)만 valid_indices에 등록합니다.
        self.scenario_dirs = []  # valid 시나리오 디렉토리들 (부분적으로 사용)
        self.camera_data = []    # 각 시나리오의 카메라 파일 리스트 (리스트 내에 카메라별 이미지 리스트)
        self.ego_data = []       # 각 시나리오의 필터링된 EGO_INFO 파일 리스트
        self.traffic_data = []   # 각 시나리오의 필터링된 TRAFFIC_INFO 파일 리스트 (없으면 None)
        self.valid_indices = []  # 전체 데이터셋에서 (시나리오 idx, 시작 프레임 인덱스) 튜플 리스트
        # self.front_data = []

        for scenario_dir in all_scenario_dirs:
            logger.debug(f"Processing scenario directory: {scenario_dir}")
            # CAMERA 데이터 처리
            camera_dirs = sorted(
                [os.path.join(scenario_dir, d) for d in os.listdir(scenario_dir)
                 if d.upper().startswith("CAMERA_")]
            )
            if not camera_dirs:
                logger.warning(f"No CAMERA_* folders found in {scenario_dir}. Skipping scenario.")
                continue
            logger.debug(f"Found CAMERA directories: {camera_dirs}")
        
            camera_files = []
            for camera_dir in camera_dirs:
                if not os.path.isdir(camera_dir):
                    logger.warning(f"{camera_dir} is not a directory. Skipping scenario.")
                    continue
                files = sorted(
                    [os.path.join(camera_dir, f) for f in os.listdir(camera_dir) if f.endswith(".jpeg")],
                    key=extract_number
                )
                if not files:
                    logger.warning(f"No .jpeg files found in {camera_dir}. Skipping scenario.")
                    continue
                logger.debug(f"Camera directory {camera_dir} has {len(files)} image files. Sample: {files[:2]}")
                camera_files.append(files)

            if not camera_files:
                continue
            # 모든 카메라가 동일한 프레임 수를 가지고 있는지 확인
            num_camera_frames = len(camera_files[0])
            if not all(len(files) == num_camera_frames for files in camera_files):
                logger.warning(f"Mismatch in number of frames across cameras in {scenario_dir}. Skipping scenario.")
                continue

            # EGO_INFO 파일 처리 (내부 프레임 번호 기준 정렬)
            ego_info_path = os.path.join(scenario_dir, self.ego_info_dir)
            if not os.path.isdir(ego_info_path):
                logger.warning(f"EGO_INFO directory not found: {ego_info_path}. Skipping scenario.")
                continue
            ego_files = sorted(
                [os.path.join(ego_info_path, f) for f in os.listdir(ego_info_path) if f.endswith(".txt")],
                key=get_ego_frame
            )
            if not ego_files:
                logger.warning(f"No EGO_INFO files found in {ego_info_path}. Skipping scenario.")
                continue
            logger.debug(f"EGO_INFO files in {ego_info_path}: {ego_files[:2]} ... (총 {len(ego_files)}개)")

            # EGO_INFO 파일 필터링: 내부의 프레임 번호가 10의 배수인 경우만 사용
            filtered_ego_files = [f for f in ego_files if get_ego_frame(f) % 10 == 0]
            if len(filtered_ego_files) < self.total_steps:
                logger.warning(f"Not enough filtered EGO_INFO files in {ego_info_path} for a full sequence. Skipping scenario.")
                continue

            # TRAFFIC_INFO 파일 처리 및 필터링 (EGO_INFO와 동일한 기준 적용)
            traffic_info_path = os.path.join(scenario_dir, self.traffic_info_dir)
            if os.path.isdir(traffic_info_path):
                traffic_files = sorted(
                    [os.path.join(traffic_info_path, f) for f in os.listdir(traffic_info_path) if f.endswith(".txt")],
                    key=extract_number
                )
                # EGO_INFO와 같은 수라면 동일하게 10의 배수 조건으로 필터링
                if len(traffic_files) == len(ego_files):
                    filtered_traffic_files = [f for f in traffic_files if extract_number(f) % 10 == 0]
                else:
                    filtered_traffic_files = traffic_files
                if len(filtered_traffic_files) != len(filtered_ego_files):
                    logger.warning(f"Number of filtered TRAFFIC_INFO files ({len(filtered_traffic_files)}) does not match filtered EGO_INFO files ({len(filtered_ego_files)}) in {scenario_dir}.")
                traffic_entry = filtered_traffic_files
            else:
                logger.warning(f"TRAFFIC_INFO directory not found: {traffic_info_path}")
                traffic_entry = None

            # 각 시나리오 내에서 전체 total_steps(예: 5) 프레임을 구성할 수 있는 시작 인덱스만 valid하게 사용
            # (예: num_camera_frames가 100이라면, 시작 인덱스 0 ~ (100 - total_steps) 까지만 사용)
            num_valid_samples = min(num_camera_frames, len(filtered_ego_files)) - self.total_steps + 1
            if num_valid_samples <= 0:
                logger.warning(f"Not enough frames in scenario {scenario_dir} to form a full sequence. Skipping scenario.")
                continue

            # valid한 시나리오 데이터를 별도 리스트에 저장
            self.scenario_dirs.append(scenario_dir)
            self.camera_data.append(camera_files)
            self.ego_data.append(filtered_ego_files)
            self.traffic_data.append(traffic_entry)
            # self.front_data.append(front_view)

            # 해당 시나리오 내에서 각 valid 시작 인덱스를 전역 valid index에 등록
            # (각 튜플: (해당 valid 시나리오 내 인덱스, 시작 frame index))
            for start_idx in range(num_valid_samples):
                self.valid_indices.append((len(self.scenario_dirs) - 1, start_idx))

        logger.debug(f"Total number of valid samples: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # valid_indices 매핑을 사용하여 시나리오와 시작 프레임 인덱스를 결정
        try:
            scenario_idx, frame_idx = self.valid_indices[idx]
        except IndexError:
            raise IndexError("Index out of range in dataset.")

        scenario_dir = self.scenario_dirs[scenario_idx]
        camera_files = self.camera_data[scenario_idx]
        ego_files = self.ego_data[scenario_idx]
        traffic_files = self.traffic_data[scenario_idx]  # None일 수도 있음
        temporal_images = []
        ego_info_data = []
        control_info_data = []
        traffic_class_seq = []   # 입력 시퀀스 내 프레임별 traffic classification 값

        # total_steps(예: 5) 프레임에 대해 데이터를 로드
        for t in range(self.total_steps):
            # --- 이미지 로드 (CAMERA) ---
            images_per_camera = []
            for cam_idx in range(len(camera_files)):
                image_path = camera_files[cam_idx][frame_idx + t]
                logger.debug(f"Loading image file: {image_path}")
                image = Image.open(image_path).convert("RGB")
                image = self.image_transform(image)
                images_per_camera.append(image)
            temporal_images.append(torch.stack(images_per_camera, dim=0))  # (num_cameras, C, H, W)

            # --- EGO_INFO 파일 로드 ---
            ego_file_path = ego_files[frame_idx + t]
            logger.debug(f"Loading EGO_INFO file: {ego_file_path}")
            ego_info_dict = {}
            with open(ego_file_path, 'r') as f:
                for line in f:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        if key in {"position", "orientation", "velocity"}:
                            ego_info_dict[key] = list(map(float, value.split()))
                        elif key in {"accel", "brake", "steer"}:
                            if "control" not in ego_info_dict:
                                ego_info_dict["control"] = []
                            ego_info_dict["control"].append(float(value))
                        elif key == "trafficlightid":
                            ego_info_dict["trafficlightid"] = value
            # flatten: 각 키에 대해 값이 없으면 기본값 [0.0, 0.0, 0.0] 사용
            ego_info_values = []
            for k in ["position", "orientation", "velocity", "control"]:
                default_length = 3
                ego_info_values.extend(ego_info_dict.get(k, [0.0] * default_length))
            ego_info_data.append(ego_info_values)

            # control 정보만 따로 저장
            control_data = ego_info_dict.get("control", [0.0] * 3)
            control_info_data.append(control_data)

            # --- TRAFFIC_INFO 파일 로드 및 traffic classification 결정 (입력 시퀀스에 대해서만) ---
            if t < self.num_timesteps:
                if "trafficlightid" not in ego_info_dict or ego_info_dict["trafficlightid"] in [None, "", "null"]:
                    traffic_class = 0
                else:
                    ego_traffic_id = ego_info_dict["trafficlightid"]
                    if traffic_files is not None:
                        traffic_file_path = traffic_files[frame_idx + t]
                        logger.debug(f"Loading TRAFFIC_INFO file: {traffic_file_path}")
                        with open(traffic_file_path, 'r') as f_traffic:
                            traffic_dict = {}
                            for line in f_traffic:
                                if ":" in line:
                                    k2, v2 = line.split(":", 1)
                                    k2 = k2.strip()
                                    tokens = v2.strip().split()
                                    if tokens:
                                        traffic_dict[k2] = int(tokens[-1])
                        traffic_class = traffic_dict.get(ego_traffic_id, 0)
                    else:
                        traffic_class = 0
                traffic_class_seq.append(traffic_class)

        # 최종 traffic classification: 입력 시퀀스의 마지막 프레임 값 사용
        current_traffic_class = traffic_class_seq[-1] if traffic_class_seq else 0
        traffic_gt = torch.zeros(self.num_traffic_classes, dtype=torch.float32)
        if current_traffic_class < self.num_traffic_classes:
            traffic_gt[current_traffic_class] = 1.0

        # EGO_INFO 텐서 생성 및 입력/미래 분리
        ego_info_tensor = torch.tensor(ego_info_data, dtype=torch.float32)  # (total_steps, feature_dim)
        ego_info_input = ego_info_tensor[:self.num_timesteps]
        ego_info_future = ego_info_tensor[self.num_timesteps:]
        control_info_tensor = torch.tensor(control_info_data, dtype=torch.float32)
        control_info_future = control_info_tensor[3,:]
        
        # --- 이미지 데이터 (CAMERA) ---
        temporal_images = torch.stack(temporal_images, dim=1)  # (num_cameras, total_steps, C, H, W)
        temporal_images = temporal_images.permute(1, 0, 2, 3, 4)  # (total_steps, num_cameras, C, H, W)
        images_input = temporal_images[:self.num_timesteps]
        images_future = temporal_images[self.num_timesteps:]

        first_camera_images = temporal_images[:, 0, :, :, :]  # Shape: (total_steps, C, H, W)
        crop_x1, crop_y1, crop_x2, crop_y2 = 50, 50, 150, 150  # 원하는 크롭 좌표 지정
        first_camera_cropped = first_camera_images[:, :, crop_y1:crop_y2, crop_x1:crop_x2]  # Shape: (total_steps, C, crop_H, crop_W)
        # front_camera = first_camera_cropped[:self.num_timesteps] 

        # --- Calibration (intrinsic & extrinsic) ---
        intrinsic = torch.stack(self.intrinsic_data, dim=0).unsqueeze(1).repeat(1, self.total_steps, 1, 1)
        extrinsic = torch.stack(self.extrinsic_data, dim=0).unsqueeze(1).repeat(1, self.total_steps, 1, 1)
        intrinsic = intrinsic.permute(1, 0, 2, 3)  # (total_steps, num_cameras, 3, 3)
        extrinsic = extrinsic.permute(1, 0, 2, 3)  # (total_steps, num_cameras, 4, 4)
        intrinsic_input = intrinsic[:self.num_timesteps]
        intrinsic_future = intrinsic[self.num_timesteps:]
        extrinsic_input = extrinsic[:self.num_timesteps]
        extrinsic_future = extrinsic[self.num_timesteps:]

        # --- HD Map 데이터 로드 및 전처리 ---
        hd_map_indices = [frame_idx + t for t in range(self.total_steps)]
        hd_map_images = self._load_hd_map(scenario_dir, hd_map_indices)
        hd_map_input = None
        hd_map_future = None
        if hd_map_images is not None:
            hd_map_images = np.stack([self._process_hd_map(frame) for frame in hd_map_images])
            hd_map_images = torch.tensor(hd_map_images, dtype=torch.float32)
            hd_map_input = hd_map_images[:self.num_timesteps]
            hd_map_future = hd_map_images[self.num_timesteps:]

        # --- GT HD Map 데이터 로드 및 전처리 (BEV segmentation GT) ---
        gt_hd_map_indices = [frame_idx + t for t in range(self.total_steps)]
        gt_hd_map_images = self._load_gt_hd_map(scenario_dir, gt_hd_map_indices)
        gt_hd_map_input = None
        gt_hd_map_future = None
        if gt_hd_map_images is not None:
            gt_hd_map_images = np.stack([self._process_gt_hd_map(frame) for frame in gt_hd_map_images])
            gt_hd_map_images = torch.tensor(gt_hd_map_images, dtype=torch.float32)
            gt_hd_map_input = gt_hd_map_images[:self.num_timesteps]
            gt_hd_map_future = gt_hd_map_images[self.num_timesteps:]
        return {
            "images_input": images_input,         # (num_timesteps, num_cameras, C, H, W)
            "images_future": images_future,         # (future_steps, num_cameras, C, H, W)
            "intrinsic_input": intrinsic_input,     # (num_timesteps, num_cameras, 3, 3)
            "intrinsic_future": intrinsic_future,   # (future_steps, num_cameras, 3, 3)
            "extrinsic_input": extrinsic_input,     # (num_timesteps, num_cameras, 4, 4)
            "extrinsic_future": extrinsic_future,   # (future_steps, num_cameras, 4, 4)
            "hd_map_input": hd_map_input,           # (num_timesteps, channels, H, W) 또는 None
            "hd_map_future": hd_map_future,         # (future_steps, channels, H, W) 또는 None
            "gt_hd_map_input": gt_hd_map_input,       # (num_timesteps, channels, H, W) 또는 None
            "gt_hd_map_future": gt_hd_map_future,     # (future_steps, channels, H, W) 또는 None
            "ego_info": ego_info_input,             # (num_timesteps, feature_dim)
            "ego_info_future": ego_info_future,     # (future_steps, feature_dim)
            "traffic": traffic_gt,                  # (num_traffic_classes,)
            "scenario": scenario_dir,
            "control": control_info_future
        }

    def _process_hd_map(self, hd_map_frame):
        """HD Map 프레임을 다중 채널 텐서로 전처리."""
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
        
        # 7채널 텐서로 스택 (채널 순서: ego, global path, drivable area, white lane, yellow lane, crosswalk, traffic light)
        return np.stack([ego_mask, global_path_mask, drivable_area_mask,
                        white_lane_mask, yellow_lane_mask, crosswalk_mask,
                        traffic_light_mask], axis=0)

    def _load_hd_map(self, scenario_path, frame_indices):
        """
        HD Map 이미지를 불러오고, 파일명을 숫자로 변환해 정렬된 순서로 로드합니다.
        Args:
            scenario_path (str): 현재 시나리오 디렉토리.
            frame_indices (list[int]): 순차적 인덱스 리스트 (0,1,2, ...).
        """
        hd_map_path = os.path.join(scenario_path, self.hd_map_dir)
        if not os.path.exists(hd_map_path):
            logger.warning(f"HD Map directory not found: {hd_map_path}")
            return None

        def extract_number(file_name):
            m = re.search(r'hd_map_(\d+)', file_name)
            return int(m.group(1)) if m else float('inf')

        hd_map_files = sorted(
            [os.path.join(hd_map_path, f) for f in os.listdir(hd_map_path) if f.endswith(".png")],
            key=extract_number
        )

        if not hd_map_files:
            logger.warning(f"No HD Map files found in directory: {hd_map_path}")
            return None

        hd_map_images = []
        for idx in frame_indices:
            if idx < len(hd_map_files):
                file_path = hd_map_files[idx]
                logger.debug(f"Loading HD Map file: {file_path}")
                hd_map_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if hd_map_image is None:
                    logger.warning(f"Failed to load HD Map image: {file_path}")
                    continue
                hd_map_image = cv2.resize(hd_map_image, self.map_size)
                hd_map_images.append(hd_map_image)
            else:
                logger.warning(f"Frame index {idx} exceeds available HD Map files")
        return np.array(hd_map_images)

    def _load_gt_hd_map(self, scenario_path, frame_indices):
        """
        GT_HD_MAP 폴더에서 BEV segmentation GT용 HD Map 이미지를 불러오고,
        정렬된 순서(오름차순)로 frame_indices에 따라 로드합니다.
        Args:
            scenario_path (str): 현재 시나리오 디렉토리.
            frame_indices (list[int]): 카메라 프레임 인덱스 리스트.
        """
        gt_hd_map_path = os.path.join(scenario_path, self.gt_hd_map_dir)
        if not os.path.exists(gt_hd_map_path):
            logger.warning(f"GT HD Map directory not found: {gt_hd_map_path}")
            return None

        def extract_number(file_name):
            m = re.search(r'gt_hd_(\d+)', file_name)
            return int(m.group(1)) if m else float('inf')

        gt_hd_map_files = sorted(
            [f for f in os.listdir(gt_hd_map_path) if f.endswith(".png")],
            key=extract_number
        )
        if not gt_hd_map_files:
            logger.warning(f"No GT HD Map files found in directory: {gt_hd_map_path}")
            return None

        gt_hd_map_images = []
        for idx in frame_indices:
            if idx < len(gt_hd_map_files):
                file_path = os.path.join(gt_hd_map_path, gt_hd_map_files[idx])
                logger.debug(f"Loading GT HD Map file: {file_path}")
                gt_hd_map_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if gt_hd_map_image is None:
                    logger.warning(f"Failed to load GT HD Map image: {file_path}")
                    continue
                gt_hd_map_image = cv2.resize(gt_hd_map_image, self.map_size)
                gt_hd_map_images.append(gt_hd_map_image)
            else:
                logger.warning(f"Frame index {idx} exceeds number of GT HD Map files in {gt_hd_map_path}")
        return np.array(gt_hd_map_images)

    def _process_gt_hd_map(self, hd_map_frame):
        """HD Map 프레임을 다중 채널 텐서로 전처리."""
        ego_color = np.array([0, 0, 255])
        global_path_color = np.array([255, 0, 0])
        drivable_area_color = np.array([200, 200, 200])
        white_lane_color = np.array([255, 255, 255])
        yellow_lane_color = np.array([0, 255, 255])
        crosswalk_color = np.array([0, 255, 0])
        traffic_light_color = np.array([0, 0, 255])
        object_info_color = np.array([128, 0, 128])  # 추가된 Object Info 색상

        # 각 채널의 픽셀값이 해당 색상과 정확히 일치하는지 확인
        ego_mask = np.all(hd_map_frame == ego_color, axis=-1).astype(np.float32)
        global_path_mask = np.all(hd_map_frame == global_path_color, axis=-1).astype(np.float32)
        drivable_area_mask = np.all(hd_map_frame == drivable_area_color, axis=-1).astype(np.float32)
        white_lane_mask = np.all(hd_map_frame == white_lane_color, axis=-1).astype(np.float32)
        yellow_lane_mask = np.all(hd_map_frame == yellow_lane_color, axis=-1).astype(np.float32)
        crosswalk_mask = np.all(hd_map_frame == crosswalk_color, axis=-1).astype(np.float32)
        traffic_light_mask = np.all(hd_map_frame == traffic_light_color, axis=-1).astype(np.float32)
        object_info_mask = np.all(hd_map_frame == object_info_color, axis=-1).astype(np.float32)  # 추가된 Object Info 마스크

        # 8채널 텐서로 스택 (채널 순서: ego, global path, drivable area, white lane, yellow lane, crosswalk, traffic light, object_info)
        return np.stack([ego_mask, global_path_mask, drivable_area_mask,
                        white_lane_mask, yellow_lane_mask, crosswalk_mask,
                        traffic_light_mask, object_info_mask], axis=0)