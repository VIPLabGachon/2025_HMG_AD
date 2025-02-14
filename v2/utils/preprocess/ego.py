import os
import glob

def parse_file(file_path):
    """
    파일을 읽어 key: value 형태의 딕셔너리로 반환합니다.
    - position, orientation, enu_velocity, velocity, angularVelocity, acceleration는 float 리스트로 변환.
    - accel, brake, steer, turn_signal_lamp는 float로 변환.
    - linkid, trafficlightid는 null이면 None으로 처리.
    """
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "position":
                data[key] = [float(x) for x in value.split()]
            elif key in ["orientation", "enu_velocity", "velocity", "angularVelocity", "acceleration"]:
                data[key] = [float(x) for x in value.split()]
            elif key in ["accel", "brake", "steer", "turn_signal_lamp"]:
                data[key] = float(value)
            elif key in ["linkid", "trafficlightid"]:
                data[key] = None if value.lower() == "null" else value
            else:
                data[key] = value
    return data

def reconstruct_text(data):
    """
    딕셔너리 데이터를 key: value 형식의 텍스트로 복원합니다.
    position은 리스트의 값을 공백으로 구분하여 출력하고,
    나머지 항목은 순서대로 출력합니다.
    """
    lines = []
    if "position" in data:
        pos_str = " ".join(str(v) for v in data["position"])
        lines.append("position: " + pos_str)
    for key in ["orientation", "enu_velocity", "velocity", "angularVelocity", "acceleration",
                "accel", "brake", "steer", "linkid", "trafficlightid", "turn_signal_lamp"]:
        if key in data:
            value = data[key]
            if isinstance(value, list):
                value_str = " ".join(str(v) for v in value)
            else:
                value_str = "null" if value is None else str(value)
            lines.append(f"{key}: {value_str}")
    return "\n".join(lines)

def numeric_key(file_path):
    """
    파일명을 숫자 기준으로 정렬하기 위한 키 함수.
    예: "2.txt" -> 2, "10.txt" -> 10, "101.txt" -> 101
    """
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    try:
        return int(name)
    except ValueError:
        return float('inf')

# 베이스 데이터셋 디렉토리 (예시)
base_data_dir = "/home/vip/hd/Dataset/"

# 시나리오 폴더 패턴: "R_KR_PG_KATRI__HMG_Scenario_0", "R_KR_PG_KATRI__HMG_Scenario_1", ...
scenario_pattern = os.path.join(base_data_dir, "R_KR_PR_Sangam_DP__HMG_Scenario_*")
scenario_dirs = sorted(glob.glob(scenario_pattern), key=lambda x: int(x.split("_")[-1]))

if not scenario_dirs:
    raise ValueError("시나리오 폴더가 존재하지 않습니다.")

for scenario_dir in scenario_dirs:
    ego_info_dir = os.path.join(scenario_dir, "EGO_INFO")
    if not os.path.exists(ego_info_dir):
        print(f"'{ego_info_dir}' 폴더가 존재하지 않습니다. 스킵합니다.")
        continue

    file_pattern = os.path.join(ego_info_dir, "*.txt")
    file_list = sorted(glob.glob(file_pattern), key=numeric_key)
    if not file_list:
        print(f"{ego_info_dir} 내에 텍스트 파일이 없습니다.")
        continue

    # 기준 파일의 position을 기준으로 (x, y)
    ref_file = file_list[0]
    ref_data = parse_file(ref_file)
    if "position" not in ref_data:
        print(f"기준 파일 {ref_file}에 position 데이터가 없습니다. 스킵합니다.")
        continue
    ref_position = ref_data["position"]
    ref_x, ref_y = ref_position[0], ref_position[1]

    # 결과 저장 폴더: EGO_INFO/relative/
    output_dir = ego_info_dir
    os.makedirs(output_dir, exist_ok=True)

    # 각 파일 처리: position만 상대좌표 변환 (x, y만 변환, z는 그대로 유지)
    for file in file_list:
        data = parse_file(file)
        if "position" in data:
            pos = data["position"]
            new_x = pos[0] - ref_x
            new_y = pos[1] - ref_y
            if len(pos) >= 3:
                new_z = pos[2]
                data["position"] = [new_x, new_y, new_z]
            else:
                data["position"] = [new_x, new_y]
        output_text = reconstruct_text(data)
        base_name = os.path.basename(file)
        output_path = os.path.join(output_dir, base_name)
        with open(output_path, "w") as f_out:
            f_out.write(output_text)
    
    print(f"{scenario_dir} 전처리 완료. 변환된 파일들은 '{output_dir}'에 저장되었습니다.")
