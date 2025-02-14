import torch
import subprocess

# CUDA 버전 확인
cuda_version = torch.version.cuda  # PyTorch에서 CUDA 버전 확인
if cuda_version is None:
    cuda_version = "CUDA is not available"

# cuDNN 버전 확인
cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "cuDNN is not available"

# PyTorch 버전 확인
torch_version = torch.__version__


# NVIDIA 드라이버 버전 확인 (리눅스 및 Windows 지원)
try:
    driver_version = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], encoding="utf-8").strip()
except Exception:
    driver_version = "NVIDIA driver not found"

# 결과 출력
print(f"CUDA Version      : {cuda_version}")
print(f"cuDNN Version    : {cudnn_version}")
print(f"PyTorch Version  : {torch_version}")
print(f"NVIDIA Driver Version: {driver_version}")
