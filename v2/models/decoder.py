import torch
import torch.nn as nn
import torch.nn.functional as F

class TrafficSignClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TrafficSignClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # Hidden layer
        self.fc2 = nn.Linear(64, 32)        # Hidden layer
        self.fc_out = nn.Linear(32, num_classes)  # Output layer
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # [1, 128] -> [1, 64]
        x = self.relu(self.fc2(x))  # [1, 64] -> [1, 32]
        x = self.fc_out(x)          # [1, 32] -> [1, num_classes]
        return x
    
class EgoStateHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=12):
        """
        GRU에서 나온 128차원 feature를 21차원의 미래 ego 상태 값으로 변환하는 MLP 헤더.
        """
        super(EgoStateHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # 128 -> 64
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 64 -> 21
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, future_steps, 128)
        Returns:
            output: (batch_size, future_steps, 21)
        """
        return self.fc(x)

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, skip_dim, residual, factor):
        super().__init__()

        dim = out_channels // factor

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels))

        if residual:
            self.up = nn.Conv2d(skip_dim, out_channels, 1)
        else:
            self.up = None

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.conv(x)

        if self.up is not None:
            up = self.up(skip)
            up = F.interpolate(up, x.shape[-2:])

            x = x + up

        return self.relu(x)


class Decoder(nn.Module):
    def __init__(self, dim, blocks, residual=True, factor=2, dim_last=64, dim_max=8):
        super().__init__()

        layers = list()
        channels = dim

        for out_channels in blocks:
            layer = DecoderBlock(channels, out_channels, dim, residual, factor)
            layers.append(layer)

            channels = out_channels

        self.layers = nn.Sequential(*layers)
        self.out_channels = channels

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.out_channels, dim_last, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max, kernel_size=1)
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W) 형태의 입력
        """
        B, T, C, H, W = x.shape  # 원래 차원 저장

        # B와 T를 결합하여 (B*T, C, H, W)로 변환
        x = x.view(B * T, C, H, W)

        y = x
        for layer in self.layers:
            y = layer(y, x)  # Skip connection도 (B*T, ...) 형태 유지

        z = self.to_logits(y)

        # 다시 (B, T, C', H', W') 형태로 복원
        C_out, H_out, W_out = z.shape[1], z.shape[2], z.shape[3]
        z = z.view(B, T, C_out, H_out, W_out)

        return z
