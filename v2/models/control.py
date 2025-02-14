import torch
import torch.nn as nn

class FutureControlMLP(nn.Module):
    def __init__(self, seq_len, input_dim, hidden_dim, output_dim):
        """
        Args:
            seq_len (int): 시퀀스 길이 (현재 + 미래 프레임 수)
            input_dim (int): 각 시점의 feature 차원
            hidden_dim (int): MLP 내부 hidden layer 크기
            output_dim (int): 최종 출력 차원 ([throttle, steer, brake])
        """
        super(FutureControlMLP, self).__init__()
        # (seq_len * input_dim)을 입력으로 받아 hidden_dim으로 매핑
        self.fc1 = nn.Linear(seq_len * input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # hidden_dim에서 최종 제어값 (output_dim=3 등)으로 매핑
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch_size, seq_len, input_dim]
                             현재와 미래 프레임 정보를 포함한 시퀀스 입력
        Returns:
            future_control_value (torch.Tensor): [batch_size, output_dim]
                                                미래 제어값 (throttle, steer, brake 등)
        """
        b, s, d = x.shape  # batch_size, seq_len, input_dim

        # 시퀀스 차원을 펼쳐서 MLP 입력으로 사용
        x = x.view(b, s * d)  # [batch_size, seq_len * input_dim]

        # MLP 통과
        x = self.fc1(x)       # [batch_size, hidden_dim]
        x = self.relu(x)
        future_control_value = self.fc2(x)  # [batch_size, output_dim]

        return future_control_value


class ControlMLP(nn.Module):
    def __init__(self, future_steps, control_dim=3, hidden_dim=256, dropout_rate=0.1):
        """
        Args:
            future_steps (int): 미래 예측 단계 수.
            control_dim (int): 출력 제어값의 차원 (예: 스티어링, 가속도, 브레이크 등).
            hidden_dim (int): MLP 내부 은닉층 차원.
        """
        super(ControlMLP, self).__init__()
        # gru_bev의 spatial feature를 flatten한 크기: 256 * 25 * 25
        self.gru_bev_fc = nn.Linear(256 * 25 * 25, 128)
        
        # 결합된 feature: front_feature (128) + ego_gru_output (256) + gru_bev (128) = 384
        self.mlp = nn.Sequential(
            nn.Linear((128+256+128+12)*3, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_dim*2,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(hidden_dim, control_dim)
        )

    def forward(self, front_feature, gru_bev, ego_gru_output, ego_state):
        """
        Args:
            front_feature: Tensor, shape [B, 128]
            gru_bev: Tensor, shape [B, future_steps, 256, 25, 25]
            ego_gru_output: Tensor, shape [B, future_steps, 128]
            Returns:
            control: Tensor, shape [B, future_steps, control_dim]
        """
        B, T, C, H, W = gru_bev.shape
        
        # 1. gru_bev 처리: flatten spatial dimension 후 선형 계층 적용
        gru_bev_flat = gru_bev.view(B, T, -1)  # shape: [B, T, 256*25*25]
        gru_bev_feat = self.gru_bev_fc(gru_bev_flat)  # shape: [B, T, 128]
            
        # 2. front_feature 확장: [B, 128] -> [B, T, 128]
        front_feature_exp = front_feature.unsqueeze(1).expand(B, T, -1)
  
        # 3. 세 feature를 concat: [B, T, 512*3]
        combined_features = torch.cat([front_feature_exp, ego_gru_output, gru_bev_feat], dim=-1)
            
        combined_flat = combined_features.view(B, -1) # [B, T * 512 * 3]
        ego_state = ego_state.view(B,-1) # [B, T*12]

        combined_flat = torch.cat((combined_flat,ego_state), dim=1)
        control = self.mlp(combined_flat)  # shape: [B, control_dim]
            
        return control    
    