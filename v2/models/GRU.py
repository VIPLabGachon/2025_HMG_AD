import torch
import torch.nn as nn

class BEVGRU(nn.Module):
    def __init__(self, input_channels, hidden_dim, output_dim, height, width, dropout_prob=0.1):
        super(BEVGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.height = height
        self.width = width

        # CNN을 사용하여 feature dimension을 hidden_dim으로 변환
        self.feature_extractor = nn.Sequential(
        nn.Conv2d(input_channels, hidden_dim // 2, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout2d(p=dropout_prob),

        nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(hidden_dim),
        nn.ReLU(),
        nn.Dropout2d(p=dropout_prob),
        nn.AdaptiveAvgPool2d((1, 1)),  # [batch*seq_len, hidden_dim, 1, 1]
        nn.Flatten()  # [batch*seq_len, hidden_dim]
    )

        # GRU Layer
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.dropout = nn.Dropout(p=dropout_prob)

        # Fully Connected Layer to project GRU output
        self.fc = nn.Linear(hidden_dim, output_dim * height * width)

    def forward(self, x, current_index=2, future_steps=2):
        batch_size, seq_len, channel, height, width = x.size()
        
        # CNN을 통해 feature dimension 줄이기
        x = x.view(batch_size * seq_len, channel, height, width)  # [batch*seq_len, channel, height, width]
        x = self.feature_extractor(x)  # [batch*seq_len, hidden_dim]
        x = x.view(batch_size, seq_len, self.hidden_dim)  # [batch, seq_len, hidden_dim]

        # GRU 처리
        gru_out, _ = self.gru(x)  # [batch, seq_len, hidden_dim]

        # 미래 예측 (Future Prediction)
        last_hidden = gru_out[:, -1, :].unsqueeze(1)  # [batch, 1, hidden_dim]
        future_pred = []

        for _ in range(future_steps):
            last_hidden, _ = self.gru(last_hidden)  # [batch, 1, hidden_dim]
            last_hidden = self.dropout(last_hidden)
            fc_out = self.fc(last_hidden)  # [batch, 1, output_dim*height*width]
            fc_out = fc_out.view(batch_size, 1, self.output_dim, self.height, self.width)
            future_pred.append(fc_out)

        future_pred = torch.cat(future_pred, dim=1)  # [batch, future_steps, output_dim, height, width]

        # Project GRU output to spatial dimensions
        output = self.fc(gru_out)  # [batch, seq_len, output_dim * height * width]
        output = output.view(batch_size, seq_len, self.output_dim, self.height, self.width)  # [batch, seq_len, output_dim, height, width]

        # Concatenate past, present, and future
        total_output = torch.cat([output, future_pred], dim=1)  # [batch, seq_len + future_steps, output_dim, height, width]

        # Extract current & future BEV
        current_bev = total_output[:, current_index].unsqueeze(1)  # [batch, 1, output_dim, height, width]
        future_bev = total_output[:, current_index + 1 : current_index + 1 + future_steps]  # [batch, 2, output_dim, height, width]
        future_bev = torch.cat((current_bev, future_bev), dim=1)
        return total_output, future_bev
    
class EgoStateGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout_prob=0.1):
        """
        Args:
            input_dim (int): Input feature dimension (e.g., 112 for ego state embedding).
            hidden_dim (int): Hidden state size of the GRU.
            output_dim (int): Output feature dimension for the GRU output projection.
            num_layers (int): Number of GRU layers.
        """
        super(EgoStateGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # GRU Layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout_prob)

        # Fully Connected Layer to project to desired output dimension
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Future Prediction을 위한 Linear Layer 추가
        self.future_fc = nn.Linear(output_dim, input_dim)

    def forward(self, x, future_steps=2):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len=3, input_dim].
            future_steps (int): Number of future steps to predict (default: 2).
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, future_steps, output_dim].
        """
        batch_size, seq_len, _ = x.size()

        # GRU forward pass (과거 2개 + 현재)
        gru_out, hidden_state = self.gru(x)  # [seq_len, batch_size, hidden_dim]

        # Initialize future predictions
        future_pred = []

        # 🔹 Future step input을 마지막 실제 입력의 변형값으로 설정
        future_input = self.future_fc(self.fc(gru_out[:, -1, :])).unsqueeze(1)  # [batch_size, 1, input_dim]

        for _ in range(future_steps):
            # 🔹 GRU로 future step 예측
            next_out, hidden_state = self.gru(future_input, hidden_state)
            next_out = self.dropout(next_out)
            next_output = self.fc(next_out.squeeze(1))  # [batch_size, output_dim]
            next_output = self.dropout(next_output)
            
            # 🔹 Append prediction
            future_pred.append(next_output.unsqueeze(1))  # [batch_size, 1, output_dim]

            # 🔹 다음 step의 input을 업데이트 (이전 예측값을 다시 입력으로 사용)
            future_input = self.future_fc(next_output).unsqueeze(1)  # [batch_size, 1, input_dim]

        # 🔹 Concatenate future predictions
        future_pred = torch.cat(future_pred, dim=1)  # [batch_size, future_steps, output_dim]
        hidden_state = hidden_state.transpose(0, 1)  # [B, Present, output_dim]
        future_pred_2 = torch.cat((hidden_state,future_pred),dim=1)    # [batch_size, present_step + future_steps, output_dim]
        return future_pred, future_pred_2
    
class FutureControlGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim: GRU 입력 차원 (현재와 미래 프레임 정보 포함)
            hidden_dim: GRU의 hidden state 크기
            output_dim: 출력 차원 ([throttle, steer, brake])
        """
        super(FutureControlGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)  # GRU hidden state에서 제어값 추출

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim] - 현재와 미래 프레임 정보를 포함한 시퀀스 입력
        Returns:
            future_control_value: [batch_size, output_dim] - 미래 제어값
        """
        output, _ = self.gru(x)  # GRU 모든 시점 출력: [batch_size, seq_len, hidden_dim]
        future_control_value = self.fc(output[:, -1])  # 마지막 시점 출력 사용: [batch_size, output_dim]
        return future_control_value