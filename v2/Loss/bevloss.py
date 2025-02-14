import torch
import torch.nn as nn

class BEVSegmentationLoss(nn.Module):
    """
    BEV Segmentation Loss Module.
    Calculates loss across all time steps for BEV segmentation tasks.
    """
    def __init__(self, loss_fn=nn.CrossEntropyLoss(), apply_weights=False):
        """
        Args:
            loss_fn (nn.Module): Loss function to use (default: CrossEntropyLoss).
            apply_weights (bool): Whether to apply weights to each time step (default: False).
        """
        super(BEVSegmentationLoss, self).__init__()
        self.loss_fn = loss_fn
        self.apply_weights = apply_weights

    def forward(self, outputs, targets, weights=None):
        """
        Forward pass to calculate loss.

        Args:
            outputs (torch.Tensor): Predicted BEV segmentation maps 
                (shape: [batch_size, seq_len, num_classes, height, width]).
            targets (torch.Tensor): Ground truth BEV segmentation maps 
                (shape: [batch_size, seq_len, height, width]).
            weights (list or torch.Tensor, optional): Weights for each time step 
                (length: seq_len).

        Returns:
            torch.Tensor: Total loss across all time steps.
        """
        batch_size, seq_len, num_classes, height, width = outputs.shape

        # Validate weights if required
        if self.apply_weights:
            if weights is None:
                raise ValueError("Weights must be provided when apply_weights=True")
            if len(weights) != seq_len:
                raise ValueError("Length of weights must match the sequence length.")

        total_loss = 0.0

        for t in range(seq_len):
            # Extract outputs and targets for the current time step
            pred = outputs[:, t]  # [batch_size, num_classes, height, width]
            gt = targets[:, t]    # [batch_size, height, width]

            # Compute loss for the current time step
            step_loss = self.loss_fn(pred, gt)

            # Apply weights if specified
            if self.apply_weights:
                step_loss *= weights[t]

            # Accumulate loss
            total_loss += step_loss

        # Average the loss across all time steps
        avg_loss = total_loss / seq_len
        return avg_loss