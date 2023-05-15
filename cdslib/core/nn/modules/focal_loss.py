#
# Copyright (C) 2021 Apple Inc. All rights reserved.
#
import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        gamma: float = 1.0,
        alpha: float = 0.5,
        dynamic_balance: bool = False,
        reduction: str = "mean",
    ):
        """
        let :math:`p_t = p` if target = 1, :math:`1-p` if target != 1

        let :math:`a_t = a` if target = 1, :math:`1-p` if target != 1

        focal loss = :math:`- a_t (1 - p_t)^{\gamma} log(p_t)`

        For details:         https://arxiv.org/pdf/1708.02002.pdf

        Args:
            gamma (float):
                See class description.
            alpha (float):
                See class description.
            dynamic_balance (bool):
                If True, balance the classes.
            reduction (str):
                'mean', 'none'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.dynamic_balance = dynamic_balance
        self.reduction = reduction
        if self.reduction not in {"mean", "none"}:
            raise ValueError(f"wrong reduction type {reduction}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (N, C):
                tensor of logits.
            targets (N, ):
                :math:`targets_i \in {0,1}`
        Returns:
            (0,)
        """
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        if self.dynamic_balance:
            pos = torch.tensor(targets, dtype=torch.float, device=logits.device)
            prob_pos = torch.mean(pos)
            at = targets * prob_pos + (1 - targets) * (1.0 - prob_pos)
        else:
            at = targets * self.alpha + (1 - targets) * (1.0 - self.alpha)
        loss = 2 * at * (1 - pt).pow(self.gamma) * BCE_loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss


if __name__ == "__main__":
    print("End of script")
