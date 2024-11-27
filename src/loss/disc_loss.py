import torch
from torch import nn


def disc_loss(ans_t, ans_f):
    loss = 0
    for t, f in zip(ans_t, ans_f, ):
        loss += torch.mean((1 - t) ** 2) + torch.mean(f ** 2)
    return loss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ans_tp, ans_fp, ans_ts, ans_fs, **batch):
        """
        Loss function calculation logic.

        Note that loss function must return dict. It must contain a value for
        the 'loss' key. If several losses are used, accumulate them into one 'loss'.
        Intermediate losses can be returned with other loss names.

        For example, if you have loss = a_loss + 2 * b_loss. You can return dict
        with 3 keys: 'loss', 'a_loss', 'b_loss'. You can log them individually inside
        the writer. See config.writer.loss_names.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        loss = disc_loss(ans_tp, ans_fp) + disc_loss(ans_ts, ans_fs)
        return {"disc_loss": loss}
