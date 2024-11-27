import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_loss(disc_class):
    loss = 0
    for out in disc_class:
        loss += torch.mean((1 - out) ** 2)
    return loss


def feat_loss(feat_t, feat_f):
    loss = 0
    for ft, ff in zip(feat_t, feat_f):
        for t, f in zip(ft, ff):
            loss += torch.mean(torch.abs(f - t))
    return 2 * loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ans_fp, ans_fs, feat_tp, feat_fp, feat_ts, feat_fs,
                real_mel, fake_mel, **batch):
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
        g_loss = gen_loss(ans_fp) + gen_loss(ans_fs)
        f_loss = feat_loss(feat_tp, feat_fp) + feat_loss(feat_ts, feat_fs)
        l1_loss = F.l1_loss(real_mel, fake_mel) * 45
        return {"gen_loss": g_loss + f_loss + l1_loss, "g_loss": g_loss, "feat_loss": f_loss,
                "l1_loss": l1_loss}
