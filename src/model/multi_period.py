from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Block(nn.Module):
    def __init__(self, p, k=5, s=3):
        super().__init__()
        self.p = p
        self.layers = nn.ModuleList()
        res = [1, 32, 128, 512, 1024, 1024]
        for i in range(4):
            self.layers.append(weight_norm(nn.Conv2d(res[i], res[i + 1], (k, 1), (s, 1), padding=(2, 0))))
        self.layers.append(weight_norm(nn.Conv2d(res[4], res[5], (k, 1), 1, padding=(2, 0))))
        self.last_conv = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        features = []
        time = x.shape[-1]
        x = F.pad(x, (0, (self.p - time % self.p) % self.p), "reflect")
        batch, ch, time = x.shape
        x = x.reshape(batch, ch, time // self.p, self.p)
        for conv in self.layers:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.last_conv(x)
        features.append(x)
        return x.reshape(batch, -1), features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(i) for i in [2, 3, 5, 7, 11]])

    def forward(self, real, fake, detach=False, **batch):
        if detach:
            fake = fake.detach()
        ans_tp, ans_fp, feat_tp, feat_fp = [], [], [], []
        for b in self.blocks:
            ans, f = b(real)
            ans_tp.append(ans)
            feat_tp.append(f)
            ans, f = b(fake)
            ans_fp.append(ans)
            feat_fp.append(f)
        return {'ans_tp': ans_tp, 'ans_fp': ans_fp, 'feat_tp': feat_tp, 'feat_fp': feat_fp}
