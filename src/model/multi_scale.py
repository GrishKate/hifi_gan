import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class Block(nn.Module):
    def __init__(self, spectral=False):
        super().__init__()
        self.layers = nn.ModuleList()
        norm = spectral_norm if spectral else weight_norm
        res = [1, 128, 128, 256, 512, 1024, 1024, 1024]
        k = [15] + [41] * 5 + [5]
        s = [1, 2, 2, 4, 4, 1, 1]
        g = [1, 4] + [16] * 4 + [1]
        p = [7] + [20] * 5 + [2]
        for i in range(4):
            self.layers.append(norm(weight_norm(nn.Conv1d(res[i], res[i + 1], k[i], s[i],
                                                          groups=g[i], padding=p[i]))))
        self.last_conv = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        batch = x.shape[0]
        features = []
        for conv in self.layers:
            x = F.leaky_relu(conv(x), 0.1)
            features.append(x)
        x = self.last_conv(x)
        features.append(x)
        return x.reshape(batch, -1), features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(spectral=True), Block(), Block()])
        self.pool = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, real, fake, detach=False, **batch):
        if detach:
            fake = fake.detach()
        ans_ts, ans_fs, feat_ts, feat_fs = [], [], [], []
        for i, b in enumerate(self.blocks):
            if i != 0:
                real = self.pool[i - 1](real)
                fake = self.pool[i - 1](fake)
            ans, f = b(real)
            ans_ts.append(ans)
            feat_ts.append(f)
            ans, f = b(fake)
            ans_fs.append(ans)
            feat_fs.append(f)
        return {'ans_ts': ans_ts, 'ans_fs': ans_fs, 'feat_ts': feat_ts, 'feat_fs': feat_fs}
