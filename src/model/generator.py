from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class ResBlock(nn.Module):
    def __init__(self, ch, kernel_size, dil=[1, 3, 5]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(3):
            self.layers.append(nn.Sequential(nn.LeakyReLU(0.1),
                                             weight_norm(nn.Conv1d(ch, ch, kernel_size, 1, dilation=dil[i],
                                                                   padding=int((kernel_size * dil[i] - dil[i]) / 2))),
                                             nn.LeakyReLU(0.1),
                                             weight_norm(nn.Conv1d(ch, ch, kernel_size, 1, dilation=1,
                                                                   padding=int((kernel_size - 1) / 2)))
                                             )
                               )
        self.layers.apply(init)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = x + self.layers[i](x)
        return x


class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, res_k, res_dil):
        super().__init__()
        self.conv_tr = weight_norm(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=kernel_size,
                                                      stride=stride,
                                                      padding=(kernel_size - stride) // 2))
        self.mrf_blocks = nn.ModuleList()
        for k, d in zip(res_k, res_dil):
            self.mrf_blocks.append(ResBlock(out_ch, k, d))
        self.conv_tr.apply(init)
        self.mrf_blocks.apply(init)

    def forward(self, x):
        x = self.conv_tr(F.leaky_relu(x, 0.1))
        s = 0.0
        for i in range(len(self.mrf_blocks)):
            s += self.mrf_blocks[i](x)
        s = s / len(self.mrf_blocks)
        return s


class Generator(nn.Module):
    def __init__(self, inp_ch, out_ch, kernel_sizes, strides, res_k, res_dil):
        super().__init__()
        self.conv_start = weight_norm(nn.Conv1d(inp_ch, out_ch, 7, 1, padding=3))
        lst = []
        num_blocks = len(kernel_sizes)
        for i in range(num_blocks):
            lst.append(GenBlock(in_ch=out_ch // (2 ** i), out_ch=out_ch // (2 ** (1 + i)),
                                kernel_size=kernel_sizes[i], stride=strides[i], res_k=res_k,
                                res_dil=res_dil))
        self.gen_blocks = nn.Sequential(*lst)
        ch = out_ch // (2 ** num_blocks)
        self.conv_end = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))
        self.conv_start.apply(init)
        self.conv_end.apply(init)

    def forward(self, real_mel, **batch):
        x = self.conv_start(real_mel)
        x = self.gen_blocks(x)
        x = F.tanh(self.conv_end(F.leaky_relu(x)))
        return {'fake': x}


def init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:  # isinstance(nn.Conv1d) or isinstance(nn.ConvTranspose1d)
        m.weight.data.normal_(0.0, 0.01)
