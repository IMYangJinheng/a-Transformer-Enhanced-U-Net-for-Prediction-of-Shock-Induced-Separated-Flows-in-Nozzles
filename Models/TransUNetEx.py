import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# =========================
# 基础层
# =========================
def create_layer(in_channels, out_channels, kernel_size,
                 wn=True, bn=True, activation=nn.ReLU,
                 conv=nn.Conv2d):
    padding = kernel_size // 2
    layer = conv(in_channels, out_channels, kernel_size, padding=padding)

    if wn:
        layer = weight_norm(layer)

    layers = [layer]

    if bn:
        layers.append(nn.BatchNorm2d(out_channels))

    if activation is not None:
        layers.append(activation(inplace=True))

    return nn.Sequential(*layers)


# =========================
# Adapter
# =========================
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(c_in, c_in // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in // reduction, c_in, 1)
        )

    def forward(self, x):
        return x + self.adapter(x)


# =========================
# Encoder / Decoder Blocks
# =========================
def create_encoder_block(in_channels, out_channels, kernel_size,
                         wn=True, bn=True, activation=nn.ReLU,
                         layers=2, use_adapter=False, adapter_reduction=4):
    encoder = []
    for i in range(layers):
        _in = in_channels if i == 0 else out_channels
        encoder.append(create_layer(_in, out_channels, kernel_size,
                                    wn, bn, activation, nn.Conv2d))

    adapter = Adapter(out_channels, adapter_reduction) if use_adapter else None
    return nn.Sequential(*encoder), adapter


def create_decoder_block(in_channels, out_channels, kernel_size,
                         wn=True, bn=True, activation=nn.ReLU,
                         layers=2, final_layer=False):
    decoder = []
    for i in range(layers):
        _in = in_channels * 2 if i == 0 else in_channels
        _out = out_channels if i == layers - 1 else in_channels
        _bn = False if final_layer and i == layers - 1 else bn
        _act = None if final_layer and i == layers - 1 else activation

        decoder.append(
            create_layer(_in, _out, kernel_size,
                         wn, _bn, _act, nn.ConvTranspose2d)
        )
    return nn.Sequential(*decoder)


def create_encoder(in_channels, filters, kernel_size,
                   wn=True, bn=True, activation=nn.ReLU,
                   layers=2, use_adapters=False):
    encoder = []
    adapters = []

    for i, f in enumerate(filters):
        _in = in_channels if i == 0 else filters[i - 1]
        enc, adp = create_encoder_block(
            _in, f, kernel_size, wn, bn,
            activation, layers,
            use_adapter=use_adapters
        )
        encoder.append(enc)
        adapters.append(adp)

    return nn.Sequential(*encoder), nn.ModuleList(adapters)


def create_decoder(out_channels, filters, kernel_size,
                   wn=True, bn=True, activation=nn.ReLU, layers=2):
    decoder = []
    for i in range(len(filters)):
        if i == 0:
            block = create_decoder_block(filters[i], out_channels,
                                         kernel_size, wn, bn,
                                         activation, layers,
                                         final_layer=True)
        else:
            block = create_decoder_block(filters[i], filters[i - 1],
                                         kernel_size, wn, bn,
                                         activation, layers)
        decoder = [block] + decoder
    return nn.Sequential(*decoder)


# =========================
# Transformer 部分
# =========================
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x),
                          self.norm1(x),
                          self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        B, N, C = x.shape
        device = x.device

        pos = torch.arange(N, device=device).unsqueeze(1)
        i = torch.arange(C, device=device).unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / C)
        angles = pos * angle_rates

        pe = torch.zeros_like(angles)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])

        return x + pe.unsqueeze(0)


# =========================
# TransUNetEx（最终模型）
# =========================
class TransUNetEx(nn.Module):
    def __init__(self,
                 in_channels=2,
                 out_channels=3,
                 kernel_size=3,
                 filters=[16, 32, 64],
                 layers=3,
                 weight_norm=True,
                 batch_norm=True,
                 activation=nn.ReLU,
                 final_activation=None,
                 use_adapters=False,
                 adapter_reduction=4,
                 transformer_layers=6,
                 num_heads=8):
        super().__init__()

        self.final_activation = final_activation
        self.use_adapters = use_adapters

        # Encoder
        self.encoder, self.encoder_adapters = create_encoder(
            in_channels, filters, kernel_size,
            weight_norm, batch_norm,
            activation, layers,
            use_adapters
        )

        # Decoder（与你原来一致：每个输出通道一个 decoder）
        self.decoders = nn.Sequential(
            *[create_decoder(1, filters, kernel_size,
                             weight_norm, batch_norm,
                             activation, layers)
              for _ in range(out_channels)]
        )

        embed_dim = filters[-1]
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads)
              for _ in range(transformer_layers)]
        )

    def encode(self, x):
        tensors, indices, sizes = [], [], []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            if self.use_adapters and self.encoder_adapters[i] is not None:
                x = self.encoder_adapters[i](x)

            sizes.append(x.size())
            tensors.append(x)
            x, ind = F.max_pool2d(x, 2, 2, return_indices=True)
            indices.append(ind)

        return x, tensors, indices, sizes

    def decode(self, x, tensors, indices, sizes):
        outputs = []
        for decoder in self.decoders:
            _x = x
            _tensors = tensors[:]
            _indices = indices[:]
            _sizes = sizes[:]

            for block in decoder:
                t = _tensors.pop()
                ind = _indices.pop()
                size = _sizes.pop()
                _x = F.max_unpool2d(_x, ind, 2, 2, output_size=size)
                _x = torch.cat([t, _x], dim=1)
                _x = block(_x)

            outputs.append(_x)

        return torch.cat(outputs, dim=1)

    def forward(self, x):
        # Encoder
        x, tensors, indices, sizes = self.encode(x)

        # Transformer bottleneck
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)   # (B, HW, C)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)

        # Decoder
        x = self.decode(x, tensors, indices, sizes)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x
