from .common import MultiViewLinearBnReLU, LinearBNReLU
from mindspore import nn
import mindspore as ms
from mindspore import ops


class Encoder(nn.Cell):
    def __init__(self,
                 dims: list,
                 activate: str = "ReLU",
                 **kwargs):
        super(Encoder, self).__init__()
        self.net = MultiViewLinearBnReLU(dims,
                                         activate=activate,
                                         **kwargs)

    def construct(self, xs):
        return self.net(xs)


class Decoder(nn.Cell):
    def __init__(self,
                 dims: list,
                 activate: str = "ReLU",
                 **kwargs):
        super(Decoder, self).__init__()
        self.net = MultiViewLinearBnReLU(dims,
                                         activate=activate,
                                         **kwargs)

    def construct(self, xs):
        return self.net(xs)


class AutoEncoder(nn.Cell):
    def __init__(self,
                 encoder: dict,
                 decoder: dict,
                 fusion: dict):
        super(AutoEncoder, self).__init__()
        self.enc = Encoder(**encoder)
        self.dec = Decoder(**decoder)
        self.fusion = nn.SequentialCell([
            LinearBNReLU(**fusion),
            nn.Softmax(axis=-1),
        ])

    def construct(self, xs: list, mask_matrix: ms.Tensor = None):
        latent = self.enc(xs)
        cat_latent = ops.Concat(axis=1)(latent)
        ws = self.fusion(cat_latent)  # n x v
        mask_matrix = ops.Stack()(mask_matrix)
        matrix = mask_matrix.squeeze().transpose((1, 0))
        w = matrix * (ws * matrix).sum(axis=0, keepdims=True) / matrix.sum(axis=0, keepdims=True)
        ws = ws / ops.abs(w).sum(axis=-1, keepdims=True)
        z = 0
        for w, h in zip(ws.T, latent):
            w = ops.expand_dims(w, 1)
            z += w * h

        mz = [z for _ in xs]
        fusion2recon = self.dec(mz)
        out = self.dec(latent)
        return out, fusion2recon, latent, z
