import mindspore as ms
from mindspore import nn


class LrLoss:
    def __init__(self):
        self.func = nn.MSELoss()

    def __call__(self, preds: list, reals: list,  miss_matrixs: list, fusion2recon: list = None, **kwargs):
        loss = 0
        for pred, fusion2x, real, mask in zip(preds, fusion2recon, reals, miss_matrixs):
            cur_loss = self.func(pred, real) * mask
            if fusion2recon:
                cur_fusion2recon_loss = self.func(fusion2x, real) * mask
                loss += cur_fusion2recon_loss.mean()
            loss += cur_loss.mean()
        return {'LrLoss': loss}
