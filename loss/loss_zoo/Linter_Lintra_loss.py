import numpy as np
import numpy.linalg

from ..util import get_adjacency_matrix
from mindspore import nn, ops
import mindspore as ms
eps = 1e-9


class LintraLoss:
    def __init__(self, q, lam):
        self.temperature = 1.0
        self.cri = nn.CrossEntropyLoss()
        self.q = float(q)
        self.lam = lam

    def __call__(self, latents: list, miss_matrixs: list, **kwargs):
        loss = 0
        for latent, miss_matrix in zip(latents, miss_matrixs):
            A, sim_matrix = get_adjacency_matrix(latent, miss_matrix, 1)
            pos_mask = A
            neg_mask = ops.OnesLike()(A) - A
            sum_pos = pos_mask * sim_matrix
            sum_neg = neg_mask * sum_pos
            part1 = -(sum_pos ** self.q) / self.q
            part2 = (self.lam * (sum_pos + sum_neg) ** self.q) / self.q
            loss += -(part1.mean() + part2.mean())

        return {"LintraLoss": loss}


class LinterLoss:
    def __init__(self, q, lam):
        self.temperature = 1.0
        self.cri = nn.CrossEntropyLoss()
        self.q = q
        self.lam = lam

    def __call__(self, latents: list, miss_matrixs: list, fusion_latent: ms.Tensor, **kwargs):
        for i in range(len(latents)):
            latents[i] = latents[i] * miss_matrixs[i].reshape(-1, 1)
        h=ops.concat((fusion_latent, ops.concat(latents, axis=0)), axis=0)

        h_norm = ms.numpy.norm(h, ord=0.5, axis=1).reshape(-1, 1) + eps
        sim_matrix = (h / h_norm) @ (h / h_norm).T
        ones = ops.Ones()

        tag_s1 = ones((fusion_latent.shape[0], ), ms.int8)
        tag_s2 = ones((fusion_latent.shape[0], ), ms.int8)
        tag_s1s2 = ones((fusion_latent.shape[0], ), ms.int8)

        tag_s1 = ops.diag(tag_s1 * miss_matrixs[0].reshape(-1))
        tag_s2 = ops.diag(tag_s2 * miss_matrixs[1].reshape(-1))


        for miss_matrix in miss_matrixs:
            tag_s1s2 = tag_s1s2 * miss_matrix.reshape(-1)
        tag_s1s2 = ops.diag(tag_s1s2)


        pos_mask = ops.concat([
            ops.concat([tag_s1s2,tag_s1,tag_s2],axis=1),
            ops.concat([tag_s1,tag_s1,tag_s1s2],axis=1),
            ops.concat([tag_s2,tag_s1s2,tag_s2], axis=1)
        ],axis=0)

        oneslike=ops.OnesLike()
        neg_mask = oneslike(pos_mask)-pos_mask

        sum_pos = pos_mask * sim_matrix
        sum_neg = neg_mask * sum_pos

        part1 = -(sum_pos ** self.q) / self.q
        part2 = (self.lam * (sum_pos + sum_neg) ** self.q) / self.q
        loss = part1.mean() + part2.mean()

        return {"LinterLoss": loss}
