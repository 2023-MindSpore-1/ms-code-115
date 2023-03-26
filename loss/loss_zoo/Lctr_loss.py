from sklearn.cluster import KMeans
import mindspore as ms
from mindspore import ops
from mindspore import nn
from utils.util import HiddenPrints


class LctrLoss:
    def __init__(self, sigma, taup, taus, num_classes):
        self.sigma = sigma
        self.taus = taus
        self.taup = taup
        self.num_classes = num_classes
        self.cri = nn.CrossEntropyLoss()
        self.kmeans = KMeans(n_clusters=num_classes)

    def __call__(self, fusion_latent: ms.Tensor, **kwargs):
        with HiddenPrints():
            self.kmeans.fit(fusion_latent.numpy())
            centers = self.kmeans.cluster_centers_
            if not isinstance(centers, ms.Tensor):
                centers = ms.Tensor(centers)
        P = ops.MatMul()(fusion_latent, centers.T) / self.taup
        P = ops.exp(P)
        P = P / P.sum(axis=-1, keepdims=True)
        sim_PL = ops.MatMul()(P, P.T)
        WL = ms.numpy.where(sim_PL > self.sigma, sim_PL, ops.ZerosLike()(sim_PL))
        WS = ops.MatMul()(fusion_latent, fusion_latent.T) / self.taus
        loss = self.cri(WS, WL)
        return {'LctrLoss': loss}

