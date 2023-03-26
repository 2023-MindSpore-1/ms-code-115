from mindspore import ops
import mindspore as ms


def get_adjacency_matrix(h, miss_matrix=None, topk=3):
    miss_matrix = miss_matrix.squeeze()
    h = ops.L2Normalize()(h) #
    n, _ = h.shape
    sim_matrix = ops.MatMul()(h, h.T)
    _, indices = ops.TopK()(sim_matrix, topk)
    A = ops.Zeros()((n, n), ms.float32)
    ops.ScatterNd()(indices, ops.OnesLike()(A), A.shape)
    A = ms.numpy.where(A + A.T > 0, ops.OnesLike()(A), ops.OnesLike()(A))   # nxn
    miss_matrix = ops.MatMul()(ops.expand_dims(miss_matrix, 1), ops.expand_dims(miss_matrix, 0))  # nxn
    A = A * miss_matrix
    return A, sim_matrix