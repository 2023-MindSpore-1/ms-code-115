import mindspore.nn as nn
import mindspore as ms
from mindspore.nn import ReLU, Sigmoid, Softmax


class LinearBNReLU(nn.Cell):
    def __init__(self, dims: list, 
                       keep_prob: float = 0.8,
                       activate: str = "ReLU",
                       pre_activate: str = None,
                       pre_bn: bool = False,  # or int
                       out_activate: str = None):
        super(LinearBNReLU, self).__init__()
        """
        单视图数据的前向
        """
        net = []
        cur_dim = dims[0]
        if pre_bn:
            net.append(nn.BatchNorm1d(pre_bn))
        if pre_activate:
            net.append(eval(pre_activate)())

        for i, dim in enumerate(dims[1:-1]):                
            net.append(nn.Dense(cur_dim, dim))
            net.append(nn.BatchNorm1d(dim))
            net.append(nn.Dropout(keep_prob))
            if activate:
                net.append(eval(activate)())
            cur_dim = dim
            
        net.append(nn.Dense(cur_dim, dims[-1]))
        if out_activate:
            net.append(eval(out_activate)())
        self.layer = nn.SequentialCell(net)

    def construct(self, x: ms.Tensor):
        # print(x.shape)
        # print(type(x))
        return self.layer(x)
    

class MultiViewLinearBnReLU(nn.Cell):
    def __init__(self, 
                 mul_dims: list,
                 keep_prob: float = 0.8,
                 activate: str = "ReLU",
                 pre_activate: str = None,
                 pre_bn: bool = False,  # or int
                 out_activate: str = None):
        super(MultiViewLinearBnReLU, self).__init__()
        self.nets = nn.CellList([])
        for dim in mul_dims:
            self.nets.append(LinearBNReLU(dim, 
                                          keep_prob=keep_prob,
                                          pre_activate=pre_activate,
                                          pre_bn=pre_bn,  # or int
                                          activate=activate,
                                          out_activate=out_activate))
            
    def construct(self, xs: list):
        outs = []
        for x, net in zip(xs, self.nets):
            outs.append(net(x))

        return outs
