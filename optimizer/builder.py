from mindspore.nn import SGD, Adam
import copy


def build_optimizer(config, net):
    cfg = copy.deepcopy(config)
    name = cfg.pop('name')    
    optimizer = eval(name)(net.trainable_params(), **cfg)
    return optimizer