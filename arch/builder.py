from .model_zoo import *
import copy


def build_arch(config):
    cfg = copy.deepcopy(config)
    name = cfg.pop('name')
    net = eval(name)(**cfg)

    return net
