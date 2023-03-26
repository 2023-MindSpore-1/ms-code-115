import copy
from .loss_zoo import *

class CombineLoss:
    def __init__(self, funcs: list):
        self.funcs = funcs

    def __call__(self, output: dict):
        losses = {}
        loss = 0
        for config in self.funcs:
            cfg = copy.deepcopy(config)
            name = list(cfg)[0]
            weight = cfg[name].pop('weight')
            cur_loss = eval(name)(**cfg[name])(**output)
            losses.update(cur_loss)
            cur_loss = weight * cur_loss[name]
            loss += cur_loss
            
        losses['loss'] = loss
        return losses

def build_loss(config):
    cfg = copy.deepcopy(config)
    return CombineLoss(cfg)
