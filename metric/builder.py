from .metric_zoo import *
import copy
from .util import align

class CombineMetric:
    def __init__(self, funcs: list, num_classes: int):
        self.funcs = funcs
        self.num_classes = num_classes

    def __call__(self, y_pred, y_true, aligned=False):
        metric = {}
        if aligned == False:
            y_pred = align(y_pred, y_true, self.num_classes)

        for func in self.funcs:
            metric.update(func(y_pred, y_true))
        return metric

def build_metric(config, num_classes):
    cfg = copy.deepcopy(config)
    funcs = []
    for m in cfg:
        name = list(m)[0]
        c = {} if m[name] is None else m[name]
        funcs.append(eval(name)(**c))
    return CombineMetric(funcs, num_classes)

    
