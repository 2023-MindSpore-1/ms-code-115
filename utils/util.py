import yaml
import os, sys
import numpy as np
import mindspore as ms
from sklearn.cluster import KMeans


def read_yaml(path):
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def predict(latent, num_classes):
    if isinstance(latent, ms.Tensor):
        latent = latent.numpy()
    model = KMeans(n_clusters=num_classes, n_init=100)
    return model.fit_predict(latent)
