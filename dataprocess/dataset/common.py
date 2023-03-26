from ..util import data_preprocess
import numpy as np
from abc import abstractmethod

class CommonDataset:
    def __init__(self, root_path, normalize, miss_rate, num_view, num_sample):
        self.root_path = root_path
        self.normalize = normalize
        self.X = []
        self.miss_rate = miss_rate
        self.num_sample = num_sample
        self.y = None

    def prepare(self):
        self._load_data()
        for i in range(len(self.X)):
            assert isinstance(self.X[i], np.ndarray)
            self.X[i] = self.X[i].astype('float32')
            self.X[i] = self.X[i] * self.miss_matrix[:, i].reshape(-1, 1)
        assert isinstance(self.y, np.ndarray)

        self.X = data_preprocess(self.X, self.normalize)
        self.y = self.y - self.y.min()
        self.y = self.y

    @abstractmethod
    def _load_data(self):
        pass

    def __getitem__(self, idx):
        return self.X[0][idx, :], self.X[1][idx, :], self.y[idx], self.miss_matrix[idx, 0], self.miss_matrix[idx, 1]

    def __len__(self):
        return len(self.y)