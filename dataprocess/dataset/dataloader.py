from .common import CommonDataset
import os
import scipy.io as io
import numpy as np
import random
class DataLoader(CommonDataset):
    def __init__(self, root_path, normalize, miss_rate, num_view, num_sample):
        super(DataLoader, self).__init__(root_path, normalize, miss_rate, num_view, num_sample)
        self.prepare()

    def get_missmatrix(miss_rate: float, n_view: int, n_sample: int) -> np.ndarray:
        miss_matrix = np.ones((n_sample, n_view))
        all_index = list(range(n_sample))
        miss_num = int(miss_rate * n_sample)
        miss_index = random.sample(all_index, miss_num)
        miss_num_of_a_sample = list(range(1, n_view))
        miss_index_of_a_sample = list(range(0, n_view))
        for index in miss_index:
            miss_num_this_sample = random.choice(miss_num_of_a_sample)
            m = random.sample(miss_index_of_a_sample, miss_num_this_sample)
            miss_matrix[index][m] = 0

        return miss_matrix.astype('float32')
    def _load_data(self):
        path = os.path.join(self.root_path, 'data.mat')
        miss_path = os.path.join(self.root_path, 'mnist_percentDel_' + str(round(self.miss_rate, 1)) + ".mat")
        data = io.loadmat(path)
        self.X = [x.T.astype('float32') for x in data['X'][0]]
        self.y = data['truth'].squeeze().astype('int32')
        self.miss_matrix = io.loadmat(miss_path)['folds'][0, 1].astype("float16")

