from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import random
import torch


def data_preprocess(X: list, normalize: str = None) -> list:
    if normalize:
        # print(normalize)
        scaler = eval(normalize)()
        X = [scaler.fit_transform(x) for x in X]
    return X


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

    miss_matrix = torch.from_numpy(miss_matrix.astype('float32'))
    miss_matrixs = [miss_matrix.T[i].unsqueeze(1).numpy() for i in range(n_view)]
    return np.stack(miss_matrixs)   # (v, n, 1)