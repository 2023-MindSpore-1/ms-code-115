from munkres import Munkres
import numpy as np


def align( y_pred: np.ndarray, y_true: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    计算评估结果,acc,nmi,ari,f1
    """
    y_true = y_true - y_true.min()
    l1 = list(set(y_true))
    l2 = list(set(y_pred))

    cost = np.zeros((n_clusters, n_clusters), dtype=int)

    for i in l1:

        indexs = [i1 for i1 in range(len(y_true)) if y_true[i1] == i]  # 记录i类别的索引
        for j in l2:
            c = [j1 for j1 in indexs if y_pred[j1] == j]
            cost[(i, j)] = len(c)

    m = Munkres()
    cost = -cost
    indexs = m.compute(cost)  # 记录最佳match
    new_x = np.zeros_like(y_pred)
    for i in indexs:
        end = i[1]
        y_pred_index = [i1 for i1 in range(len(y_pred)) if y_pred[i1] == end]
        new_x[y_pred_index] = i[0]

    return new_x
    
