# vim: expandtab:ts=4:sw=4
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import distance

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.
    计算a和b之间的成对平方距离
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
        a是NxM矩阵，N个M维度样本
    b : array_like
        An LxM matrix of L samples of dimensionality M.
        b是LxM矩阵，L个M维度样本

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
        返回一个len(a) x len(b)的矩阵，元素（i,j）包含a[i]与b[j]之间的距离

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    # 对a2,b2的每一行求和，即对每个样本的所有维度求平方和，a2,b2是N维,L维的向量
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    # a2[:,None]是（N，1）的向量
    # b2[None,:]是（1，L）的向量
    # 得到的r2是一个（N，L）矩阵
    r2 = np.clip(r2, 0., float(np.inf))
    # 将距离小于0的值全部置为0
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.
    计算点‘a’和‘b’之间的pair-wise距离
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.
        If True，假设a和b是单位长度矢量，否则将被明确归一化为1

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)
        b = np.asarray(b) / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-6)
        # 对每一行（每个样本）正则化
    # np.dot(a, b.T)表示了两个向量间相似程度，就像是attention， 取值在(0, 1)之间，数值越大相似度越高
    return 0.5 - np.dot(a, b.T)  # (num_a, num_b)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).
    最近邻距离度量（欧几里得）的辅助函数。
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).样本点（N行矩阵）
    y : ndarray
        A matrix of M row-vectors (query points).查询点（M行矩阵）

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.
        M行的矩阵，包含y中每个元素对于样本x的最小欧几里得距离

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).
    最近邻距离度量（余弦）的辅助函数。
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)  # (num_x, num_y)
    return distances.min(axis=0)    # axis=0 means 取每一列的最小值, (num_y, )


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.
    最近邻距离尺度，对于每个目标，返回目前为止对所有已观察到的样本中最近的距离

    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine". 欧拉距离或余弦距离
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
        匹配门限，更大距离的samples被认为是无效的匹配
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.
        If not None, 将每个类别的样本数固定在最多这个值，当目标数足够时移除最老的样本（每一类个体的数目上限）

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.
        samples是一个字典，从目标ID映射到目前为止观测到的sample列表
        (ID ： 样本）

    """

    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    # def partial_fit(self, features, targets, active_targets):
    #     """Update the distance metric with new data.
    #     用新数据更新距离尺度
    #     Parameters
    #     ----------
    #     features : ndarray
    #         An NxM matrix of N features of dimensionality M.
    #         NxM矩阵，N个特征，每个特征有M个维度
    #     targets : ndarray
    #         An integer array of associated target identities.
    #         关联目标id的整数数组
    #     active_targets : List[int]
    #         A list of targets that are currently present in the scene.
    #         当前场景最近出现的目标列表
    #     """
    #     for feature, target in zip(features, targets):
    #         # zip 将对应元素打包成元组的列表
    #         self.samples.setdefault(target, []).append(feature)
    #         # setdefault 相当于get，查找sample中（这个字典）是否含有名为target的key，如果没有，将其添加为键，并把feature作为值
    #         if self.budget is not None:
    #             self.samples[target] = self.samples[target][-self.budget:]
    #     self.samples = {k: self.samples[k] for k in active_targets}
    #     # 获得一个所有target的字典，然后更新这个字典，获得只含有当前场景出现的目标的字典
    #     # feature是特征向量，target是ID

    def distance(self, tracks_feature, detections_feature):
        """Compute distance between features and targets.
        计算features和targets之间的距离
        Parameters
        ----------
        features : ndarray, tracks
            An NxM matrix of N features of dimensionality M.
        targets : List[int],
            A list of targets to match the given `features` against.
            一个用于匹配给定features的target列表
        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
            返回成本矩阵，维度为len(target) x len(feature)

        """
        cost_matrix = np.zeros((len(tracks_feature), len(detections_feature)))
        feature_dim = tracks_feature.shape[1]
        for i, target in enumerate(tracks_feature):
            # enumerate 生产一个索引和原元素序列，i是s索引，target是targets内的元素
            track_feature = tracks_feature[i].reshape(1, feature_dim)
            cost_matrix[i, :] = self._metric(track_feature, detections_feature)
        return cost_matrix
