#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Distance Measurement methods

Author: Kaveh Mahdavi <kavehmahdavi74@yahoo.com>
License: BSD 3 clause
last update: 30/09/2022
"""

import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances

__all__ = ['check_input_compatible',
           'euclidean_distance',
           'rbf_kernel',
           'mahalanobis',
           'mean_mahalanobis']


def check_input_compatible(x, y):
    """ Check X and Y array inputs compatible to compute the distance in the m-dimensional feature space

    Args:
        x (array-like, sparse matrix): shape (n_features,1)
        y: (array-like, sparse matrix): shape (n_features,1)

    Returns:
        safe_X : {array-like, sparse matrix}, shape (n_features,1)
        An array equal to X, guaranteed to be a numpy array.
        safe_Y : {array-like, sparse matrix}, shape (n_features,1)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.
    """
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        if x.shape[0] != y.shape[0]:
            raise ValueError("Incompatible dimensions: X.shape[1] == %d while Y.shape[1] == %d" % (x.shape[0],
                                                                                                   y.shape[0]))
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            raise ValueError("Incompatible dimensions:X.shape[1] == %d while Y.shape[1] == %d" % (len(x), len(y)))
    elif isinstance(x, pd.DataFrame) != isinstance(y, pd.DataFrame):
        raise ValueError("The data type is not compatible.")
    return x, y


def euclidean_distance(x, y):
    """ Computes the euclidean distance between two vectors

    Args:
        x (pandas series): indicate the fires vector
        y (pandas series):  indicate the second vector

    Returns:
        A float indicates the distance
    """
    x, y = check_input_compatible(x, y)
    return distance.euclidean(x, y)


def rbf_kernel(x=None, y=None, gamma=None, n_samples=2, pre_distance=None):
    """ Compute the rbf (gaussian) kernel between X and Y:
        K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

    Args:
    ----------
    X (ndarray): of shape (n_samples_X, n_features)
    Y (ndarray): of shape (n_samples_X, n_features)
    gamma (float): Technically, the gamma parameter is the inverse of the standard deviation of the RBF
                   kernel (Gaussian function), which is used as similarity measure between two points. Intuitively,
                   a small gamma value define a Gaussian function with a large variance.
                   The default is None If None, defaults to 1.0 / n_samples.
    n_sample (int): indicate the number of the datapoint that are toke to account to compute the rbf.
    pre_distance (float): indicate the pre-computed the distance between vectors
    Returns:
        A float Rbf_kernel value
    """
    # TODO: set the error condition and parameter checking
    if not any((gamma, n_samples)):
        raise ValueError("Both gamma and n_samples are None.")
    elif gamma is None:
        gamma = 1.0 / n_samples

    if pre_distance is None:
        x, y = check_input_compatible(x, y)
        _distance = distance.euclidean(x, y)
    else:
        _distance = pre_distance
    _distance **= 2
    _distance *= -gamma
    return math.exp(_distance)


def mkl_blas_euclidean_matrix(matrix_a, matrix_b=None):
    """ Matrix operation based method for generalized Euclidean distance matrix calculation.

    It is a function that calculate the distance matrix without intended loops (Intel kernel library).
    It is my version of pd.DataFrame(distance_matrix(matrix_a, matrix_a))
    """
    # Fixme: needed to take look at the original conceptual paper
    def __index_uniforming(df, typeis='int64'):
        df.columns = list(range(0, df.shape[0]))  # df.columns.astype(typeis)
        df.index = list(range(0, df.shape[0]))  # df.index.astype(typeis)
        return df

    if matrix_b is None:
        # Fixme: I do not know why the index and columns are changed to the string.
        # Dimension: matrix_a rows number X 1
        row_sum_of_product = np.power(matrix_a, 2).sum(axis=1).to_frame()
        ones = pd.DataFrame(1, index=np.arange(0, len(row_sum_of_product)), columns=np.arange(1))

        # X^2 ^ Y^2 Dimension: matrix_a rows number X  matrix_a rows number

        p1 = ones.dot(row_sum_of_product.T)

        p1 = __index_uniforming(p1)

        # XY Dimension: matrix_a rows number X  matrix_a rows number
        p3 = matrix_a.dot(matrix_a.T)
        p3 = __index_uniforming(p3)

        D = p1 + p1.T - 2 * p3

    else:
        # Dimension: matrix_a rows number X 1
        row_sum_of_product = np.power(matrix_a, 2).sum(axis=1).to_frame()
        ones = pd.DataFrame(1, index=np.arange(0, len(row_sum_of_product)), columns=np.arange(1))
        # X^2 ^ Y^2 Dimension: matrix_a rows number X  matrix_a rows number
        p1 = ones.dot(row_sum_of_product.T)
        p1 = __index_uniforming(p1)

        # Dimension: matrix_b rows number X 1
        row_sum_of_product = np.power(matrix_b, 2).sum(axis=1).to_frame()
        ones = pd.DataFrame(1, index=np.arange(0, len(row_sum_of_product)), columns=np.arange(1))
        # X^2 ^ Y^2 Dimension: matrix_a rows number X  matrix_a rows number
        p2 = ones.dot(row_sum_of_product.T)
        p2 = __index_uniforming(p2)

        # XY Dimension: matrix_a rows number X  matrix_a rows number
        p3 = matrix_a.dot(matrix_b.T)
        p3 = __index_uniforming(p3)

        D = p1 + p2.T - 2 * p3

    return D.apply(np.sqrt)


def mahalanobis(x=None, data=None, cov=None):
    """ The function to calculate Mahalanobis distance

    Args:
        x:
        data:
        cov:

    Returns:

    Note:
        https://www.statology.org/mahalanobis-distance-python/
    """
    x_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    return mahal.diagonal()


def mean_mahalanobis(x=None, data=None, cov=None):
    """ The function to calculate avrage Mahalanobis distance

    Args:
        x:
        data:
        cov:

    Returns:

    """
    _amd = mahalanobis(x, data, cov)
    return np.mean(_amd)


'''
x = [-1.9623440818022793, -1.3062248054892618, -1.2638622312540577, -1.3902130439944476, -0.44232586846469146,
     -1.3591000865553797, -1.125, 0.0, -0.5]
y = [0.3905945376664751, -1.1377332902046104, -1.178692549298139, -1.0443889036973213, -0.44232586846469146,
     -1.0783236565400163, -1.125, 0.0, -0.5]
print(rbf_kernel(x, y, 0.5031152949374527))

data = np.array([(1, 1, 1, 1, 1, 1, 1),
                 (2, 2, 2, 2, 1, 2, 2),
                 (2, 2, 45, 23, 24, 13, 16),
                 (3, 12, 0, 9, 5, 20, 89)])
data1 = np.array([("ind", "F1", "F2", "F3", "F4", "F5", "F6"),
                  (1, 1, 1, 1, 1, 1, 1),
                  (2, 2, 2, 2, 2, 2, 2),
                  (3, 4, 45, 23, 24, 19, 16),
                  (4, 2, 44, 23, 22, 13, 11),
                  (5, 2, 4, 3, 2, 1, 1),
                  (6, 1, 1, 1, 1, 1, 1),
                  (7, 2, 2, 2, 2, 2, 2),
                  (8, 2, 45, 23, 24, 13, 16),
                  (9, 12, 0, 9, 5, 20, 89)])

headers = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
index = [1, 2, 3, 4]
df = pd.DataFrame(data, columns=headers, index=index)

x = pd.to_numeric(df.loc[1])
y = pd.to_numeric(df.loc[2])

print(rbf_kernel(x, y))
'''