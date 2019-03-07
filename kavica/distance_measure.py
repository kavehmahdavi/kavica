# -*- coding: utf-8 -*-
""" Distance Measurement methods"""

# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause

import math
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix, distance

__all__ = ['euclideanDistance',
           'rbf_kernel']


def check_input(X, Y):
    """ Check X and Y array inputs

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_features,1)
    Y : {array-like, sparse matrix}, shape (n_features,1)

    Returns
    -------
    safe_X : {array-like, sparse matrix}, shape (n_features,1)
        An array equal to X, guaranteed to be a numpy array.
    safe_Y : {array-like, sparse matrix}, shape (n_features,1)
        An array equal to Y if Y was not None, guaranteed to be a numpy array.
        If Y was None, safe_Y will be a pointer to X.
    """
    if isinstance(X, pd.DataFrame) and isinstance(Y, pd.DataFrame):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Incompatible dimension for X and Y matrices: "
                             "X.shape[1] == %d while Y.shape[1] == %d" % (
                                 X.shape[0], Y.shape[0]))
    elif isinstance(X, list) and isinstance(Y, list):
        if len(X) != len(Y):
            raise ValueError("Incompatible dimension for X and Y matrices: "
                             "X.shape[1] == %d while Y.shape[1] == %d" % (
                                 len(X), len(Y)))
    elif isinstance(X, pd.DataFrame) != isinstance(Y, pd.DataFrame):
        raise ValueError("The data type is not compatible.")

    return X, Y


def euclideanDistance(X, Y):
    X, Y = check_input(X, Y)
    return distance.euclidean(X, Y)


def rbf_kernel(X=None, Y=None, gamma=None, n_samples=2, pre_distance=None):
    """
    Compute the rbf (gaussian) kernel between X and Y::
        K(x, y) = exp(-gamma ||x-y||^2)
    for each pair of rows x in X and y in Y.
    Read more in the :ref:`User Guide <rbf_kernel>`.
    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
    Y : array of shape (n_samples_Y, n_features)
    gamma : float, default None
        If None, defaults to 1.0 /
    Returns
    -------
    Rbf_kernel value : float
    """
    # TODO: set the error condition and parameter checking
    if not any((gamma, n_samples)):
        raise ValueError("Both gamma and n_samples are None.")

    if gamma is None:
        gamma = 1.0 / n_samples

    if pre_distance is None:
        X, Y = check_input(X, Y)
        euclidean = distance.euclidean(X, Y)
    else:
        euclidean = pre_distance
    euclidean **= 2
    euclidean *= -gamma
    rbf = math.exp(euclidean)  # exponential in-place
    return rbf


def MKL_BLAS_Euclidean_matrix(matrix_a, matrix_b=None):
    """
    Matrix operation based method for generalized Euclidean distance matrix calculation.
    It is a function that calculate the distance matrix without intended loops (Intel kernel library).
    It is my version of pd.DataFrame(distance_matrix(matrix_a, matrix_a))
    """

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


'''
x = [-1.9623440818022793, -1.3062248054892618, -1.2638622312540577, -1.3902130439944476, -0.44232586846469146,
     -1.3591000865553797, -1.125, 0.0, -0.5]
y = [0.3905945376664751, -1.1377332902046104, -1.178692549298139, -1.0443889036973213, -0.44232586846469146,
     -1.0783236565400163, -1.125, 0.0, -0.5]
print(rbf_kernel(x,y,0.5031152949374527))



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
