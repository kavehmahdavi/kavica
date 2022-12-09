#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Factor rotation methods, Promax and Varimax

# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# last update: 05/10/2022
"""
import numpy as np

__all__ = ['normalize_numpy',
           'promax',
           'varimax', ]


def normalize_numpy(x, axis=1):
    """ Normalizes a numpy

    Dividing each cell by the square root of the sum of squares in that row(axis=1)/column(axis=0)

    Args:
        x (numpy): Includes the input data
        axis (int in [0,1]): Indicates the normalization is aligned with the row(axis=1) or column(axis=0)

    Returns:

    """

    denominators = 1 / np.sqrt(np.sum(np.square(x), axis=axis))
    if axis == 0:
        return x * denominators
    elif axis == 1:
        return x * denominators[:, np.newaxis]
    else:
        raise ValueError("The axis value is {}. It has to be 0/1.".format(axis))


def promax(x, iteration=14, k=2):
    """ Computes the Promxa factor rotation

    An oblique rotation, which allows factors to be correlated. This rotation can be calculated more quickly than a
    direct oblimin rotation, so it is useful for large datasets.

    Promax rotation is performed in the following steps:
        * Determine varimax rotated patterns :math:`V`.
        * Construct a rotation target matrix :math:`|V_{ij}|^k/V_{ij}`
        * Perform procrustes rotation towards the target to obtain T
        * Determine the patterns

    First, varimax rotation a target matrix :math:`H` is determined with orthogonal varimax rotation.
    Then, oblique target rotation is performed towards the target.

    Args:
        x (np.array(m_features,c_factors)): Includes the input data
        iteration (int): Indicates the number of the iteration
        k (float): should be positive

    Returns:
        A np.array(m_features,c_factors))

    See:
        - Browne (2001) - An overview of analytic rotation in exploratory factor analysis

        - Navarra, Simoncini (2010) - A guide to empirical orthogonal functions for climate data analysis
    """
    orthogonal_rotated_matrix = varimax(x, iteration=iteration)
    power_matrix = orthogonal_rotated_matrix.copy()
    for _power_item in np.nditer(power_matrix, flags=['external_loop'], op_flags=['readwrite'], order='F'):
        _denominator = np.sqrt(np.sum(np.square(_power_item)))
        _power_item[...] = np.power((_power_item / _denominator),
                                    (k + 1)) / np.power(_denominator,
                                                        _power_item)

    orthogonal_rotated_matrix = orthogonal_rotated_matrix.T  # Lambda matrix
    power_matrix = power_matrix.T  # P matrix
    load_matrix = np.dot(np.dot(np.linalg.pinv(np.dot(orthogonal_rotated_matrix.T,
                                                      orthogonal_rotated_matrix)),
                                orthogonal_rotated_matrix.T), power_matrix)  # L matrix

    normalized_load_matrix = load_matrix * (1 / np.sqrt(np.diagonal(np.dot(load_matrix.T,
                                                                           load_matrix))))  # Q matrix
    matrix_c = np.diag(1 / np.sqrt(np.diagonal(np.linalg.pinv(np.dot(normalized_load_matrix.T,
                                                                     normalized_load_matrix)))))  # C matrix
    promax_rotated_factor_pattern = np.dot(np.dot(orthogonal_rotated_matrix, normalized_load_matrix),
                                           np.linalg.pinv(matrix_c))
    promax_factors_correlation_matrix = np.dot(np.dot(matrix_c,
                                                      np.linalg.pinv(np.dot(load_matrix.T,
                                                                            load_matrix))),
                                               matrix_c.T)
    promax_factor_structure = np.dot(promax_rotated_factor_pattern, promax_factors_correlation_matrix).T
    return promax_factor_structure


def varimax(x, iteration=14):
    """ Return rotated components by Varimax

    x (np.array(m_features,c_factors)): Includes the input data
    iteration (int): Indicates the number of the iteration

    http://www.real-statistics.com/linear-algebra-matrix-topics/varimax/

    Returns:
        A np.array(m_features,c_factors))
    """

    # TODO: set more intelligent angle evaluator
    def _calculate_rotation_angle(x, y):
        u = np.square(x) - np.square(y)
        v = 2 * x * y
        A = np.sum(u)
        B = np.sum(v)
        C = np.sum(np.square(u) - np.square(v))
        D = np.sum(u * v)
        X = D - (2 * A * B) / len(x)
        Y = C - (A ** 2 - B ** 2) / len(x)
        return np.arctan(X / Y) / 4

    x = normalize_numpy(x, axis=1)
    for _ in range(iteration):
        for factorLoad1 in range(x.shape[1]):
            for factorLoad2 in range(factorLoad1 + 1, x.shape[1]):
                np.sum(np.square(x[:, factorLoad1]) - np.square(x[:, factorLoad2]))
                angle = _calculate_rotation_angle(x[:, factorLoad1], x[:, factorLoad2])
                rotationMatrix = np.array([[np.cos(angle), -np.sin(angle)],
                                           [np.sin(angle), np.cos(angle)]])
                x[:, factorLoad1], x[:, factorLoad2] = np.dot(np.concatenate(([x[:, factorLoad1]],
                                                                              [x[:, factorLoad2]])).T,
                                                              rotationMatrix).T
    return x
