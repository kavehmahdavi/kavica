#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two factor rotation methods, Promax and Varimax
"""
# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# last update: 17/12/2018

import numpy as np

__all__ = ['_normalize_numpy',
           'promax',
           'varimax'
           ]


def _normalize_numpy(x, axis=1):
    """ Dividing each cell by the square root of the sum of squares in that row(axis=1)/column(axis=0) """

    denominators = 1 / np.sqrt(np.sum(np.square(x), axis=axis))
    if axis == 0:
        return x * denominators
    elif axis == 1:
        return x * denominators[:, np.newaxis]
    else:
        raise ValueError("The axis value is {}. It has to be 0/1.".format(axis))


def promax(x, iteration=14, k=2):
    orthogonalRotatedMatrix = varimax(x, iteration=iteration)
    powerMatrix = orthogonalRotatedMatrix.copy()
    for powerElement in np.nditer(powerMatrix, flags=['external_loop'], op_flags=['readwrite'], order='F'):
        denomarator = np.sqrt(np.sum(np.square(powerElement)))
        powerElement[...] = np.power((powerElement / denomarator),
                                     (k + 1)) / np.power(denomarator,
                                                         powerElement)

    orthogonalRotatedMatrix = orthogonalRotatedMatrix.T  # Lambda matrix
    powerMatrix = powerMatrix.T  # P matrix
    loadMatrix = np.dot(np.dot(np.linalg.pinv(np.dot(orthogonalRotatedMatrix.T,
                                                     orthogonalRotatedMatrix)),
                               orthogonalRotatedMatrix.T),
                        powerMatrix)  # L matrix

    normalizedLoadMatrix = loadMatrix * (1 / np.sqrt(np.diagonal(np.dot(loadMatrix.T,
                                                                        loadMatrix))))  # Q matrix
    matrixC = np.diag(1 / np.sqrt(np.diagonal(np.linalg.pinv(np.dot(normalizedLoadMatrix.T,
                                                                    normalizedLoadMatrix)))))  # C matrix
    promaxRotatedFactorPattern = np.dot(np.dot(orthogonalRotatedMatrix,
                                               normalizedLoadMatrix),
                                        np.linalg.pinv(matrixC))
    promaxFactorsCorrelationMatrix = np.dot(np.dot(matrixC,
                                                   np.linalg.pinv(np.dot(loadMatrix.T,
                                                                         loadMatrix))),
                                            matrixC.T)
    promaxFactorStructure = np.dot(promaxRotatedFactorPattern, promaxFactorsCorrelationMatrix).T

    return promaxFactorStructure


def varimax(x, iteration=14):
    """ http://www.real-statistics.com/linear-algebra-matrix-topics/varimax/"""

    # TODO: set more intelligent angle evaluator
    # parameter: x np.array(m_features,c_factors)
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

    x = _normalize_numpy(x, axis=1)
    for _ in range(iteration):
        for factorLoad1 in range(x.shape[1]):
            for factorLoad2 in range(factorLoad1 + 1, x.shape[1]):
                np.sum(np.square(x[:, factorLoad1]) - np.square(x[:, factorLoad2]))
                angle = _calculate_rotation_angle(x[:, factorLoad1], x[:, factorLoad2])
                rotationMatrix = np.array([[np.cos(angle), -np.sin(angle)],
                                           [np.sin(angle), np.cos(angle)]])
                x[:, factorLoad1], x[:, factorLoad2] = np.dot(np.concatenate(([x[:, factorLoad1]],
                                                                              [x[:, factorLoad2]])).T, rotationMatrix).T
    return x
