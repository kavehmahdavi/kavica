#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orthogonal and Oblique rotation.
"""
# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# last update: 17/12/2018

import numpy as np

__all__ = ['_normalize_numpy',
           'Rotatin',
           'OrthogonalRotation',
           'ObliqueRotation'
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


class Rotatin(object):
    """Initialize the factor rotation."""

    def __init__(self, X=None):
        self.hasFitted = False
        self.originData = X
        self.rotatedFactors = None
        self.numberOfFactors = None
        self.numberOfFeatures = None

    def fit(self, X):
        """ Check the input data and fit to the model.
        Parameters
        ----------
        X : array-like, shape = [n_features, p_factor]
            The training input samples.

        Returns
        -------
        self : object
        """
        if isinstance(X, np.ndarray):
            self._check_params(X)
            self.originData = X
            self.numberOfFeatures = self.originData.shape[0]
            self.numberOfFactors = self.originData.shape[1]
            self.hasFitted = True
            return self
        else:
            raise ValueError(
                "Input {}: The factor pattern matrix has to be np.ndarray [n_features, p_factor]".format(type(X)))

    def _check_params(self, X):
        pass


class OrthogonalRotation(Rotatin):
    methods = ['varimax',
               'equimax',
               'quartimax']

    def __init__(self, method='varimax', iteration=14, sv=1.0e-5):
        super(OrthogonalRotation, self).__init__()
        assert (method.lower() in self.methods), \
            "The method {} has not supported, either 'varimax', 'equimax' " \
            "or 'quartimax' should be selected.".format(method)
        self.method = method.lower()
        self.iteration = iteration
        self.sv = sv

    def _calculate_rotation_angle(self, x, y):
        u = np.square(x) - np.square(y)
        v = 2 * x * y
        A = np.sum(u)
        B = np.sum(v)
        C = np.sum(np.square(u) - np.square(v))
        D = np.sum(u * v)
        if self.method == 'varimax':
            X = D - (2 * A * B) / self.numberOfFeatures
            Y = C - (A ** 2 - B ** 2) / self.numberOfFeatures
        elif self.method == 'equimax':
            X = D - (self.numberOfFactors * A * B) / self.numberOfFeatures
            Y = C - (self.numberOfFactors * (A ** 2 - B ** 2)) / (2 * self.numberOfFeatures)
        elif self.method == 'quartimax':
            X = D
            Y = C
        return np.arctan(X / Y) / 4

    def orthogonal_rotate(self):
        self.rotatedFactors = _normalize_numpy(self.originData, axis=1)
        for _ in range(self.iteration):
            for factorLoad1 in range(self.rotatedFactors.shape[1]):
                for factorLoad2 in range(factorLoad1 + 1, self.rotatedFactors.shape[1]):
                    np.sum(
                        np.square(self.rotatedFactors[:, factorLoad1]) - np.square(self.rotatedFactors[:, factorLoad2]))
                    angle = self._calculate_rotation_angle(self.rotatedFactors[:, factorLoad1],
                                                           self.rotatedFactors[:, factorLoad2])
                    rotationMatrix = np.array([[np.cos(angle), -np.sin(angle)],
                                               [np.sin(angle), np.cos(angle)]])
                    self.rotatedFactors[:, factorLoad1], self.rotatedFactors[:, factorLoad2] = np.dot(
                        np.concatenate(([self.rotatedFactors[:, factorLoad1]],
                                        [self.rotatedFactors[:, factorLoad2]])).T,
                        rotationMatrix).T
        return self.rotatedFactors


class ObliqueRotation(OrthogonalRotation):
    orthogonalMethods = ['promax']

    def __init__(self, methodOrthogonal='promax', k=2):
        # TODO: it is inherit from the Rotation class.
        # TODO: it is needed to write multi constructor implemented for more flexibility.
        super(ObliqueRotation, self).__init__(method='varimax', iteration=14, sv=1.0e-5)
        assert (methodOrthogonal.lower() in self.orthogonalMethods), \
            "The method {} has not supported,  " \
            " 'promax' should be selected.".format(methodOrthogonal)
        self.methodOrthogonal = methodOrthogonal.lower()
        self.k = k

    def oblique_rotate(self):
        orthogonalRotatedMatrix = self.orthogonal_rotate()
        powerMatrix = orthogonalRotatedMatrix.copy()
        for powerElement in np.nditer(powerMatrix, flags=['external_loop'], op_flags=['readwrite'], order='F'):
            denomarator = np.sqrt(np.sum(np.square(powerElement)))
            powerElement[...] = np.power((powerElement / denomarator),
                                         (self.k + 1)) / np.power(denomarator,
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


def main():
    a = np.array([
        (0.758, 0.413, 0.001164),
        (0.693, 0.489, -0.199),
        (0.362, 0.656, -0.204),
        (0.826, 0.06589, 0.235),
        (0.540, -0.510, 0.441),
        (0.654, -0.335, 0.507),
        (-0.349, 0.539, 0.669),
        (-0.580, 0.450, 0.551)])
    tt = OrthogonalRotation(method='varimax')
    tt.fit(a)
    dd = tt.orthogonal_rotate()
    print(dd)
    tt2 = ObliqueRotation('promax')
    tt2.fit(a)
    dd2 = tt2.oblique_rotate()
    print(dd2)


if __name__ == '__main__':
    main()
