#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Orthogonal and Oblique rotation.

# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# last update: 05/10/2022
"""

import numpy as np
from kavica.factor_analysis.rotation_function import normalize_numpy

__all__ = ['Rotation',
           'OrthogonalRotation',
           'ObliqueRotation'
           ]


class Rotation(object):
    """ Base factor rotation class.
    """

    def __init__(self, X=None):
        self.has_fitted = False
        self.origin_data = X
        self.rotated_factors = None
        self.factors_number = None
        self.features_number = None

    def fit(self, X):
        """ Check the input data and fit to the model.

        Args:
            X : array-like, shape = [n_features, p_factor] the training input samples.

        Returns
            self : object
        """
        if isinstance(X, np.ndarray):
            self._check_params(X)
            self.origin_data = X
            self.features_number = self.origin_data.shape[0]
            self.factors_number = self.origin_data.shape[1]
            self.has_fitted = True
            return self
        else:
            raise ValueError("Input {}: Factor pattern matrix must be ndarray [n_features, p_factor]".format(type(X)))

    def _check_params(self, x):
        # Todo: It is a function to check the type of the input data
        pass


class OrthogonalRotation(Rotation):
    ORTHOGONAL_METHODS = ['varimax', 'equimax', 'quartimax']

    def __init__(self, method='varimax', iteration=14, sv=1.0e-5):
        super(OrthogonalRotation, self).__init__()
        assert (method.lower() in self.ORTHOGONAL_METHODS), \
            "The method {} has not supported, either 'varimax', 'equimax' " \
            "or 'quartimax' should be selected.".format(method)
        self.method = method.lower()
        self.iteration = iteration
        self.sv = sv
        self.rotated_factors = None

    def _calculate_rotation_angle(self, x, y):
        """ Computes the rotation angle

        Args:
            x:
            y:

        Returns:
            A float indicates the angle
        """
        u = np.square(x) - np.square(y)
        v = 2 * x * y
        A = np.sum(u)
        B = np.sum(v)
        C = np.sum(np.square(u) - np.square(v))
        D = np.sum(u * v)
        if self.method == 'varimax':
            X = D - (2 * A * B) / self.features_number
            Y = C - (A ** 2 - B ** 2) / self.features_number
        elif self.method == 'equimax':
            X = D - (self.factors_number * A * B) / self.features_number
            Y = C - (self.factors_number * (A ** 2 - B ** 2)) / (2 * self.features_number)
        elif self.method == 'quartimax':
            X = D
            Y = C
        return np.arctan(X / Y) / 4

    def orthogonal_rotate(self):
        """ Computes rotated factors

        Returns:
            A np.ndarray
        """
        self.rotated_factors = normalize_numpy(self.origin_data, axis=1)
        for _ in range(self.iteration):
            for factorLoad1 in range(self.rotated_factors.shape[1]):
                for factorLoad2 in range(factorLoad1 + 1, self.rotated_factors.shape[1]):
                    np.sum(
                        np.square(self.rotated_factors[:, factorLoad1]) - np.square(
                            self.rotated_factors[:, factorLoad2]))
                    angle = self._calculate_rotation_angle(self.rotated_factors[:, factorLoad1],
                                                           self.rotated_factors[:, factorLoad2])
                    rotationMatrix = np.array([[np.cos(angle), -np.sin(angle)],
                                               [np.sin(angle), np.cos(angle)]])
                    self.rotated_factors[:, factorLoad1], self.rotated_factors[:, factorLoad2] = np.dot(
                        np.concatenate(([self.rotated_factors[:, factorLoad1]],
                                        [self.rotated_factors[:, factorLoad2]])).T,
                        rotationMatrix).T
        return self.rotated_factors


class ObliqueRotation(OrthogonalRotation):
    OBLIQUE_METHODS = ['promax']

    def __init__(self, method_orthogonal='promax', k=2):
        # TODO: it is needed to write multi constructor implemented for more flexibility.
        super(ObliqueRotation, self).__init__(method='varimax', iteration=14, sv=1.0e-5)
        assert (method_orthogonal.lower() in self.OBLIQUE_METHODS), \
            "The method {} has not supported, 'promax' should be selected.".format(method_orthogonal)
        self.method_orthogonal = method_orthogonal.lower()
        self.k = k

    def oblique_rotate(self):
        """ Computes rotated factors

        Returns:
            A np.ndarray
        """
        _rotated_matrix = self.orthogonal_rotate()
        power_matrix = _rotated_matrix.copy()
        for power_element in np.nditer(power_matrix, flags=['external_loop'], op_flags=['readwrite'], order='F'):
            _denominator = np.sqrt(np.sum(np.square(power_element)))
            power_element[...] = np.power((power_element / _denominator),
                                          (self.k + 1)) / np.power(_denominator,
                                                                   power_element)

        _rotated_matrix = _rotated_matrix.T  # Lambda matrix
        power_matrix = power_matrix.T  # P matrix
        load_matrix = np.dot(np.dot(np.linalg.pinv(np.dot(_rotated_matrix.T,
                                                          _rotated_matrix)),
                                    _rotated_matrix.T), power_matrix)  # L matrix

        normalized_load_matrix = load_matrix * (1 / np.sqrt(np.diagonal(np.dot(load_matrix.T,
                                                                               load_matrix))))  # Q matrix
        matrix_c = np.diag(1 / np.sqrt(np.diagonal(np.linalg.pinv(np.dot(normalized_load_matrix.T,
                                                                         normalized_load_matrix)))))  # C matrix
        _rotated_factor_pattern = np.dot(np.dot(_rotated_matrix, normalized_load_matrix),
                                         np.linalg.pinv(matrix_c))
        _factors_correlation_matrix = np.dot(np.dot(matrix_c, np.linalg.pinv(np.dot(load_matrix.T,
                                                                                    load_matrix))),
                                             matrix_c.T)
        _factor_structure = np.dot(_rotated_factor_pattern, _factors_correlation_matrix).T

        return _factor_structure


def _test_me():
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
    _test_me()
