#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Polygonal Cage

These module is a computational geometry to map the position of a vector from a deformed grid into a
Regular Rectangular Grid (RRG)

Note:
    https://www.youtube.com/watch?v=qBJCCe81OCg

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
First update: 01/09/2021
Last update: 11/11/2022
"""

import random
import math
import sys
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from itertools import count
from numpy import dot
from sklearn.metrics.pairwise import cosine_similarity
from sympy.solvers import solve
from sympy import Symbol
from numpy.linalg import norm
from sympy.plotting import plot as syplot
from kavica.utils._bcolors import BColors

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize, linewidth=200)
random.seed(4)
np.random.seed(4)
plt.rcParams.update({'font.size': 16})


class PolygonCage(object):
    """ Class
    A class for computing the relative position of a point in a grid after non_affine grid deformation

    It is for the uniform RRG and the cells are unit cell.

    Parameters
        target_id (int): The data point id
        anchor (int): The lattice position of anchor point "Cage lattice center"
        target_vector (np.array): The target feature vector shape (1, n_features)
        anchor_vector (np.array): The anchor feature vector shape (1, n_features)

    Class Attribute:
        boundary_sides (dict): Includes the lateral lattice points {left:[], top:[], right:[], bottom:[]}
        x_dim (int): The grid side size when the grid is shaped in (x_dim * x_dim)

    Attributes:
        self.target_id (int): The data point id
        self.anchor (int): The lattice position of anchor point
        self.target_vector (np.array): The target feature vector shape (1, n_features)
        self.anchor_vector (np.array): The anchor feature vector shape (1, n_features)
        self.lattice_neighbour_vectors (np.array): The neighbour feature vector shape (n_neighbour, n_features)
        self.lattice_neighbour_points (np.array): The lattice position of neighbour (1, n_neighbour)
        self.target_lattice_distances (np.array): The distance of target point to the neighbours (1, n_neighbour)
        self.target_anchor_distances (int): The distance of target point to the anchor point
        self.target_position (np.array): The vectors between anchor and target (1,n_features)
        self.lattice_neighbour_position (np.array): The vectors between anchor and neighbours (n_neighbour ,n_features)
        self.target_angles (dict{np.array}): The angle between neighbours and target position (rad & deg)
        self.lattices_angels (dict{np.array}): The angle between two adjacent neighbours position clockwise (rad & deg)
        self.target_cosine_similarity (np.array): The cosine similarity between neighbours and target position
        self.control_points (np.array): The two compulsory lattice points to indicates the associated cage cell.
        self.lattice_neighborhood_size (int): The number of the lattice topological direct neighbour of anchor point
        self.cage_type (str): The relational location of the cage in the grid {left, top, right, bottom, interior}
    """
    boundary_sides = None
    x_dim = None

    def __init__(self, target_id, anchor, target_vector, anchor_vector, lattice_neighbour_points,
                 lattice_diagonal_points):
        self.target_id = target_id
        self.anchor = anchor
        self.target_vector = target_vector
        self.anchor_vector = anchor_vector
        self.lattice_neighbour_vectors = None
        self.lattice_neighbour_points = None
        self.lattice_diagonal_vectors = None
        self.lattice_diagonal_points = None
        self.target_lattice_distances = None
        self.target_anchor_distances = None
        self.target_position = None
        self.lattice_neighbour_position = None
        self.target_angles = None
        self.lattices_angels = None
        self.target_cosine_similarity = None
        self.control_points = None
        self.lattice_neighborhood_size = None
        self.cage_type = None
        self.cell_coordinate = None
        self.lw = (1, 1)
        self.predicted_position = None
        self.predicted_offset = None
        self.roots = None
        self.root = None
        self.root_type = None
        self.compacted = False
        self.lattice_neighbour_points = lattice_neighbour_points
        self.lattice_diagonal_points = lattice_diagonal_points

    def sort_neighbours(self, inplace=True):
        """ Sort the neighbours list

        Args:
            inplace (boolean): If True, it does the sort inplace

        Returns:
            A sorted pandas
        """
        if inplace:
            self.lattice_neighbour_points = np.sort(self.lattice_neighbour_points)
            return self.lattice_neighbour_points
        else:
            return np.sort(self.lattice_neighbour_points)

    @property
    def get_neighborhood_size(self):
        """Compute the number of alternative control points in the cage

        Returns:
            An int the number of the neighbours
        """
        return len(self.lattice_neighbour_points)

    def sort_clockwise(self):
        """Sorts the lattice neighbour clockwise

        Returns:
            A list of the sorted lattices
        """
        self.sort_neighbours(inplace=True)
        lattice_neighborhood_size = self.get_neighborhood_size

        if lattice_neighborhood_size == 2:
            clockwise_neighbour_index = [0, 1]
            # lattice_control_points = np.argsort(self.lattice_neighbour_points)
        else:
            if lattice_neighborhood_size == 4:
                clockwise_neighbour_index = [0, 2, 3, 1]
            elif lattice_neighborhood_size == 3:
                if self.anchor in self.boundary_sides.get('left'):
                    clockwise_neighbour_index = [1, 2, 0]
                elif self.anchor in self.boundary_sides.get('top'):
                    clockwise_neighbour_index = [2, 1, 0]
                elif self.anchor in self.boundary_sides.get('right'):
                    clockwise_neighbour_index = [1, 0, 2]
                elif self.anchor in self.boundary_sides.get('bottom'):
                    clockwise_neighbour_index = [0, 1, 2]
                else:
                    raise ValueError("The neighborhood is not correct.")
        self.lattice_neighbour_points = self.lattice_neighbour_points[clockwise_neighbour_index]
        return clockwise_neighbour_index

    def get_lattice_distances(self):
        """Computes the distance between the target point and the neighbours

        Returns:
            A distance matrix
        """
        self.target_lattice_distances = np.apply_along_axis(euclidean, 1,
                                                            self.lattice_neighbour_vectors,
                                                            np.array(self.target_vector))
        return self.target_lattice_distances

    def get_anchor_distance(self):
        """Computes the distance between the target point and the anchor lattice"""
        self.target_anchor_distances = euclidean(np.array(self.anchor_vector), np.array(self.target_vector))
        return self.target_anchor_distances

    def get_target_position(self):
        """ Computes the vector between target and anchor"""
        self.target_position = np.array(self.target_vector) - np.array(self.anchor_vector)
        return self.target_position

    def get_lattice_neighbour_position(self):
        """ Computes the vector between anchor and lattice neighbours"""
        self.lattice_neighbour_position = self.lattice_neighbour_vectors - np.array(self.anchor_vector)
        return self.lattice_neighbour_position

    @staticmethod
    def angle_between_vectors(a, b, deg=False):
        """ Computes the angle between two vector

        The value is between [0,1]

        Args:
            a (ndarray): Indicates the first vector
            b (ndarray): Indicates the second vector
            deg (boolean): If True, it will return the angel as dgree else radian

        Returns:
            A float indicates the angel (degree/radian) between two vectors.
        """
        arccos_input = dot(a, b) / norm(a) / norm(b)
        arccos_input = 1.0 if arccos_input > 1.0 else arccos_input
        arccos_input = -1.0 if arccos_input < -1.0 else arccos_input
        if not deg:
            return arccos_input
        else:
            return math.degrees(np.arccos(arccos_input))

    def get_target_angels(self):
        """Computes the angle between target position and lattices neighbours"""
        degree_angles = np.apply_along_axis(self.angle_between_vectors, 1,
                                            self.lattice_neighbour_position,
                                            self.target_position, True)
        radians_angles = np.apply_along_axis(self.angle_between_vectors, 1,
                                             self.lattice_neighbour_position,
                                             self.target_position, False)

        self.target_angles = {"degree": degree_angles, "radian": radians_angles}
        return self.target_angles

    def get_lattices_angels(self):
        """Computes the angle between peers of lattices neighbour position vectors clockwise"""
        _index = list(range(self.get_neighborhood_size))
        _degrees = []
        _radians = []
        for left_norm, right_norm in zip(_index, np.roll(_index, -1)):
            _degrees.append(self.angle_between_vectors(self.lattice_neighbour_position[left_norm],
                                                       self.lattice_neighbour_position[right_norm], True))
            _radians.append(self.angle_between_vectors(self.lattice_neighbour_position[left_norm],
                                                       self.lattice_neighbour_position[right_norm], False))
        self.lattices_angels = {"degree": _degrees, "radian": _radians}
        return self.lattices_angels

    def get_similarity(self):
        """Computes the angel based similarities between target position and lattices neighbours"""
        self.target_cosine_similarity = cosine_similarity([self.target_position],
                                                          self.lattice_neighbour_position)
        return self.target_cosine_similarity

    # Todo: the control_points=None needs a solution.
    def get_corresponding_cell(self):
        """Computes the corresponding grid cell that data point is located after coordinate transformation

        First of all, we compute the lattice neighbour point as norm. It's position vector most similar one to the
        target position vector. As result, the target cage cell is just limited to two alternatives.
        Then we make decision between two. If the target position is located between two sided position vectors,
        we figure out the target point is located in this cell.
        In order to do it, we compute the angle between two sided position vectors (one is norm position).
        So, we compare it with angle between norm position and target position. In case of selected cell the angle
        norm_target is smaller than sided one.

        Returns:
            A list of the control points
        """
        self.lattice_neighborhood_size = self.get_neighborhood_size
        if self.lattice_neighborhood_size == 2:
            lattice_control_args = [0, 1]
            self.control_points = np.array(self.lattice_neighbour_points[lattice_control_args])
        else:
            if self.lattice_neighborhood_size == 4:
                # Inter grid cage: 4 alternative cell is reduced to 2
                norm_lattice_arg = np.argmax(self.target_angles.get("radian"))
                left_adjacent_lattice_arg = norm_lattice_arg - 1 if norm_lattice_arg - 1 >= 0 else 3
                right_adjacent_lattice_arg = norm_lattice_arg + 1 if norm_lattice_arg + 1 <= 3 else 0
                self.cage_type = 'Interior'
            elif self.lattice_neighborhood_size == 3:
                # Border grid cage: 2 alternative cell
                left_adjacent_lattice_arg, norm_lattice_arg, right_adjacent_lattice_arg = 0, 1, 2
                if self.anchor in self.boundary_sides.get('left'):
                    self.cage_type = 'Left'
                elif self.anchor in self.boundary_sides.get('top'):
                    self.cage_type = 'Top'
                elif self.anchor in self.boundary_sides.get('right'):
                    self.cage_type = 'Right'
                elif self.anchor in self.boundary_sides.get('bottom'):
                    self.cage_type = 'Bottom'
                else:
                    raise ValueError("The neighborhood is not correct.")
            decision_table = np.array([[self.lattices_angels.get("radian")[left_adjacent_lattice_arg],
                                        self.target_angles.get("radian")[left_adjacent_lattice_arg]],
                                       [self.lattices_angels.get("radian")[norm_lattice_arg],
                                        self.target_angles.get("radian")[right_adjacent_lattice_arg]]])
            # Cell selection
            decision = [False, False]
            for index, alternative_cell in enumerate(decision_table):
                if alternative_cell[1] > alternative_cell[0]:
                    decision[index] = True
            if np.logical_xor(decision[0], decision[1]):
                if np.where(decision)[0][0] == 0:
                    lattice_control_args = sorted([norm_lattice_arg, left_adjacent_lattice_arg])
                else:
                    lattice_control_args = sorted([norm_lattice_arg, right_adjacent_lattice_arg])

                self.control_points = np.array(self.lattice_neighbour_points[lattice_control_args])
                print(BColors.OKGREEN,
                      "Id:{}, Anchor:{}, Cage:{}, {}, Norm:{}, Decision_table:{}, Decision:{}, Control:{}".format(
                          self.target_id, self.anchor, self.lattice_neighbour_points, self.cage_type,
                          self.lattice_neighbour_points[norm_lattice_arg], decision_table.flatten(), decision,
                          self.control_points), BColors.ENDC)
            else:
                # Since there is some miss angels such as +180 degrees
                # Fixme: The way that will be interpreting the situation [T, T] , [F,F] and None
                self.control_points = None
                print(BColors.FAIL,
                      "Id:{}, Anchor:{}, Cage:{}, {}, Norm:{}, Decision_table:{}, Decision:{}, Control:{}".format(
                          self.target_id, self.anchor, self.lattice_neighbour_points, self.cage_type,
                          self.lattice_neighbour_points[norm_lattice_arg], decision_table.flatten(), decision,
                          self.control_points), BColors.ENDC)
                # todo: raise ValueError("The criterion of cell selection is not logically correct")
        return self.control_points

    @staticmethod
    def get_lattice_coordinate(x_dim, ordinal_index):
        """ Computes the xy coordinate of the ordinal lattice in topological map

        The RRG (0,0) is located in bottom left corner and the lattices are vertically incremented

        Args:
            x_dim (int): The grid side size when the grid is shaped in (x_dim * x_dim)
            ordinal_index (int): Indicates the neuron index number

        Returns:
            An array includes the neuron coordinates.
        """
        return [ordinal_index // x_dim, ordinal_index % x_dim]

    def get_lattice_coordinate(self, ordinal_index):
        """ Computes the xy coordinate of the ordinal lattice in topological map

        The RRG (0,0) is located in bottom left corner and the lattices are vertically incremented

        Args:
            ordinal_index (int): Indicates the neuron index number

        Returns:
            An array includes the neuron coordinates.
        """
        return [ordinal_index // self.x_dim, ordinal_index % self.x_dim]

    def _get_BL_coordinates(self, lattices):
        """ Computes the bottom left coordinate of cells

        Args:
            lattices (list): The 4 lattices index of a grid cell

        Returns:
            A tuple x,y
        """

        return self.get_lattice_coordinate(min(lattices))

    def get_cell_coordinates(self):
        """ Computes the x , y coordinate of the corresponding cell in regular rectangular grid

                A____C1    C0____A    .____C0    C1____.
                |    |      |    |    |    |      |    |
               C0____.     .____C1   C1____A      A____C0

               (X)-------(X+x_dim)
                |        |
                |        |
                |        |
            (X-1)--------(X+x_dim-1)

            KPI = C0 + C1 - 2A -> {x_dim+1:"BL", x_dim-1:"TL" , -(x_dim+1):"TR", 1-x_dim:"BR"}

        Returns:
            A dict includes the cell lattices coordinates. The first list item is corresponding to the anchor point.
        """
        if self.control_points is not None:
            _sorted_control = sorted(self.control_points)
            # e.g: BL = anchor lateral location, [1,1] = Sorted control points
            _decision_dict = {self.x_dim + 1: ["BL", [1, 0]],
                              self.x_dim - 1: ["TL", [0, 1]],
                              -(self.x_dim + 1): ["TR", [0, 1]],
                              1 - self.x_dim: ["BR", [1, 0]]}
            _kpi = sum(_sorted_control) - 2 * self.anchor
            _coordinates = _decision_dict.get(_kpi)
            _lattices = [self.control_points[_coordinates[1][0]],
                         self.anchor,
                         self.control_points[_coordinates[1][1]],
                         self.anchor + _kpi]
            self.cell_coordinate = {'position': _coordinates[0],
                                    'lattices': _lattices,
                                    'coordinates': [self.get_lattice_coordinate(_lattice) for _lattice in _lattices],
                                    'BL_coordinates': self._get_BL_coordinates(_lattices)}
        else:
            self.cell_coordinate = None
        print(self.cell_coordinate)
        return self.cell_coordinate

    def __preserve_cell(self):
        """ Evaluates the point is correctly transferred to the cell

        The topological location of cell has to be the same in both feature space and topological space

        Returns:
           A tuple of real offset
        """
        log_flag = False
        offset_list = list(self.predicted_offset)
        for index, offset_item in zip(count(), offset_list):
            if offset_item < 0:
                offset_list[index] = abs(offset_item)
                log_flag = True
            elif offset_item > 1:
                offset_list[index] = 1 - abs(offset_item % 1)
                log_flag = True
            else:
                pass
        self.predicted_offset = tuple(offset_list)
        self.predicted_position = np.add(self.cell_coordinate.get('BL_coordinates'), self.predicted_offset)
        if log_flag:
            print(BColors.WARNING +
                  "WARNING: The offset was out of cell, It preserved to {}".format(self.predicted_position) +
                  BColors.ENDC)
        return self.predicted_offset, self.predicted_position

    def inter_cell_distance(self, _controls=2):
        """ Computes the real target distance to the corresponding cell lattices

        Args:
            _controls (int): The number of the lattice to define the inside cell relational distances

        Returns:
            A dict include the distance of target to the anchor point and the control points.
        """
        __distances_dict = dict(zip(self.lattice_neighbour_points, self.target_lattice_distances))
        __distances_dict.update({self.anchor: self.target_anchor_distances})
        if _controls == 3:
            point_4th_index = self.cell_coordinate['lattices'][3]
            point_4th_vector = self.lattice_diagonal_vectors[point_4th_index]
            fourth_lattice_dist = euclidean(point_4th_vector, np.array(self.target_vector))
            __distances_dict.update({point_4th_index: fourth_lattice_dist})
        return {control: __distances_dict[control] for control in self.cell_coordinate['lattices'][0:_controls + 1]}

    def __compact(self, __magnitude=None, eps=None):
        """ Keeps the target point into the radius of the BMU

        Args:
            eps (float): The neighborhood radius around the BMU
            __magnitude (float): The magnitude of the target position vector.
        Returns:
            An array of the predicted position
        """
        predicted_position = self.predicted_position.copy()
        reduction_ratio = __magnitude / np.random.uniform(low=1E-6, high=eps)
        anchor_coordinate = self.get_lattice_coordinate(self.anchor)
        for index, anchor_comp, predicted_comp in zip(count(), anchor_coordinate, self.predicted_position):
            predicted_position[index] = (anchor_comp * (reduction_ratio - 1) + predicted_comp) / reduction_ratio
        return predicted_position

    def transform(self, eps=0.4, controls=2):
        """ non_affine transformation

        A non_affine transformations is one where the parallel lines in the space are not conserved after the
        transformations (like perspective projections) or the mid points between lines are not conserved
        (for example non_linear scaling along an axis)

        It computes the inside position of a target point after non_affine grid cell coordinate transforming
        to the Uniform RRG.

        The x,y value are related to the BL lattice point:

               TL--------TR
                |        |
               y|-----*  |
                |     |  |
               BL-----|--BR
                      x
        Args:
            eps (float): The neighborhood radius around the BMU
            controls (int): The number of the lattice to define the inside cell relational distances

        Returns:
            A tuple that indicates to excite location (x,y) of the target point in the Uniform RRG

        Note:

        """
        assert controls in [2, 3], "ERROR: the cell control points has to be in [2,3]."
        self.get_lattice_distances()
        self.get_anchor_distance()
        self.get_target_position()
        self.get_lattice_neighbour_position()
        self.get_target_angels()
        self.get_lattices_angels()
        self.get_similarity()
        self.get_corresponding_cell()
        self.get_cell_coordinates()

        def _get_coefficients(__distance_dict):
            """Computes the relational distance coefficients
            alpha = distance(C0) / distance(A)
            beat  = distance(C1) / distance(A)
            """
            print("Inter cell distances:{}".format(__distance_dict))
            __alpha = __distance_dict[self.cell_coordinate['lattices'][0]] / __distance_dict[self.anchor]
            __beta = __distance_dict[self.cell_coordinate['lattices'][2]] / __distance_dict[self.anchor]
            if controls == 3:
                __gamma = __distance_dict[self.cell_coordinate['lattices'][3]] / __distance_dict[self.anchor]
                print('\u03B1 = {} , \u03B2 = {} and \u03B3 = {}'.format(__alpha, __beta, __gamma))
                return __alpha, __beta, __gamma
            print('\u03B1 = {} and \u03B2 = {}'.format(__alpha, __beta))
            return __alpha, __beta

        def _solve_polynomial4(_alpha, _beta, lw=self.lw, _log=True, _plot=False):
            """ Computes the root of f(r) = c - 2ar^2 +br^4

            Args:
                _alpha:
                _beta:
                lw:
                _log:
                _plot:
            Returns:

            Note: the calculation is attached as pdf
            """
            # Solving equations
            c = (lw[0] ** 4 * lw[1] ** 2) + (lw[1] ** 4 * lw[0] ** 2)
            a = (lw[0] ** 2) * (lw[1] ** 2) * (_beta ** 2 + _alpha ** 2)
            b = (lw[1] ** 2) * (1 - _beta ** 2) ** 2 + (lw[0] ** 2) * (1 - _alpha ** 2) ** 2
            r = Symbol('r')

            # Todo: Use bigger cell size
            # Todo: Use nearest neighbours with real roots.
            # Todo: find the best approach for negative and complex roots
            self.roots = solve(c - (2 * a) * r ** 2 + b * r ** 4, r)
            self.roots = np.fromiter(self.roots, dtype=complex)

            if np.all(np.isreal(self.roots)):
                self.root, self.root_type = min([root_item.real for root_item in self.roots if root_item > 0]), 0
            elif np.all(np.iscomplex(self.roots)):
                # Todo: Module "np.absolute(root_item)" of complex number instead of root_item.real
                self.root, self.root_type = min(
                    [root_item.real for root_item in self.roots if root_item > 0]), 1

                # derivative -----------------------------------
                """
                syplot(c - (2 * a) * r ** 2 + b * r ** 4, (r, -1, 1))
                print(solve(- (4 * a) * r + 4 * b * r ** 3, r))
                syplot(- (4 * a) * r + 4 * b * r ** 3, (r, -1, 1))
                """

            # Note: root_type {0: real , 1: complex}
            # print the results
            if _log:
                if self.root_type == 1:
                    print(BColors.FAIL + 'Polynomial equation:' + BColors.ENDC)
                else:
                    print(BColors.OKGREEN + 'Polynomial equation:' + BColors.ENDC)
                print('a = {:.3f}, b = {:.3f}, c = {:.3f}'.format(a, b, c))
                print('f(r) = {} - {:.3f} r\N{SUPERSCRIPT TWO} + {:.3f} r\N{SUPERSCRIPT FOUR}'.format(c, a * 2, b))
                print('The roots: {} & {}'.format(self.roots, self.root_type))
                print('The r: {}'.format(self.root))
            if _plot:
                syplot(c - (2 * a) * r ** 2 + b * r ** 4, (r, -8, 8))

            return self.root

        def _get_3cycle_intersection(_r, bottom_left_coordinate=(0, 0), lw=self.lw):
            """ Computes the intersection point of 3 cycles

            We assume the bottom cell of the cell is (0,0)
                         (1,0)______(1,1)
                             |      |
                             |      |
                        (0,0)------(1,0)

            Args:
                _r (list of float): The list of the cycles' radius
                bottom_left_coordinate:
                lw:

            Returns:
                A tuble includes the x , y cordinate of the intersection point insid the cell
            """

            # Generate the anchors coordinates
            x = [bottom_left_coordinate[0], bottom_left_coordinate[0],
                 bottom_left_coordinate[0] + lw[0], bottom_left_coordinate[0] + lw[0]]
            y = [bottom_left_coordinate[1], bottom_left_coordinate[1] + lw[1],
                 bottom_left_coordinate[1] + lw[1], bottom_left_coordinate[1]]

            # compute the intersection
            y_inside = ((x[0] - x[1]) * (x[2] ** 2 - x[1] ** 2 + y[2] ** 2 - y[1] ** 2 + _r[1] ** 2 - _r[2] ** 2) -
                        (x[1] - x[2]) * (x[1] ** 2 - x[0] ** 2 + y[1] ** 2 - y[0] ** 2 + _r[0] ** 2 - _r[1] ** 2)) / \
                       (2 * (((y[0] - y[1]) * (x[1] - x[2])) - ((y[1] - y[2]) * (x[0] - x[1]))))

            x_inside = ((y[0] - y[1]) * (y[2] ** 2 - y[1] ** 2 + x[2] ** 2 - x[1] ** 2 + _r[1] ** 2 - _r[2] ** 2) -
                        (y[1] - y[2]) * (y[1] ** 2 - y[0] ** 2 + x[1] ** 2 - x[0] ** 2 + _r[0] ** 2 - _r[1] ** 2)) / \
                       (2 * (((x[0] - x[1]) * (y[1] - y[2])) - ((x[1] - x[2]) * (y[0] - y[1]))))

            print('Predicted X= {} , y={}'.format(x_inside, y_inside))
            return x_inside, y_inside

        def _rotate(xy_component, _anchor_position, lw=self.lw):
            """ Computes the x,y after rotating the cell through an angle theta

            Args:
                xy_component:
                _anchor_position (float):
                lw:
            Returns:

            """
            # todo: it has to be double checked
            if _anchor_position == "TL":  # ok
                return xy_component
            elif _anchor_position == "TR":  # ok
                return xy_component[1], lw[0] - xy_component[0]
            elif _anchor_position == "BR":
                return lw[0] - xy_component[0], lw[1] - xy_component[1]
            elif _anchor_position == "BL":
                return lw[1] - xy_component[1], xy_component[0]  # ok
            else:
                raise ValueError("The point's inter cell position is not defined")

        # fixme: the Control:None situation
        if self.control_points is None:
            warnings.warn(BColors.WARNING + "The Control points is None. (logical bug)" + BColors.ENDC)
            self.predicted_offset = (None, None)
            self.predicted_position = (None, None)
        elif self.control_points is not None:
            if controls == 2:
                alpha, beta = _get_coefficients(self.inter_cell_distance(controls))
                radii = _solve_polynomial4(alpha, beta)
                # compute the intersection point
                radius = [radii * alpha, radii, radii * beta]
                print('\u03B1.r = {:.3f} , r = {:.3f} , \u03B2.r = {:.3f} '.format(*radius))
                predicted_inside_position = _get_3cycle_intersection(radius)
                self.predicted_offset = _rotate(predicted_inside_position, self.cell_coordinate['position'])
                self.predicted_position = np.add(self.cell_coordinate.get('BL_coordinates'), self.predicted_offset)
                print('Bottom Left: x= {:.3f} , y={:.3f} + '
                      'Predicted Offset: x_o= {:.3f} , y_o={:.3f} = '
                      'Predicted Position: X= {:.3f} , Y={:.3f}'.format(*self.cell_coordinate.get('BL_coordinates'),
                                                                        *self.predicted_offset,
                                                                        *self.predicted_position))
            elif controls == 3:
                # Fixme: the calculation has to be revised
                alpha, beta, gamma = _get_coefficients(self.inter_cell_distance(controls))
                radii = _solve_polynomial4(alpha, beta)
                # compute the intersection point
                radius = [radii * alpha, radii, radii * beta]
                print('\u03B1.r = {:.3f} , r = {:.3f} , \u03B2.r = {:.3f} '.format(*radius))
                predicted_inside_position = _get_3cycle_intersection(radius)
                self.predicted_offset = _rotate(predicted_inside_position, self.cell_coordinate['position'])
                self.predicted_position = np.add(self.cell_coordinate.get('BL_coordinates'), self.predicted_offset)
                print('Bottom Left: x= {:.3f} , y={:.3f} + '
                      'Predicted Offset: x_o= {:.3f} , y_o={:.3f} = '
                      'Predicted Position: X= {:.3f} , Y={:.3f}'.format(*self.cell_coordinate.get('BL_coordinates'),
                                                                        *self.predicted_offset,
                                                                        *self.predicted_position))

            # Compacts
            self.__preserve_cell()
            _magnitude = euclidean(self.get_lattice_coordinate(self.anchor), self.predicted_position)
            if _magnitude > eps:
                self.predicted_position = self.__compact(_magnitude, eps)
                self.compacted = True
                print(BColors.WARNING + "It is compacted to {}".format(self.predicted_position) + BColors.ENDC)

        return self.predicted_position

    def set_lattice_neighbour_vectors(self, lattice_vectors=None, diagonal_vectors=None):
        """ Set the lattice_neighbour_vectors

        Each lattice_neighbour_vector is feature vector that indicated the position of the
        lattice point in a Cartesian m-dimensional space


               D1-----L1----D2
                |     |     |
               L0-----*-----L2
                |     |     |
               D0-----L3----D3


        Args:
            lattice_vectors (list(int)): A list of the sorted (clockwise) lattice_neighbour_points
            diagonal_vectors (list(int)): A list of the diagonal points of the cage

        Returns:
            A dict includes lattice_neighbour_vector (with/without diagonal)
        """
        self.sort_neighbours(inplace=True)
        self.sort_clockwise()
        lattice_vectors = dict([(l, lattice_vectors[l]) for l in self.lattice_neighbour_points])
        if lattice_vectors is not None:
            self.lattice_neighbour_vectors = np.array(list(lattice_vectors.values()))
        if diagonal_vectors is not None:
            self.lattice_diagonal_vectors = diagonal_vectors
        return self.lattice_neighbour_vectors
