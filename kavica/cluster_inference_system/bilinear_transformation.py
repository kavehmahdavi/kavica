#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Bilinear Transformation 

These module is a Metric Manifold based transformation includes:
    - Manifold: Modeling the cloud data points with leveraging of SpaceTime Curvature & General relativity theory
    - Tensor Metric

A Tensor metric makes it possible to define several geometric notions on the manifold, such
as angle at an intersection, length of a curve, area of a surface and higher-dimensional analogues (volume, etc.),

See:
    https://link.springer.com/article/10.1186/s40064-015-1038-z
    https://www.particleincell.com/2012/quad-interpolation/#ref2

    Extra:
        https://www.av8n.com/physics/geodesics.htm#sec-straight
        A computational differential geometry approach to grid generation

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
First update: 11/10/2022
Last update:
"""

import warnings
import random
import pickle
import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
import math
import json
from kavica.utils._models import dbscan
from kavica.cluster_inference_system.space_curvature_map import FSCM
from kavica.cluster_inference_system.polygon_cage import PolygonCage
from kavica.cluster_inference_system.som import PARAVER_STATES_COLOR, get_color_palette, palette_size
from kavica.utils._plots import density_plot_3d, density_plot_2d, density_plotly_3d
from kavica.utils import BColors

import matplotlib
#matplotlib.use('module://backend_interagg') # export MPLBACKEND=TkAgg

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=sys.maxsize, linewidth=200)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
plt.rcParams.update({'font.size': 16})
np.random.seed(4)
random.seed(4)

PARAVER_STATES_COLOR_PALETTE = get_color_palette(PARAVER_STATES_COLOR, None, True)

__all__ = [
    'CurrentPolygonCage',
    'BLT',
]


class CurrentPolygonCage(PolygonCage):
    """ Data class to present the current corresponding cell of the data point

    """

    def __init__(self, target_id, anchor, target_vector, anchor_vector,
                 lattice_neighbour_points, lattice_diagonal_points):
        super(CurrentPolygonCage, self).__init__(target_id, anchor, target_vector, anchor_vector,
                                                 lattice_neighbour_points, lattice_diagonal_points)

    def get_current_cell(self):
        self.get_lattice_distances()
        self.get_anchor_distance()
        self.get_target_position()
        self.get_lattice_neighbour_position()
        self.get_target_angels()
        self.get_lattices_angels()
        self.get_similarity()
        self.get_corresponding_cell()
        self.get_cell_coordinates()
        lattices_coordinates = dict(zip(self.cell_coordinate.get('lattices'),
                                        self.cell_coordinate.get('coordinates')))
        bl_coordinates = min(lattices_coordinates, key=lattices_coordinates.get)
        return lattices_coordinates.get(bl_coordinates)


class BLT(FSCM):
    """ Class Bilinear Transformation 2D (latter 3d)

        It computes both backward and forward transformation between two grids. It uses the "Bilinear Transformation"
        that take to account forth lattice coordinates.Then model a function to calculate the current coordinate of
        each data point based on the cell that it is located.

        Attributes:
            cell_lattices (ndarray): The grid cell lattice indexes.
            grid_x_features (int): Indicates the feature of x-axis for weight position
            grid_y_features (int): Indicates the feature of y-axis for weight position

        Algorithm (2D):

        Input:
            previous_lattices
            current_lattices
            data
            x_dim

        The steps:
            1- compute the previous cell lattices (4 points)
            2- compute the current cell lattices (4 points)
            3- compute the forward transformation function (previous to current)
            4- compute the backward transformation function (current to previous)
            5- apply forward transformation (previous to current):
                5.1- find the cell location of point in previous grid
                5.2- apply the transformation
            6- apply forward transformation (previous to current):
                6.1- find the cell location of point in current grid
                6.2- apply the transformation

        See:
            Convergent: https://springerplus.springeropen.com/articles/10.1186/s40064-015-1038-z
            Divergent: https://www.particleincell.com/2012/quad-interpolation/
    """

    def __init__(self, map_shape=(3, 3), training_rate=None, epoch=None, norm=None,
                 scaled=False, make_jupiter_report=False, grid_features=None,
                 neighborhood_size=None, grid_x_features=0, grid_y_features=1):

        super(BLT, self).__init__(map_shape, training_rate, epoch, norm, scaled,
                                  make_jupiter_report, grid_features, neighborhood_size)

        self.grid_x_features = grid_x_features
        self.grid_y_features = grid_y_features
        self.cell_lattices = None
        self.init_cell_width = None
        self.init_cell_length = None
        self.grid_upper_bound = None
        self.grid_lower_bound = None
        self.forward_map_parameters = pd.DataFrame(columns=['ai_1', 'ai_2', 'ai_3', 'ai_4',
                                                            'ai_5', 'ai_6', 'ai_7', 'ai_8'])
        self.forwarded_position = None
        self.backward_position = None
        self.current_cell = []

    @classmethod
    def cast_from_fscm(cls, parent: FSCM):
        """ Castes a superior class object (FSCM)

        Copy all values: It does not have any problem since they have common template

        Args:
            parent (FSCM object): An object of superior class

        Returns:
            A BLT object
        """
        blt_obj = cls()
        for key, value in parent.__dict__.items():
            blt_obj.__dict__[key] = value
        blt_obj.__constructor()
        return blt_obj

    @property
    def limits(self):
        """ The grid (x,y) limits

        Returns:
            A dict includes the min and max of the grid features
        """
        limits = self.neurons[np.ix_([0, -1], [self.grid_x_features, self.grid_y_features])].T
        return {'x_lim': limits[0], 'y_lim': limits[0]}

    def get_cell_init_limits(self, _index):
        """ Computes the cell x , y limits

        Args:
            _index (int): the cell index in the grid

        Returns:
            A dict includes {'x':[low,high] , 'y':[low,high]}
        """
        _cell_init_vectors = self.get_init_cell(_index)
        _cell_init_vectors = _cell_init_vectors[:, [self.grid_x_features, self.grid_y_features]]
        _min = _cell_init_vectors.min(axis=0)
        _max = _cell_init_vectors.max(axis=0)
        return {'x': [_min[self.grid_x_features], _max[self.grid_x_features]],
                'y': [_min[self.grid_y_features], _max[self.grid_y_features]]}

    def __constructor(self):
        """ Constructs the class

        It runs a set of functions to constructor the class and to set up the attribute

        Returns:
            self
        """
        self.cell_lattices = self.get_cells_lattices(self.x_dim)
        self.init_cell_width, self.init_cell_length = self.__cell_previous_sides(self.cell_lattices[0])
        self.grid_upper_bound = self.initial_neurons_wight[-1]
        self.grid_lower_bound = self.initial_neurons_wight[0]
        self.forwarded_position = np.zeros((self.data.shape[0], 2))
        self.backward_position = np.zeros((self.data.shape[0], 2))
        return self

    def get_current_cell(self, _index):
        """ Computes the lattices feature vectors of a cell in current mesh

        Args:
            _index (int) : The cell index, zero is in bottom left and the increasing orientation is vertical

        Returns:
            A ndarray in shape (number of cells, number of lattices, number of active features)
        """
        return self.neurons[self.cell_lattices[_index]]

    def get_init_cell(self, _index):
        """ Computes the lattices feature vectors of a cell in previous mesh

        Args:
            _index (int) : The cell index, zero is in bottom left and the increasing orientation is vertical

        Returns:
            A ndarray in shape (number of cells, number of lattices, number of active features)
        """
        return self.initial_neurons_wight[self.cell_lattices[_index]]

    def __displacement_vector(self, _cell_lattice, _from=1):
        """ Computes the displacement of a lattice between two spaces

        Args:
            _cell_lattice (int): The lattice index of the nodes
            _from (str): Indicates the source space forward == 1 or backward == -1

        Returns:
            A array in size of active feature space
        """
        return (self.neurons[_cell_lattice] - self.initial_neurons_wight[_cell_lattice]) * _from

    def __displacement_vectors(self, _cell_lattices, _from=1):
        """ Compute the displacement vector of the cell lattices

        Args:
            _cell_lattices (1d_array): An array of the lattices that describes the cell corners
            _from (str): Indicates the source space forward == 1 or backward == -1

        Returns:
            An array includes all displacement vectors
        """
        return (self.neurons[_cell_lattices] - self.initial_neurons_wight[_cell_lattices]) * _from

    def __cell_previous_sides(self, _cell_lattices):
        """ Computes the previous cell width and length

        Args:
            _cell_lattices (1d_array): An array of the lattices that describes the cell corners

        Returns:
            A dict include w and l
                         |
                      l  |
                         |-----
                             w
        """
        ll = euclidean(self.initial_neurons_wight[_cell_lattices[0]], self.initial_neurons_wight[_cell_lattices[1]])
        wb = euclidean(self.initial_neurons_wight[_cell_lattices[0]], self.initial_neurons_wight[_cell_lattices[3]])
        lr = euclidean(self.initial_neurons_wight[_cell_lattices[2]], self.initial_neurons_wight[_cell_lattices[3]])
        wt = euclidean(self.initial_neurons_wight[_cell_lattices[1]], self.initial_neurons_wight[_cell_lattices[2]])
        if ll != lr or wt != wb:
            raise ValueError("ERROR: The cell is not symmetric.")
        return ll, wb

    def __get_cell_index(self, _coordinate):
        """ Computes the ordinal index of a cell from the topological coordinates

        The index is in [0 , self.x_dim-1 ** 2].

        Args:
            _coordinate (id array): Indicates the xy coordinate of cell

        Returns:
            An int that indicates the ordinal index of a cell.
         """
        return _coordinate[1] + _coordinate[0] * (self.x_dim - 1)

    def _forward_map(self):
        """ Computes the forward map

        It maps the previous grid to the current grid cell by cell. For each cell it will give two displacement function
        f(x,y) and g(x,y). Latter, I could use those functions to compute the current coordinate of the data points.

        Returns:
            A pandas includes the cells (indexes) and the bilinear coefficients (columns)
        """

        def __get_bilinear_parameters(d, l, w, tl):
            """ Computes parameters of the bilinear transformation function

            Args:
                d (array): The displacement vectors
                l (float): Cell previous length
                w (float): Cell previous width
                tl (1d_array) : The coordinate of the top left lattice

            Returns:
                A dict include the 8 parameters

            test_configs data:
                d = np.ndarray([[0,0],[1,1],[1,0],[1,0]]), l=2, w=2, tl=[0,2]
            """
            ai_1 = ((d[2, 0] - d[1, 0]) * (l - tl[1]) + tl[1] * (d[3, 0] - d[0, 0]) + (l * w)) / (l * w)
            ai_2 = ((tl[0] + w) * (d[1, 0] - d[0, 0]) + tl[0] * (d[3, 0] - d[2, 0])) / (l * w)
            ai_3 = (d[2, 0] - d[1, 0] + d[0, 0] - d[3, 0]) / (l * w)
            ai_4 = tl[0] + d[1, 0] - (ai_1 * tl[0]) - (ai_2 * tl[1]) - (ai_3 * tl[0] * tl[1])
            ai_5 = ((l - tl[1]) * (d[2, 1] - d[1, 1]) + tl[1] * (d[3, 1] - d[0, 1])) / (l * w)
            ai_6 = ((tl[0] + w) * (d[1, 1] - d[0, 1]) + tl[0] * (d[3, 1] - d[2, 1]) + (l * w)) / (l * w)
            ai_7 = (d[0, 1] - d[1, 1] + d[2, 1] - d[3, 1]) / (l * w)
            ai_8 = tl[1] + d[1, 1] - (ai_5 * tl[0]) - (ai_6 * tl[1]) - (ai_7 * tl[0] * tl[1])
            return {'ai_1': ai_1, 'ai_2': ai_2, 'ai_3': ai_3, 'ai_4': ai_4,
                    'ai_5': ai_5, 'ai_6': ai_6, 'ai_7': ai_7, 'ai_8': ai_8}

        for _item, _cell in enumerate(self.cell_lattices):
            d_vectors = self.__displacement_vectors(_cell)
            _l, _w = self.__cell_previous_sides(_cell)
            parameters = __get_bilinear_parameters(d_vectors, _l, _w, self.initial_neurons_wight[_cell[1]])
            self.forward_map_parameters = self.forward_map_parameters.append(parameters, ignore_index=True)
        return self.forward_map_parameters

    def forward_transformation(self):
        """ Transform the data from the initial space to the current space

        After modeling the data cloud via Space curvature map (FSCM) then we use
        the transformation to displease the data points to the current space.

        Returns:
            A ndarray includes the current position of the data points
        """

        def __rrg_search(_point):
            """ Finds the cell that data point located in Regular Rectangular Grid

            The approach that use the strategy of dividing the space. It stops when there is not any grid cage to fraction.

            Args:
                _point (1d_array): Data point feature vector to search its location in grid.

            Returns:
                An int that indicate the cell index.
            """
            _point = self.data.iloc[_point].to_numpy()
            cell_location = np.floor((_point - self.grid_lower_bound) / [self.init_cell_length, self.init_cell_width])
            return self.__get_cell_index(cell_location.astype('int'))

        def __transform(_point, _init_cell):
            """ Computes the Bilinear transformation of x value

            Args:
                _point (int): The index value os the data point (in self.data)
                _init_cell (int): The index of the init cell that _point was located in

            Returns:
                A array that indicate the transformed position (coordinates) of the _point
            """
            x_point = self.data.loc[_point, self.grid_x_features]
            y_point = self.data.loc[_point, self.grid_y_features]
            fxy = (self.forward_map_parameters.loc[_init_cell, 'ai_1'] * x_point +
                   self.forward_map_parameters.loc[_init_cell, 'ai_2'] * y_point +
                   self.forward_map_parameters.loc[_init_cell, 'ai_3'] * x_point * y_point +
                   self.forward_map_parameters.loc[_init_cell, 'ai_4'])
            gxy = (self.forward_map_parameters.loc[_init_cell, 'ai_5'] * x_point +
                   self.forward_map_parameters.loc[_init_cell, 'ai_6'] * y_point +
                   self.forward_map_parameters.loc[_init_cell, 'ai_7'] * x_point * y_point +
                   self.forward_map_parameters.loc[_init_cell, 'ai_8'])
            return np.array([fxy, gxy])

        self._forward_map()
        for data_item in range(self.data.shape[0]):
            init_cell = __rrg_search(data_item)
            self.forwarded_position[data_item] = __transform(data_item, init_cell)
        return self.forwarded_position

    def backward_transformation(self):
        """ Transforms the data from the curved space to a new flat-euclidean feature space

        Mapping function for an arbitrary quadrilateral

        Returns:
            Self
        """

        def __inverse_map(_point, _cell):
            """ Computes the inverse of the forward map

            Args:
                _cell (int): The index of the
                _point (1d_array): Data point feature vector to search its location in grid.

            Returns:
                A numpy array include the new coordinate

            Test data:
                _coff.ai_1 = 1.5, _coff.ai_2= 0, _coff.ai_3 = 0, _coff.ai_4 = 0
                _coff.ai_5 = 0, _coff.ai_6 = 1, _coff.ai_7 = 0.25, _coff.ai_8 = 0
                _data = [3,1]
            """
            _coff = self.forward_map_parameters.loc[_cell, :]
            _data = self.data.loc[_point, :]
            a = _coff.ai_6 * _coff.ai_3 - _coff.ai_2 * _coff.ai_7
            c_1 = _coff.ai_8 * _coff.ai_1 - _coff.ai_5 * _coff.ai_4
            c = c_1 + _coff.ai_5 * _data[self.grid_x_features] - _coff.ai_1 * _data[self.grid_y_features]
            b_1 = _coff.ai_8 * _coff.ai_3 - _coff.ai_7 * _coff.ai_4 + _coff.ai_6 * _coff.ai_1 - _coff.ai_5 * _coff.ai_2
            b = b_1 + _coff.ai_7 * _data[self.grid_x_features] - _coff.ai_3 * _data[self.grid_y_features]

            __limits = self.get_cell_init_limits(_cell)

            _y = np.roots([a, b, c])
            _y = np.array([_root for _root in _y if not (math.isinf(_root) or math.isnan(_root))])

            # Fixme: It has to be more mathematical constraint
            for _item in _y:
                y_lim = __limits.get('y')
                x_lim = __limits.get('x')
                if y_lim[0] <= _item <= y_lim[1]:
                    Y = _item
                    X = (_data[self.grid_x_features] - _coff.ai_4 - _coff.ai_2 * Y) / (_coff.ai_1 + _coff.ai_3 * Y)
                    if x_lim[0] <= X <= x_lim[1]:
                        print(BColors.OKBLUE +
                              'Inverse map of {} to cell {}: Old position = {} & New position = {}. Roots{}'.format(
                                  _point, _cell, _data.to_list(), [X, Y], _y) + BColors.ENDC)
                        return np.array([X, Y])

        def __get_anchors(instance, full_list=False, distance='euclidean'):
            """ Computes the must similar neuron.

            It computes the distance of the instance over all neurons and returns the closest ones' index.

            Args:
                instance (array): Indicates an instance vector.
                full_list (boolean): If True, it returns a sorted list of the neurons
                distance (str): Indicates the distance measurement.

            Returns:
                A list/int that indicates the neuron index number.
            """
            if distance == 'euclidean':
                distance_list = np.apply_along_axis(euclidean, 1, self.neurons, instance)
            sorted_distance_list = np.argsort(distance_list)
            if full_list:
                return sorted_distance_list
            else:
                return sorted_distance_list[0]

        def __get_anchors_associate_to_neurons():
            """ Computes the neurons' association.

            Returns:
                A list indicates which training data is associated with each of the neurons
            """
            neurons_association_list = []
            for index, row in self.data.iterrows():
                b = __get_anchors(row)
                neurons_association_list.extend([b])
            return neurons_association_list

        self._forward_map()
        __anchors_list = __get_anchors_associate_to_neurons()
        CurrentPolygonCage.boundary_sides, PolygonCage.x_dim = self.get_boundary_sides(self.x_dim), self.x_dim
        for data_item, anchor_lattice_point in enumerate(__anchors_list):
            # init cage
            cage_object = CurrentPolygonCage(data_item,
                                             anchor_lattice_point,
                                             np.array(self.data.iloc[data_item]),
                                             self.neurons[anchor_lattice_point],
                                             self.get_neuron_direct_neighbours(anchor_lattice_point),
                                             self.get_diagonal_neighbours(anchor_lattice_point))
            # update cage
            _lattice_vectors = self.get_neuron_dict(cage_object.lattice_neighbour_points)
            _diagonal_vectors = self.get_neuron_dict(cage_object.lattice_diagonal_points)
            cage_object.set_lattice_neighbour_vectors(_lattice_vectors, _diagonal_vectors)
            current_cell_coordinate = cage_object.get_current_cell()
            self.current_cell.append(self.__get_cell_index(current_cell_coordinate))
            print(data_item, '->', current_cell_coordinate, '->', self.current_cell[-1])
            self.backward_position[data_item] = __inverse_map(data_item, self.current_cell[-1])
        return self.backward_position

    def _drop_out_of_range(self, __backward_position):
        """ Drops the out of range transformed data

        The projected data have to be in the range of the x in [min(x) , max(y)] and y in [min(y) , max(y)] of the
        grid.

        Returns:
            A return the cleaned transformed data.
        """
        _limits = self.limits
        if isinstance(__backward_position, (np.ndarray, np.generic)):
            __backward_position = __backward_position[
                np.logical_and(np.logical_and(__backward_position[:, 0] <= _limits.get('x_lim')[1],
                                              __backward_position[:, 0] >= _limits.get('x_lim')[0]),
                               np.logical_and(__backward_position[:, 1] <= _limits.get('y_lim')[1],
                                              __backward_position[:, 1] >= _limits.get('y_lim')[0]))]
            __dropped_number = self.data.shape[0] - __backward_position.shape[0]
        elif isinstance(__backward_position, pd.DataFrame):
            __in_range = __backward_position.loc[(__backward_position['X'] <= _limits.get('x_lim')[1]) &
                                                 (__backward_position['X'] >= _limits.get('x_lim')[0]) &
                                                 (__backward_position['Y'] <= _limits.get('y_lim')[1]) &
                                                 (__backward_position['Y'] >= _limits.get('y_lim')[0])]
            __in_range_index = __in_range.index.tolist()
            __out_of_ranges = __backward_position[~__backward_position.index.isin(__in_range_index)]
            __dropped_number = self.data.shape[0] - __in_range.shape[0]
        print(BColors.WARNING + 'WARNING (_drop_out_of_range): {} of row are dropped.'.format(__dropped_number) +
              BColors.ENDC)
        return __in_range

    def plot_forward_transformation(self, feature_x, feature_y, neuron_number=True, _som=True,
                                    boundary=False, point_annotation=None, _rrg=False,
                                    neuron_scatter=False, _c='lab', _s=40, _scatter=True):
        """ Plots the forward projected data

        It plots a scatter and the neurons prototype as a top layer.

        Args:
            _s (int): Indicates the data points size in scatter plot
            _scatter (boolean): If True, the (scatter) of the real data is plotted also.
            neuron_number (boolean): If True, the neuron number will be plotted.
            feature_x (str): Indicates the x-axis data series.
            feature_y (str): Indicates the y-axis data series.
            boundary (boolean): If True, the boundary nodes will be distinctive in plot.
            point_annotation (str):
                - id: The sample number will be plotted.
                - bmu: The bmu will be plotted.
                - lab: The real labels
                - None: No annotation and labeling
            _rrg (boolean): If True it shows the grid
            neuron_scatter (boolean): If True it shows the neurones
            _c (str): Indicates the categorical variable
            _som (boolean): If True It shows the SOM initial grid.

        Returns:
        """

        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)

        if _scatter:
            ax = self._scatter(self.forwarded_position, 'X', 'Y', self.labels, _s)

        if neuron_scatter:
            ax = self._neurons_scatter(feature_x, feature_y)

        if boundary:
            ax = self._boundary_scatter(feature_x, feature_y)

        if _som:
            ax = self._draw_som(ax, feature_x, feature_y)

        if neuron_number:
            ax = self._draw_neuron_number(ax, feature_x, feature_y)

        if point_annotation:
            ax = self._draw_point_annotation(ax, self.forwarded_position, feature_x,
                                             feature_y, point_annotation)

        if _rrg:
            ax = self._draw_init_rrg(ax, feature_x, feature_y)

        ax.legend().set_visible(False)
        plt.grid(False)
        plt.title('Forward Transformation Scatter Plot.')
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/weight_positions_plot.png')
            plt.close()
        else:
            plt.show()
            plt.close()

    def plot_backward_transformation(self, feature_x, feature_y, neuron_number=True,
                                     boundary=False, point_annotation=None, _rrg=True, _scatter=True,
                                     neuron_scatter=False, _som=True, _c='lab', _drop=False, _s=40):
        """ Plost the forward projected data

        It plots a scatter and the neurons prototype as a top layer.

        Args:
            _s (int): Indicates the data points size in scatter plot
            _scatter (boolean): If True, the (scatter) of the real data is plotted also.
            neuron_number (boolean): If True, the neuron number will be plotted.
            feature_x (str): Indicates the x-axis data series.
            feature_y (str): Indicates the y-axis data series.
            boundary (boolean): If True, the boundary nodes will be distinctive in plot.
            point_annotation (str):
                - id: The sample number will be plotted.
                - bmu: The bmu will be plotted.
                - lab: The real labels
                - None: No annotation and labeling
            _rrg (boolean): If True it shows the grid
            neuron_scatter (boolean): If True it shows the neurones
            _c (str): Indicates the categorical variable
            _som (boolean): If True It shows the SOM initial grid.
            _drop (boolean): If True It drops out of cell limits data points

        Returns:
        """
        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)

        if _scatter:
            ax = self._scatter(self.backward_position, 'X', 'Y', self.labels, _s, False)

        if neuron_scatter:
            ax = self._neurons_scatter(feature_x, feature_y)

        if boundary:
            ax = self._boundary_scatter(feature_x, feature_y)

        if _som:
            ax = self._draw_som(ax, feature_x, feature_y)

        if neuron_number:
            ax = self._draw_neuron_number(ax, feature_x, feature_y)

        if point_annotation:
            ax = self._draw_point_annotation(ax, self.backward_position, feature_x,
                                             feature_y, point_annotation)

        if _rrg:
            ax = self._draw_init_rrg(ax, feature_x, feature_y)

        ax.legend().set_visible(False)
        plt.grid(False)
        plt.title('Backward Transformation Scatter Plot.')
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/weight_positions_plot.png')
            plt.close()
        else:
            plt.show()
            plt.close()

    def plot_current_cell(self, feature_x, feature_y, neuron_number=True,
                          boundary=True, _scatter=True, point_annotation=None,
                          _rrg=True, neuron_scatter=False, _som=True, _c='lab', _s=40):
        """ Plots the forward projected data

        It plots a scatter and the neurons prototype as a top layer.

        Args:
            _s (int): Indicates the data points size in scatter plot
            _scatter (boolean): If True, the (scatter) of the real data is plotted also.
            neuron_number (boolean): If True, the neuron number will be plotted.
            feature_x (str): Indicates the x-axis data series.
            feature_y (str): Indicates the y-axis data series.
            boundary (boolean): If True, the boundary nodes will be distinctive in plot.
            point_annotation (str):
                - id: The sample number will be plotted.
                - bmu: The bmu will be plotted.
                - lab: The real labels
                - None: No annotation and labeling
            _rrg (boolean): If True it shows the grid
            neuron_scatter (boolean): If True it shows the neurones
            _c (str): Indicates the categorical variable
            _som (boolean): If True It shows the SOM initial grid.

        Returns:
        """
        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)

        if _scatter:
            data_df = self.data[[self.grid_x_features, self.grid_y_features]]
            data_df.columns = ['X', 'Y']
            ax = self._scatter(data_df, 'X', 'Y', self.current_cell, _s)

        if neuron_scatter:
            ax = self._neurons_scatter(feature_x, feature_y)

        if boundary:
            ax = self._boundary_scatter(feature_x, feature_y)

        if _som:
            ax = self._draw_som(ax, feature_x, feature_y)

        if neuron_number:
            ax = self._draw_neuron_number(ax, feature_x, feature_y)

        if point_annotation:
            ax = self._draw_point_annotation(ax, self.data, feature_x,
                                             feature_y, point_annotation)

        if _rrg:
            ax = self._draw_init_rrg(ax, feature_x, feature_y)

        ax.legend().set_visible(False)
        plt.grid(False)
        plt.title('Cell Selection Scatter Plot.')
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/weight_positions_plot.png')
            plt.close()
        else:
            plt.show()
            plt.close()

    def plot_init_weight_positions_bw(self, feature_x, feature_y, neuron_number=True,
                                      _scatter=True, boundary=True, _c=None, _drop=False, _s=40):
        """ Plots the neurons initial prototype.

        It plots a scatter and the neurons prototype as a top layer.

        Args:
            _scatter (boolean): If True, the (scatter) of the real data is plotted also.
            neuron_number (boolean): If True, the neuron number will be plotted.
            feature_x (str or int): Indicates the x-axis data series.
            feature_y (str or int): Indicates the y-axis data series.
            boundary (boolean): If True, the boundary nodes will be distinctive in plot.
            _drop (boolean): If True It drops out of cell limits data points
            _c (str): Indicates the categorical variable
            _s (int) : The size of the data points in the scatter plot

        Returns:
        """
        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)

        if _scatter:
            ax = self._scatter(self.backward_position, 'X', 'Y', self.labels, _s)

        if boundary:
            ax = self._init_boundary_scatter(feature_x, feature_y)

        ax = self._draw_init_som(ax, feature_x, feature_y)

        if neuron_number:
            ax = self._draw_init_neuron_number(ax, feature_x, feature_y)

        plt.grid(False)
        ax.legend().set_visible(False)
        plt.title('SOM Weight Positions and Scatter Plot')
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/weight_positions_plot.png')
        else:
            plt.show()
        plt.close()


def __test_me(_dataset='iris', _path='outputs/fscm/fscf_test.pkl'):
    FORWARD_PATH = "outputs/fscm/forwared.pkl"
    BACKWARD_PATH = "outputs/fscm/backwared.pkl"

    def __config(__dataset):
        """ Reads the configuration info from the file

        The config file is in test_configs/blt_test_config.json

        Args:
            __dataset (str): Indicates the dataset name

        Returns:
            A dict includes the configuration
        """
        with open('config/test_configs/blt_test_config.json') as json_data:
            dataset_dict = json.load(json_data)
        return dataset_dict.get(__dataset)

    _config = __config(_dataset)

    if _config.get('full'):
        with open(_path, 'rb') as infile:
            fscm_obj = pickle.load(infile)

        fscm_obj.cat2int()
        blt_obj = BLT.cast_from_fscm(fscm_obj)
        blt_obj.forward_transformation()
        blt_obj.backward_transformation()

        blt_obj.plot_init_weight_positions(0, 1, neuron_number=True, _c='lab', _s=_config.get('s'))
        blt_obj.plot_weight_positions(0, 1, point_annotation='id', neuron_number=False, _c='lab',
                                      _s=_config.get('s'))
        blt_obj.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _c='lab',
                                      _s=_config.get('s'), _scatter=False)
        blt_obj.plot_forward_transformation(0, 1, point_annotation=None, neuron_number=False, _rrg=True,
                                            _s=_config.get('s'), _c='lab')
        blt_obj.plot_backward_transformation(0, 1, point_annotation=None, neuron_number=False, _rrg=False,
                                             _s=_config.get('s'), _c='lab')
        blt_obj.plot_init_weight_positions_bw(0, 1, neuron_number=False, _c='lab', _s=_config.get('s'))
        blt_obj.plot_current_cell(0, 1, point_annotation=None, neuron_number=False, _rrg=True, _s=_config.get('s'))

        # Store forward
        df = pd.DataFrame(data=blt_obj.forwarded_position, columns=['X', 'Y'])
        df['Class'] = blt_obj.labels
        pd.DataFrame(df).to_pickle(FORWARD_PATH)

        # Store & clustering backward
        df = pd.DataFrame(data=blt_obj.backward_position, columns=['X', 'Y'])
        df['Class'] = blt_obj.labels

        # df = blt_obj._drop_out_of_range(df)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        pd.DataFrame(df).to_pickle(BACKWARD_PATH)

        # clustering forward
        df = pd.read_pickle(FORWARD_PATH)
        print("=" * 20, "Forward Transformation", "=" * 20)
        dbscan(df=df, _eps=_config.get('epc_fw'), _min_points=_config.get('minpt_fw'),
               x_axis='X', y_axis='Y', _lab='Class', _s=_config.get('s'))

    # clustering backward
    df = pd.read_pickle(BACKWARD_PATH)
    print(df.shape)
    print("=" * 20, "Backward Transformation", "=" * 20)
    dbscan(df=df, _eps=_config.get('epc_bw'), _min_points=_config.get('minpt_bw'), x_axis='X',
           y_axis='Y', _lab='Class', _s=_config.get('s'), _lab_font=22, show_grid=False)

    # 3D density plot
    print("=" * 20, "3D density plot", "=" * 20)
    density_plot_3d(df, 'X', 'Y')
    density_plot_2d(df, 'X', 'Y')
    density_plotly_3d(df, 'X', 'Y')

    # Todo: a test_configs for the gromacs case.
    """
    df_duration = pd.read_csv('../ECII/data/PhD_thesis/gromacs_64p.chop1.cl.IPCxINS.DATA.ECII.csv')
    _duration = df_duration['Duration']
    dbscan(df=df, _eps=_config.get('epc_bw'), _min_points=_config.get('minpt_bw'), 
           x_axis='X', y_axis='Y', _lab='Class',
           _s=_config.get('s'), _lab_font=22, show_grid=False, duration=_duration)
    """


if __name__ == '__main__':
    # TODO: add the command line parsr to here
    # TODO: Add the iris as a test_configs example (full-test_configs)
    # Fixme: the evaluation of the dbscan needs change
    __test_me(_dataset='wifi')
