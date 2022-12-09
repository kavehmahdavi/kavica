#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Feature Space Curvature Map class

These module is a Metric Manifold. It computes the feature space curvature. This model can describe the density
structure of the data. Then later, we use this model to homogenize the data density

References:
    - https://ieeexplore.ieee.org/document/9892921

See:
    - https://www.youtube.com/watch?v=qBJCCe81OCg

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
First update: 13/10/2022
"""
import pprint
import random
import pickle
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from matplotlib.patches import RegularPolygon
from scipy.spatial.distance import euclidean
from tabulate import tabulate
from kavica.utils._plots import (get_color_palette, palette_size, PARAVER_STATES_COLOR)
from kavica.cluster_inference_system.som import get_rrg
from kavica.cluster_inference_system.somis import SOMIS
from kavica.cluster_inference_system.polygon_cage import PolygonCage
from sklearn.preprocessing import StandardScaler
from kavica.utils._plots import density_plot_3d, density_plot_2d, density_plotly_3d
from kavica.utils import map_index_level, BColors
from kavica.utils._models import *
from kavica.utils._prv_utility import *
from kavica.utils._generate_dataset import make_dataset
import json

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
plt.rcParams.update({'font.size': 16})
np.set_printoptions(threshold=sys.maxsize, linewidth=200)
np.random.seed(4)
random.seed(4)

PARAVER_STATES_COLOR_PALETTE = get_color_palette(PARAVER_STATES_COLOR, None, True)

__all__ = ['FSCM', ]


class FSCM(SOMIS):
    """ Feature Space Curvature Map

    It computes the feature space wrapping with the mass of the data and their mutual gravitation. In the analogy to
    the general relativity theory, the feature space will be wrapped with the distribution of the data mass all over
    the space.
    We model it as a one layer ANN, and we use the leveraging of SOM as well.

    """

    def __init__(self, map_shape=(5, 5), training_rate=0.3, neighborhood_size=1, epoch=1000, norm=None, scaled=False,
                 make_jupiter_report=True, grid_features=[0, 1], lr_decay=.00001, radius_decay=.00001):
        super(FSCM, self).__init__(map_shape, training_rate, neighborhood_size, epoch, norm, scaled,
                                   make_jupiter_report, grid_features, lr_decay, radius_decay)
        self.mass = np.ones((map_shape[0] * map_shape[1],), dtype=int)
        self.unified_projected_position = np.empty((0, 2), float)
        self.neighborhood_size = neighborhood_size
        self.root_types = np.empty((0, 0), int)
        self.grid_features = grid_features
        self.torsion_list = []
        self.g = 1  # Gravitational
        self.hyper_parameters = None

    def gamma(self, c, neurons_xy_coordinates, topology_size, neighborhood_size, training_rate):
        """ Computes neighborhood of a neurone

        Args:
            neurons_xy_coordinates (numpy): Indicates the neurons coordinates
            topology_size (int): Indicates the som grid size
            c (int): indicates the neurons index
            neighborhood_size (float): Indicates the neighborhood size of each BMU

        Returns:
            A numpy.ndarray when m on the grid is in neighborhood then alpha else 0.0
        """
        # TODO: it has to vectorized
        # lookup the 2D map coordinate for c
        c_xy_coordinate = neurons_xy_coordinates[c,]

        # a matrix with each row equal to c 2D
        c_xy_coordinates = np.outer(np.linspace(1, 1, topology_size), c_xy_coordinate)

        # distance vector of each neuron from c in terms of map coordinates!
        neuron_distance_vector = np.sqrt(np.dot((c_xy_coordinates - neurons_xy_coordinates) ** 2, [1, 1]))

        # if m on the grid is in neighborhood then alpha else 0.0
        # TODO: the training rate has to be tuned based on the mass and distance (Gravitation)
        neighborhood = np.where(neuron_distance_vector < neighborhood_size * 1.5, training_rate, 0.0)
        return neighborhood

    def __initialize_neurons(self, _method='rrg', _margin=0.2):
        """ Set the initial position of any neuron

        Args:
            _method (str): indicates the initialization method:
                - rrg: regular rectangular grid (our method I)
                - irrg: irregular rectangular grid(our method II)
                - pca_rrg: linear projection method with regular rectangular grid
                - pca_irrg: linear projection method with irregular rectangular grid

        Notes:
            The methods ['pca_rrg', 'irrg', 'pca_irrg'] are on progress.

        Returns:
            A numpy array where each row represents a neuron, each column represents a dimension.
        """
        # TODO: The methods ['pca_rrg', 'irrg', 'pca_irrg'] are on progress. Later update the assert
        assert _method in ['rrg'], "Error: the initialization method does not exist"

        # Indicates the number of the neurones
        topology_size = self.x_dim * self.y_dim
        instances_number, features_number = self.data.shape
        feature_bound = self.data.describe().loc[['min', 'max']]

        # Neurons holder
        if _method == 'rrg':
            init_vectors = get_rrg(feature_bound, topology_size, features_number,
                                   grid_features=self.grid_features, margin=_margin)
        self.initial_neurons_wight = init_vectors.copy()

        self.animation[0, :, :] = init_vectors[:, self.grid_features]
        return init_vectors

    def initialize_network(self, method=None):
        """ Initializes the position of the whole SOM network

        Args:
            method (str): indicates the initialization method:
                - rrg: regular rectangular grid (our method I)
                - irrg: irregular rectangular grid(our method II)
                - pca_rrg: linear projection method with regular rectangular grid
                - pca_irrg: linear projection method with irregular rectangular grid

        Notes:
            The methods ['pca_rrg', 'irrg', 'pca_irrg'] are on progress.

        Returns:
            A numpy array where each row represents a neuron, each column represents a dimension.
        """
        # TODO: The methods ['pca_rrg', 'irrg', 'pca_irrg'] are on progress. Later update the assert
        assert method in ['rrg'], "Error: the initialization method does not exist"

        # The vector size of the neurons and instances are equal.
        instances_number, features_number = self.data.shape
        topology_size = self.x_dim * self.y_dim

        # Hyper parameters: compute the initial neighborhood size and epoch
        # Fixme: It needs a dynamic function to adjust it gradually instead of the exponential descending by time
        neighborhood_size = self.neighborhood_size  # max(self.x_dim, self.y_dim) + 1
        neighborhood_epoch = np.ceil(self.epoch / neighborhood_size)

        self.neurons = self.__initialize_neurons(_method=method)

        # Constants for the Gamma function
        neurons_xy_coordinate = np.matrix.transpose(
            self._get_neurons_xy_coordinates(range(0, topology_size)).reshape(2, topology_size))

        self.hyper_parameters = {'neighborhood_epoch': neighborhood_epoch,
                                 'neighborhood_size': neighborhood_size,
                                 'training_rate': self.training_rate}
        pprint.pprint(self.hyper_parameters)
        return (instances_number, features_number, topology_size, self.neurons,
                neurons_xy_coordinate, neighborhood_epoch, neighborhood_size)

    def _train_som(self, _plot=True, fix_boundary=False, _init_method=None,
                   hv_movement=False, x_index=0, y_index=1, gravity=True):
        """ The stochastic SOM training algorithm (Son class method overwrite)

        Args:
            _plot (boolean): If True, the model training progress will be plotted
            fix_boundary (boolean): If True, the boundary nodes wil not updated during training
            _init_method (str): indicates the initialization method
            hv_movement (boolean): If True the boundary nodes could move just vertically or horizontally
            x_index (int): The x_axis feature index in feature space
            y_index (int): The y_axis feature index in feature space
            gravity (boolean): If true the BMU compute based on Fg = M.g/ R^2
        """
        # fixme: It has a problem with the feature wight plot (when training rate in high)

        # Initiate the neurons
        instances_number, features_number, topology_size, neurons, neurons_xy_coordinate, neighborhood_epoch, neighborhood_size = self.initialize_network(
            method=_init_method)

        self.neurons_energy.append(self.mean_distance_closest_unit(self.data, neurons))
        step_counter = 0  # counts the number of epochs per neighborhood_epoch

        # Use for moving the boundary nodes aligned with the x , y features
        if hv_movement:
            boundary_catalog = self.get_boundary_sides(self.x_dim)
            v_side = boundary_catalog.get('left') + boundary_catalog.get('right')
            h_side = boundary_catalog.get('top') + boundary_catalog.get('bottom')

        # training epoch by epoch
        for epoch in range(self.epoch):
            # Todo: Add the training progress data

            """
            # neighborhood size decreases in discrete neighborhood_epoch
            step_counter = step_counter + 1
            if step_counter == neighborhood_epoch:
                step_counter = 0
                neighborhood_size = neighborhood_size - 1
            """
            # neighborhood size decreases in discrete neighborhood_epoch
            learn_rate = self.training_rate * np.exp(-epoch * self.lr_decay)
            neighborhood_size = self.neighborhood_size * np.exp(-epoch * self.radius_decay)

            # Pick a sample vector in random
            random_index = randint(0, instances_number - 1)
            random_vector = self.data.iloc[[random_index]]

            # competitive step
            # TODO: it has to be vectorized
            xk_m = np.outer(np.linspace(1, 1, topology_size), random_vector)
            if gravity:
                diff = neurons - xk_m
                c = self.get_associated_neuron(random_vector)
            else:
                diff = neurons - xk_m
                squ = diff * diff
                s = np.dot(squ, np.linspace(1, 1, features_number))
                o = np.argsort(s)
                c = o[0]

            self.mass[c] += 1

            # update the neurones weight vector
            # TODO: it have to vectorized
            gamma_m = np.outer(self.gamma(c, neurons_xy_coordinate, topology_size, neighborhood_size, learn_rate),
                               np.linspace(1, 1, features_number))

            # It is controlling the boundary nodes movements
            if not fix_boundary:
                neurons = neurons - diff * gamma_m
            else:
                neurons[self.unboundary_neurons_indexes] = neurons[self.unboundary_neurons_indexes] - \
                                                           diff[self.unboundary_neurons_indexes] * \
                                                           gamma_m[self.unboundary_neurons_indexes]
            if hv_movement:
                neurons[v_side, y_index] = neurons[v_side, y_index] - diff[v_side, y_index] * gamma_m[v_side, y_index]
                neurons[h_side, x_index] = neurons[h_side, x_index] - diff[h_side, x_index] * gamma_m[h_side, x_index]

            self.animation[epoch + 1, :, :] = neurons[:, self.grid_features]
            self.neurons_energy.append(self.mean_distance_closest_unit(self.data, neurons))
        if _plot:
            self.plot_training_progress(log=False)
        self.neurons = neurons

    def _prepare_data(self, _filter='all'):
        """ Clean data

        Convert the data to an integrated data frame and drop Nones and filter data before and after projection

        Args:
            _filter (str): Just show the sub category of the data points:
                - real: The projected data with the real root
                - complex: The projected data with the complex root
                - all: no filter (default)

        Returns:
            A pandas includes the clean data
        """
        _df = pd.concat([pd.DataFrame(self.unified_projected_position, columns=['X', 'Y']),
                         pd.DataFrame(self.torsion_list, columns=['G']),
                         self.labels.rename('lab'), pd.DataFrame(self.root_types, columns=['root'])], axis=1)
        shape_0 = _df.shape
        _df.dropna(inplace=True)
        print(BColors.WARNING +
              "plot_unified_projection >> {} data point does not projected with no control point".format(
                  shape_0[0] - _df.shape[0]) +
              BColors.ENDC)

        filter_dict = {'real': 0, 'complex': 1}
        if _filter == 'all':
            pass
        else:
            _df = _df[_df['root'] == filter_dict.get(_filter)]
            print(BColors.OKGREEN +
                  "plot_unified_projection >> Filter: {} data point are selected by applying {} filter.".format(
                      _df.shape[0], _filter) +
                  BColors.ENDC)
        return _df

    def unified_projection(self, method='topo', bmu_number=5):
        """ Compute the location of each data point in the topological mesh/grid.

        We assume each data point assigned to the first BMU (Best Match Unit). So it will be located
        in a grid cell of the topological grid. Here we compute to excite position of each data point
        in the grid cell by normalizing and unifying the distance of the point to the first three BMUs.

        Args:
            method (str): It will indicate how it finds the anchor points. The location of a data point is
            calculated based on the scaled distance to the:
                - bmu: first three BMUs
                - topo: first three topological neighbours
            bmu_number (int): The first N BMUs
            _ filter (str): Just show the sub category of the data points:
                - real: The projected data with the real root
                - complex: The projected data with the complex root
                - all: no filter (default)
        Returns:
            A numpy includes the projected data in unified feature space
        """
        if method is 'bmu':
            # Fixme: it has to be revised. Maybe it is not a good approach
            anchor_matrix = np.empty((0, bmu_number), dtype=int)
            to_anchor_distance_matrix = np.empty((0, bmu_number), dtype=float)
            for data_item, bmu_item in enumerate(self.neurons_association_list):
                anchor_points = self.get_bmu(self.data.iloc[data_item], bmu_number)
                anchor_matrix = np.append(anchor_matrix, np.array([anchor_points]), axis=0)
                to_anchor_distance_matrix = np.append(to_anchor_distance_matrix,
                                                      np.array([np.apply_along_axis(euclidean, 1,
                                                                                    self.neurons[anchor_points],
                                                                                    np.array(
                                                                                        self.data.iloc[data_item]))]),
                                                      axis=0)
                print(data_item, ':', anchor_points, '-->', to_anchor_distance_matrix[-1])
                # , '==>', np.true_divide(to_anchor_distance_matrix[-1], to_anchor_distance_matrix[-1][0])
        elif method is 'topo':
            # Fixme: it has to be revised in case of complex numbers
            PolygonCage.boundary_sides, PolygonCage.x_dim = self.get_boundary_sides(self.x_dim), self.x_dim
            for data_item, anchor_lattice_point in enumerate(self.neurons_association_list):
                # Init cage
                cage_object = PolygonCage(data_item,
                                          anchor_lattice_point,
                                          np.array(self.data.iloc[data_item]),
                                          self.neurons[anchor_lattice_point],
                                          self.get_neuron_direct_neighbours(anchor_lattice_point),
                                          self.get_diagonal_neighbours(anchor_lattice_point))
                # update cage
                _lattice_vectors = self.get_neuron_dict(cage_object.lattice_neighbour_points)
                _diagonal_vectors = self.get_neuron_dict(cage_object.lattice_diagonal_points)
                cage_object.set_lattice_neighbour_vectors(_lattice_vectors, _diagonal_vectors)
                self.unified_projected_position = np.vstack((self.unified_projected_position, cage_object.transform()))
                self.root_types = np.append(self.root_types, cage_object.root_type)
            return self.unified_projected_position
            # Todo: Create the projected array and plot BMU
            """
            # find the sorted neighbours
            # target_lattice_distances = np.apply_along_axis(euclidean, 1,self.neurons[lattice_neighbour_points],np.array(self.data.iloc[data_item]))
            # print("Anchor:", anchor_lattice_point, '-', lattice_neighbour_points, '-', target_lattice_distances)

            # target_position = np.array(self.data.iloc[data_item]) - self.neurons[anchor_lattice_point]
            # print("Target position:", target_position)

            # lattice_neighbour_position = self.neurons[lattice_neighbour_points] - self.neurons[anchor_lattice_point]
            # print("Neighbours position:", lattice_neighbour_position)

            # target_lattice_angles = np.apply_along_axis(self.angle_between_vectors, 1,lattice_neighbour_position,target_position, True)
            # print("Target angles:", target_lattice_angles)

            _index = list(range(len(lattice_neighbour_position)))
            for left_norm, right_norm in zip(_index, np.roll(_index, -1)):
                print(self.angle_between_vectors(lattice_neighbour_position[left_norm],
                                                 lattice_neighbour_position[right_norm], True))
            """
        else:
            raise ValueError("The anchoring method {} is not defined.".format(method))

    def unified_projection_report(self, _plot=True):
        """ Generate tabular report about the unified projection.

        Args:
            _plot (boolean): If True plot the bar chart of root type per class label

        Returns:
            A pandas
        """
        __mapper = {0: 'Real', 1: 'Complex'}
        df = pd.concat([pd.DataFrame(self.unified_projected_position, columns=['X', 'Y']),
                        self.labels.rename('lab'), pd.DataFrame(self.root_types, columns=['root'])], axis=1)
        table_content = df.pivot_table(index=['lab', 'root'], aggfunc='size')
        table_content.index.set_names(['Label', 'Root'], inplace=True)
        table_content.index = map_index_level(index=table_content.index, mapper=__mapper, level=1)
        table_content = table_content.unstack()
        print(tabulate(table_content, headers=__mapper.values(), tablefmt='fancy_grid'))
        if _plot:
            table_content.plot(kind='bar', stacked=True, figsize=self._figsize,
                               title="The labels and root type interconnection")
            plt.show()
        return table_content

    def _draw_init_rrg(self, ax, _x, _y):
        """ Draws the initial regular rectangular grid (blue lines)

        Args:s
            _x (str): The x-axis feature name
            _y (str): The y-axis feature name

        Returns:
        """
        boundary_catalog = self.get_boundary_sides(self.x_dim)
        for start_vertex, end_vertex in zip(boundary_catalog.get('left'), boundary_catalog.get('right')):
            ax.plot(self.neurons[[start_vertex, end_vertex], self.get_feature_index(_x)],
                    self.neurons[[start_vertex, end_vertex], self.get_feature_index(_y)],
                    'b-', linewidth=0.4)
        for start_vertex, end_vertex in zip(boundary_catalog.get('top'), boundary_catalog.get('bottom')):
            ax.plot(self.neurons[[start_vertex, end_vertex], self.get_feature_index(_x)],
                    self.neurons[[start_vertex, end_vertex], self.get_feature_index(_y)],
                    'b-', linewidth=0.4)
        return ax

    def _draw_spline(self, ax, _x, _y, _vh):
        """ Draws the self organizing map topological graph as a layer over the plat

        Args:
            _x (str): The x-axis feature name
            _y (str): The y-axis feature name
            _vh (str): Show vertical or horizontal or both splines

        Returns:
        """

        # Fixme: The star_flag should be checked again
        direct_neighbours = self.get_oriented_direct_neighbours(self.x_dim,
                                                                self.y_dim,
                                                                star_flag=False,
                                                                orientation=_vh)
        for neurons_connection in direct_neighbours:
            ax.plot(self.neurons[neurons_connection, self.get_feature_index(_x)],
                    self.neurons[neurons_connection, self.get_feature_index(_y)],
                    'r-', linewidth=0.4)

        return ax

    def _draw_point_annotation(self, ax, _data, _x, _y, _note='id'):
        """ Draws the data points annotation as a layer over the plat

        Args:
            _x (str): The x-axis feature name
            _y (str): The y-axis feature name
            _note (str): Indicates which data should be annotated for each data
                - id: The sample number will be plotted.
                - bmu: The bmu will be plotted.
                - lab: The real labels
                - cc: Current cell (Note: it is jsu applicable for the current cell plot)
                - None: No annotation and labeling

        Returns:
        """
        __data = pd.DataFrame(data=_data, columns=[_x, _y])
        if _note == 'bmu':
            for index, row in __data.iterrows():
                ax.annotate(self.neurons_association_list[index],
                            (row[self.get_feature_index(_x)],
                             row[self.get_feature_index(_y)]))
        elif _note == 'id':
            for index, row in __data.iterrows():
                ax.annotate(index,
                            (row[self.get_feature_index(_x)],
                             row[self.get_feature_index(_y)]))
        elif _note == 'lab':
            for index, row in __data.iterrows():
                ax.annotate(self.labels[index],
                            (row[self.get_feature_index(_x)],
                             row[self.get_feature_index(_y)]))
        elif _note == 'cc':
            for index, row in __data.iterrows():
                ax.annotate(self.current_cell[index],
                            (row[self.get_feature_index(_x)],
                             row[self.get_feature_index(_y)]))
        else:
            raise ValueError("ERROR: the annotation {} is not defined.".format(_note))
        return ax

    def plot_unified_projection(self, _c=None, point_annotation=None, _filter='all'):
        """ Plot the non-affine projected data points into the URRG (Uniform Regular Rectangular Grid)

        It plots a scatter and the neurons prototype as a top layer.

        Args:
            _c (str): The data point color index
                - lab: The original labels
                - root: Root type real=0 & complex=1
            point_annotation (str):
                - id: The sample number will be plotted.
                - bmu: The bmu will be plotted.
                - lab: The real labels
                - None: No annotation and labeling (default)
            _filter (str): Just show the sub category of the data points:
                - real: The projected data with the real root
                - complex: The projected data with the complex root
                - all: no filter (default)
        Returns:
        """
        _pp_df = self._prepare_data(_filter=_filter)

        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)
        ax = self._scatter(_pp_df, 'X', 'Y', self.labels)

        if point_annotation == 'bmu':
            for index, _ in self.data.iterrows():
                if np.all(self.unified_projected_position[index, :]):
                    ax.annotate(self.neurons_association_list[index],
                                (self.unified_projected_position[index, 0],
                                 self.unified_projected_position[index, 1]))
        elif point_annotation == 'id':
            for index, _ in self.data.iterrows():
                if np.all(self.unified_projected_position[index, :]):
                    ax.annotate(index,
                                (self.unified_projected_position[index, 0],
                                 self.unified_projected_position[index, 1]))
        elif point_annotation == 'lab':
            for index, _ in self.data.iterrows():
                if np.all(self.unified_projected_position[index, :]):
                    ax.annotate(self.labels[index],
                                (self.unified_projected_position[index, 0],
                                 self.unified_projected_position[index, 1]))

        # make the legend
        if _c == 'root':
            label_text = ['Real', 'Complex']
        elif _c == 'lab':
            label_text = []
            for label_item in sorted(_pp_df['lab'].unique()):
                label_text.append('C{}'.format(label_item))
        handles, legend_labels = ax.legend_elements()
        legend_labels = map(lambda __label_item: str('$\\mathdefault{' + __label_item + '}$'), label_text)
        legend = ax.legend(handles, legend_labels, loc="lower left", title="Original")
        ax.add_artist(legend)

        major_ticks = np.arange(0, self.x_dim, 1)
        ax.set_xticks(major_ticks)
        ax.set_yticks(major_ticks)
        ax.margins(0.15)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(linestyle='--', color='red', which='major')
        plt.title('Non affine projected data in to uniform grid (filter: {})'.format(_filter))

        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/non_affine_projected.png')
            plt.close()
        else:
            plt.show()
            plt.close()

    def plot_splines(self, feature_x, feature_y, neuron_number=True, _scatter=True, boundary=True,
                     point_annotation=None, vh='V', _rrg=True, _s=40, neuron_scatter=True):
        """ Plot the feature space curvature splines (vertical / horizontal or both)

        With or without plots a scatter and the neurons prototype as a top layer.

        Args:
            _scatter (boolean): If True, the (scatter) of the real data is plotted also.
            neuron_number (boolean): If True, the neuron number will be plotted.
            feature_x (str): Indicates the x_axis data series.
            feature_y (str): Indicates the y_axis data series.
            boundary (boolean): If True, the boundary nodes will be distinctive in plot.
            neuron_scatter (boolean): If True it shows the neurones
            point_annotation (str):
                - id: The sample number will be plotted.
                - bmu: The bmu will be plotted.
                - lab: The real labels
                - None: No annotation and labeling
            vh (str): Show vertical or horizontal or both splines
            _rrg (boolean): Draw the original space grid as well
            _s (int): Indicates the data points size in scatter plot
        Returns:
        """
        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)

        self._draw_spline(ax, feature_x, feature_y, vh)

        if _scatter:
            ax = self._scatter(self.data, feature_x, feature_y, self.labels, _s)

        if neuron_scatter:
            ax = self._neurons_scatter(feature_x, feature_y)

        if boundary:
            ax = self._boundary_scatter(feature_x, feature_y)

        if neuron_number:
            ax = self._draw_neuron_number(ax, feature_x, feature_y)

        if point_annotation:
            ax = self._draw_point_annotation(ax, self.data, feature_x,
                                             feature_y, point_annotation)
        if _rrg:
            ax = self._draw_init_rrg(ax, feature_x, feature_y)

        plt.title('Splines ({}) and Scatter Plot ({}).'.format(vh, point_annotation))
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/Splines_{}_plot.png'.format(vh))
            plt.close()
        else:
            plt.show()
            plt.close()

    def plot_mass_hit(self, aligned_vertical=False, label=True):
        """ Plot the mass of each neuron

        Args:
            aligned_vertical (boolean): If True, the hexagons are aligned vertically.
            label (boolean): If True, the number of the instances are plotted in any cell.

        Returns:

        """
        # TODO: the horizontal alignment have to be revised.

        assert self.x_dim > 1 and self.y_dim > 1, "Topology map has too small dimensions, x={} and y={}.".format(
            self.x_dim, self.y_dim)

        max_association = max(self.mass)
        labels = self.mass
        coordinates = []

        for y_coordinate in range(self.y_dim):
            for x_coordinate in range(self.x_dim):
                coordinates.append([x_coordinate, y_coordinate, -(y_coordinate + x_coordinate % 2)])

        if aligned_vertical:
            horizontal_coordinates = [c[0] for c in coordinates]
            vertical_coordinates = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in coordinates]
            rotation_angel = 30
        else:
            vertical_coordinates = [c[0] for c in coordinates]
            horizontal_coordinates = [2. * np.sin(np.radians(60)) * (c[1] - c[2]) / 3. for c in coordinates]
            rotation_angel = 60

        fig, ax = plt.subplots(1, figsize=self._figsize, dpi=self._dpi)
        ax.set_aspect('equal')
        ax.set_facecolor('gray')

        for x, y, l in zip(horizontal_coordinates, vertical_coordinates, labels):
            hexbin = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                    orientation=np.radians(rotation_angel),
                                    facecolor='w', alpha=None, edgecolor='gray')
            ax.add_patch(hexbin)
            ax.scatter(horizontal_coordinates, vertical_coordinates, alpha=0.5)

            hexbin = RegularPolygon((x, y), numVertices=6, radius=(2.0 * int(l)) / (3.0 * max_association),
                                    orientation=np.radians(rotation_angel),
                                    facecolor='violet', alpha=None, edgecolor=None)
            ax.add_patch(hexbin)

            if label and int(l) > 0:
                ax.text(x, y, l, ha='center', va='center', size=10)
        plt.title('SOM Mass Hits')
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/som_mass_hit_plot.png')
            plt.close()
        else:
            plt.show()
            plt.close()

    def _get_gravitation_force(self, mass=0, _distance=1):
        """ Compute the gravitation force between a neuron and a data sample

         G = g.M / ||R||^2

        Args:
            g (float): Constant gravity coefficient
            mass (int): The mass value of the center of gravity (neuron)
            _distance (float): The distance between an data point and neuron

        Returns:
            A sorted list of the G-force
        """
        return self.g * mass / (_distance ** 2)

    def get_associated_neuron(self, instance, full_list=False, distance='euclidean'):
        """ Compute association neuron based on the gravity filed

        Fg = g.M.m / ||R||^2

        Args:
            instance (array): Indicates an instance vector.
            full_list (boolean): If True, it returns a sorted list of the neurons
            distance (str): Indicates the distance measurement.
        Returns:
            A list/int that indicates the neuron index number.

        """
        distance_list = np.apply_along_axis(euclidean, 1, self.neurons, instance)
        v_func = np.vectorize(self._get_gravitation_force)
        gravity_force = v_func(self.mass, distance_list)
        gravity_force = gravity_force.tolist()
        gravity_force_list = np.argsort(gravity_force)
        if full_list:
            return gravity_force_list
        else:
            return gravity_force_list[-1]

    def get_associate_torsion(self, instance):
        """ Compute the BUM based on the gravitational force

        Args:
            instance (int): Index of a data point
        Returns:
            A list indicates which training data is associated with each of the neurons
        """
        _bmu = self.neurons_association_list[instance]
        _mass = self.mass[_bmu]
        _distance = euclidean(np.array(self.data.loc[instance, :]), self.neurons[_bmu])
        return self._get_gravitation_force(_mass, _distance)

    def _associate_to_neurons(self):
        """ Compute the neurons' association.

        Returns:
            A list indicates which training data is associated with each of the neurons

        """
        for index, row in self.data.iterrows():
            neuron_index = self.get_associated_neuron(row)
            self.neurons_association_list.extend([neuron_index])
            torsion = self.get_associate_torsion(index)
            self.torsion_list.extend([torsion])
        return self.neurons_association_list, self.torsion_list


def __fscm_test(_dataset='synt'):
    ORIGINAL_PATH = "outputs/fscm/original.pkl"

    def __config(__dataset):
        """ Reads the configuration info from the file

        The config file in /test_configs/scm_test_config.json

        Args:
            __dataset (str): Indicates the dataset name

        Returns:
            A dict includes the configuration
        """
        with open('config/test_configs/scm_test_config.json') as json_data:
            dataset_dict = json.load(json_data)
        return dataset_dict.get(__dataset)

    def __load_data(__dataset):
        if _dataset == 'iris':
            # load data
            from sklearn import datasets
            iris = datasets.load_iris()
            _y = pd.Series(iris.target)
            _data = pd.DataFrame(iris.data[:, :4])
            _data.columns = iris.feature_names
            _subset = [0, 2]  # [1,3]
            _data.columns = [0, 1, 2, 3]
            _data = _data.loc[:, _subset]
            _data.columns = [0, 1]
            _original_data = _data.copy()
            _original_data['lab'] = _y.to_numpy()
            pd.DataFrame(_original_data).to_pickle("outputs/fscm/original.pkl")
        return _data, _y

    _config = __config(_dataset)

    if _dataset == 'synt':
        # Data Generation
        _pattern = _config["pattern"]
        # FSCM paper
        if _pattern == "blobs":
            synthetic_config = {'n_samples': [100,
                                              250,
                                              100, 120,
                                              130, 180, 140,
                                              200, 190, 220],
                                'cluster_std': [1.5,
                                                3,
                                                1.2, 0.9,
                                                1.5, 1.1, 1.2,
                                                1.1, 0.8, 1.5],
                                'centers': [(22, -3),
                                            (-28, 0),
                                            (-17, 14), (-16, 9),
                                            (5, -13), (9, -9), (2, -8),
                                            (10, 15), (15, 15), (20, 15)],
                                'random_state': 4}  # Syn10
        elif _pattern == "circles":
            synthetic_config = {'n_samples': (1000, 250),
                                'factor': 0.5,
                                'noise': 0.05,
                                'random_state': 4}
        elif _pattern == "moons":
            synthetic_config = {'n_samples': 1200,
                                'noise': 0.05,
                                'random_state': 4}
        elif _pattern == "aniso":
            synthetic_config = {'n_samples': [200, 600, 300],
                                'cluster_std': [1, 1, 1],
                                'random_state': 4}  # Syntactic_9
        elif _pattern == "quantile":
            synthetic_config = {'conf': {1: {'cov': 4., 'n_samples': 800, 'n_features': 2, 'n_classes': 1},
                                         2: {'cov': 2., 'n_samples': 1500, 'n_features': 2, 'n_classes': 1},
                                         3: {'cov': 0.5, 'n_samples': 2300, 'n_features': 2, 'n_classes': 1}}}

        data, y = make_dataset(_pattern, **synthetic_config)
        original_data = data.copy()
        original_data['lab'] = y

        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")

        density_plot_3d(data, 0, 1)

        class_analysis(pickle_path=ORIGINAL_PATH, _kernel='poly', _test_size=0.20, axis_lab=['X', 'Y'])
        # best f1 score
        # cluster_analysis(original_path, _eps=0.65, _min_points=12, _grid=True, _model='dbscan', _lab='lab')
        cluster_analysis(ORIGINAL_PATH, _eps=1.41, _min_points=22, _grid=True, _model='dbscan', _lab='lab')
        # ------------------------------------------------------------------

        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)
        outfile.close()

    elif _dataset == 'iris':
        data, y = __load_data(_dataset)

        class_analysis(pickle_path=ORIGINAL_PATH,
                       _kernel=_config["classification"]["_kernel"],
                       _test_size=_config["classification"]["_test_size"],
                       axis_lab=_config["classification"]["axis_lab"])

        cluster_analysis(ORIGINAL_PATH,
                         _eps=_config["clustering"]["_eps"],
                         _min_points=_config["clustering"]["_min_points"],
                         _grid=_config["clustering"]["_grid"],
                         _model=_config["clustering"]["_model"],
                         _lab=_config["clustering"]["_lab"],
                         _s=_config["clustering"]["_s"], )

        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)

        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        m.get_animation(0, 1, _format='mp4', _output_path="outputs/som")

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCM class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)
        outfile.close()

    elif _dataset == 'prv':
        # load data
        __path = "/home/kaveh/Documents/Jesus/Trace_for_eps_tuning/nemo.6x8/nemo.6x8.10its.cl.DATA.csv"
        # "../../data/som_test_data/lulesh_27.IPC_INS.DATA.csv"
        # [' n_IPC', ' n_PAPI_TOT_INS', ' x_PAPI_L1_DCM', ' x_PAPI_L2_DCM', 'ClusterID']
        feature_list = [' n_IPC', ' n_PAPI_TOT_INS', ' x_L1 MPKI', 'ClusterID']
        data, y = load_trace_csv(__path, clusters=[6, 7, 8, 9, 10], featurs=feature_list)
        # Fixme: hard coding needed to be acceptable with index or name of features
        data.columns = [0, 1, 2]
        y.rename(None, inplace=True)

        # -------------------------------------------------------
        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)

        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        # Animation -----------------------------------
        m.plot_init_weight_positions(1, 0)
        m.plot_weight_positions(1, 0, point_annotation=None, neuron_number=False)
        m.plot_weight_positions(1, 0, point_annotation=None, neuron_number=False, _scatter=False)
        m.plot_weight_positions(1, 0, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(1, 0, point_annotation='bmu')
        m.plot_weight_positions(1, 0, point_annotation='id')
        m.plot_splines(1, 0, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(1, 0, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        predicted_data = m.unified_projection(method='topo')
        m.unified_projection_report()
        m.plot_unified_projection(_c='lab')
        m.plot_unified_projection(_c='root', _filter='all')
        m.plot_unified_projection(_c='lab', _filter='real')
        m.plot_unified_projection(_c='lab', _filter='complex')
        # m.get_animation(0, 1, _format='mp4', _output_path="/home/kaveh/Desktop/fscm_temp")

        # Reusable
        __path_projected_pickle = "outputs/fscm/non_affine.pkl"
        if _config["_filter"] is None:
            predicted_data['lab'] = m.labels
            pd.DataFrame(predicted_data).to_pickle(__path_projected_pickle)
        else:
            predicted_data['lab'] = m.labels
            pd.DataFrame(m._prepare_data(_filter=_config["_filter"])).to_pickle(__path_projected_pickle)

        # m.plot_weight_positions(0, 1, point_annotation='id')
        # m.plot_weight_positions(0, 1, point_annotation='lab')
        # m.plot_weight_positions(0, 1, point_annotation='bmu')

    elif _dataset == 'mall':
        # https://jovian.ai/manishagdas/customer-segmentation-using-dbscan
        # https://www.kaggle.com/datark1/customers-clustering-k-means-dbscan-and-ap
        __path = 'empirical_report/FSCM_paper_dataset/Mall_Customers.csv'
        feature_list = ['AnnualIncome', 'SpendingScore']
        label_name = 'Genre'
        df = pd.read_csv(__path)

        label_map = {
            'Male': 0,
            'Female': 1,
        }
        df[label_name] = df[label_name].map(label_map)

        data = df[feature_list]
        y = df[label_name]
        df = pd.read_clipboard()

        data.columns = [0, 1]
        original_data = data.copy()
        original_data['lab'] = y
        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")
        # ---------------------------------------------------------

        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)
        outfile.close()

    elif _dataset == 'sequoia':
        __path = 'empirical_report/FSCM_paper_dataset/sequoia.csv'
        feature_list = ['lat', 'long']
        label_name = 'rig'
        df = pd.read_csv(__path)
        df['rig'] = df._class.str.split().str.get(0)
        df = df.sample(n=4000)
        top_area = df[label_name].value_counts().head(10).index.to_list()
        df = df[df.rig.isin(top_area)]

        df.reset_index(inplace=True)
        data = df[feature_list]
        y = df[label_name]
        y, uniques = y.factorize()

        data.columns = [0, 1]
        original_data = data.copy()
        original_data['lab'] = y
        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")

        # ---------------------------------------------------------
        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)
        outfile.close()

    elif _dataset == 'pan_cancer':
        if not _config["_forward"]:
            __path = 'empirical_report/FSCM_paper_dataset/emperical_final_2d/pan_cancer_pca.pkl'
        else:
            __path = "outputs/fscm/forwared.pkl"

        feature_list = [0, 1]
        label_name = 'Class'
        df = pd.read_pickle(__path)
        data = df[feature_list]
        y = df[label_name]

        data.columns = [0, 1]
        original_data = data.copy()
        original_data['lab'] = y

        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")

        class_analysis(pickle_path=ORIGINAL_PATH, _kernel='poly', _test_size=0.2, axis_lab=['PC1', 'PC2'])

        cluster_analysis(ORIGINAL_PATH, _eps=13.141, _min_points=20, _grid=True, _model='dbscan', _lab='lab')
        # ---------------------------------------------------------

        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)
        outfile.close()

    elif _dataset == 'breast-cancer':
        # TODO: 8,8 680 (paper plot dim=(8,8), epo=70, tr=0.015, neighborhood_size=1.3)
        if not _config["_forward"]:
            __path = 'empirical_report/FSCM_paper_dataset/emperical_final_2d/breast-cancer_pca.pkl'
        else:
            __path = "outputs/fscm/forwared.pkl"

        feature_list = [0, 1]
        label_name = 'Diagnosis'
        df = pd.read_pickle(__path)
        data = df[feature_list]
        y = df[label_name]

        data.columns = [0, 1]
        original_data = data.copy()
        original_data['lab'] = y

        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")

        class_analysis(pickle_path=ORIGINAL_PATH, _kernel='rbf', _test_size=0.2, axis_lab=['PC1', 'PC2'])

        cluster_analysis(ORIGINAL_PATH, _eps=1, _min_points=25, _grid=False, _model='dbscan', _lab='lab')
        # ---------------------------------------------------------

        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)
        outfile.close()

    elif _dataset == 'wifi':
        if not _config["_forward"]:
            __path = 'empirical_report/FSCM_paper_dataset/emperical_final_2d/wifi_pca.pkl'
        else:
            __path = "outputs/fscm/forwared.pkl"

        feature_list = [0, 1]
        label_name = 'class'
        df = pd.read_pickle(__path)
        data = df[feature_list]
        y = df[label_name]

        data.columns = [0, 1]
        original_data = data.copy()
        original_data['lab'] = y
        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")
        class_analysis(pickle_path=ORIGINAL_PATH, _kernel='poly', _test_size=0.1, axis_lab=['PC1', 'PC2'])

        cluster_analysis(ORIGINAL_PATH, _eps=0.2481, _min_points=13, _grid=True, _model='dbscan', _lab='lab', _s=50)
        # ---------------------------------------------------------

        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_mass_hit()

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation='bmu', neuron_number=True, _scatter=True, vh='V')
        m.plot_splines(0, 1, point_annotation='id', neuron_number=True, _scatter=True, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)
        outfile.close()

    elif _dataset == 'NBA':
        if not _config["_forward"]:
            __path = 'empirical_report/FSCM_paper_dataset/emperical_final_2d/NBA_pca.pkl'
        else:
            __path = "outputs/fscm/forwared.pkl"

        feature_list = [0, 1]
        label_name = 'Playoffs'
        df = pd.read_pickle(__path)
        data = df[feature_list]
        y = df[label_name]

        data.columns = [0, 1]
        original_data = data.copy()
        original_data['lab'] = y
        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")

        class_analysis(pickle_path=ORIGINAL_PATH, _kernel='rbf', _test_size=0.2, axis_lab=['PC1', 'PC2'])
        cluster_analysis(ORIGINAL_PATH, _eps=0.6, _min_points=13, _grid=True, _model='dbscan', _lab='lab', _s=50)
        # ---------------------------------------------------------

        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)
        outfile.close()

    elif _dataset == 'hbscan':
        __path = 'empirical_report/OCA_paper_dataset/Hbscan_dataset.csv'
        feature_list = ['x', 'y']
        label_name = 'lab'
        df = pd.read_csv(__path)

        data = df[feature_list]
        y = df[label_name]

        data.columns = [0, 1]
        original_data = data.copy()
        original_data['lab'] = y
        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")
        # ---------------------------------------------------------

        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)

    elif _dataset == 'synt_mix':
        __path = 'empirical_report/FSCM_paper_dataset/arbitrary_clusters.csv'
        feature_list = ['x', 'y']
        label_name = 'lab'
        df = pd.read_csv(__path)

        data = df[feature_list]
        y = df[label_name]

        data.columns = [0, 1]
        original_data = data.copy()
        original_data['lab'] = y
        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")
        # ---------------------------------------------------------

        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)

    elif _dataset == 'synt_shape':
        # Data Generation
        _pattern = _config["pattern"]
        if _pattern == "circles":
            synthetic_config = {'n_samples': (1000, 250),
                                'factor': 0.5,
                                'noise': 0.05,
                                'random_state': 4}
        elif _pattern == "moons":
            synthetic_config = {'n_samples': 1200,
                                'noise': 0.05,
                                'random_state': 4}
        elif _pattern == "aniso":
            synthetic_config = {'n_samples': [200, 600, 300],
                                'cluster_std': [1, 1, 1],
                                'random_state': 4}

        data, y = make_dataset(_pattern, **synthetic_config)
        original_data = data.copy()
        original_data['lab'] = y

        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")

        density_plot_3d(data, 0, 1)

        class_analysis(pickle_path=ORIGINAL_PATH, _kernel='poly', _test_size=0.20, axis_lab=['X', 'Y'])
        # best f1 score
        # cluster_analysis(original_path, _eps=0.65, _min_points=12, _grid=True, _model='dbscan', _lab='lab')
        cluster_analysis(ORIGINAL_PATH, _eps=1.41, _min_points=22, _grid=True, _model='dbscan', _lab='lab')
        # ------------------------------------------------------------------

        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        """ Poly4 projection
        predicted_data = m.unified_projection(method='topo')
        m.unified_projection_report()
        if _filter is None:
            pd.DataFrame(predicted_data).to_pickle("outputs/non_affine_projection/syntatic_4/non_affine.pkl")
        else:
            pd.DataFrame(m._prepare_data(_filter=_filter)).to_pickle(
                "outputs/non_affine_projection/syntatic_4/non_affine.pkl")

        m.plot_unified_projection(_c='lab')
        m.plot_unified_projection(_c='root', _filter='all')
        m.plot_unified_projection(_c='lab', _filter='real')
        m.plot_unified_projection(_c='lab', _filter='complex')

        # m.get_animation(0, 1, _format='mp4', _output_path="/home/kaveh/Desktop/fscm_temp")
        """

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)
        outfile.close()

        m.get_animation(0, 1, _format='mp4', _output_path="/home/kaveh/Desktop/fscm_temp")

    elif _dataset == 'OFF_EVENTES':
        if not _config["_forward"]:
            __path = 'empirical_report/FSCM_paper_dataset/emperical_final_2d/OFF_EVENTES_tsen.pkl'
        else:
            __path = "outputs/fscm/forwared.pkl"

        feature_list = [0, 1]
        label_name = 'cluster'
        df = pd.read_pickle(__path)
        data = df[feature_list]
        y = df[label_name]

        data.columns = [0, 1]
        original_data = data.copy()
        original_data['lab'] = y
        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")

        cluster_analysis(ORIGINAL_PATH, _eps=4.65, _min_points=20, _grid=True, _model='dbscan', _lab='lab', _s=50)

        class_analysis(pickle_path=ORIGINAL_PATH, _kernel='poly', _test_size=0.1, axis_lab=['PC1', 'PC2'])
        # ---------------------------------------------------------

        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)
        outfile.close()

    elif _dataset == 'DRWR':
        __path = 'empirical_report/FSCM_paper_dataset/RD_BW_in_long_burst@lulesh_4+4_with_uncores_sockets1+2.chop_5it_final.csv'
        feature_list = ['RD_BW', 'WR_BW']
        df = pd.read_csv(__path)
        data = df[feature_list]
        df = pd.read_clipboard()
        data.columns = [0, 1]
        _normalize = True
        _log = False
        # Standard scale
        if _normalize:
            scaler = StandardScaler()
            data.iloc[:, :] = scaler.fit_transform(data.to_numpy())

        original_data = data.copy()
        original_data['lab'] = 1
        y = original_data['lab']

        # log
        if _log:
            original_data.style.set_precision(2)
            for column in original_data.columns:
                try:
                    original_data[column] = np.log10(original_data[column])
                except (ValueError, AttributeError):
                    pass

        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")

        # Clustering Analysis
        cluster_analysis(ORIGINAL_PATH, _eps=0.21, _min_points=8, _grid=True, _model='dbscan', _lab='lab', _s=50)

        # ---------------------------------------------------------

        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)
        outfile.close()

    elif _dataset == 'Gromacs':
        if True:
            df = pd.read_csv('../ECII/data/PhD_thesis/gromacs_64p.chop1.cl.IPCxINS.DATA.ECII.csv')
            print(df.shape)
            # df=df.sample(n=3000)
            # df.reset_index(inplace=True)
            _duration = df['Duration']
            feature_list = ['x_PAPI_L1_DCM', 'd_PAPI_TOT_INS']  # ['L1_Rat','n_PAPI_TOT_INS']
            label_name = 'ClusterID'
            # log
            if _config["_log"]:
                # df.style.set_precision(2)
                df[feature_list[0]] = df[feature_list[0]] / df[feature_list[1]]
                df[feature_list[1]] = np.log10(df[feature_list[1]])

                scaler = StandardScaler()
                df.iloc[:, :] = scaler.fit_transform(df.to_numpy())
                # df = df.replace([np.inf, -np.inf], 0)
        elif not _forward:
            __path = 'empirical_report/FSCM_paper_dataset/emperical_final_2d/Gromacs_pca.pkl'
            feature_list = [0, 1]
            label_name = 'lab'
            df = pd.read_pickle(__path)
        else:
            __path = "outputs/fscm/forwared.pkl"
            feature_list = [0, 1]
            label_name = 'lab'
            df = pd.read_pickle(__path)

        data = df[feature_list]

        y = df[label_name]
        data.columns = [0, 1]
        original_data = data.copy()
        original_data['lab'] = y
        pd.DataFrame(original_data).to_pickle("outputs/fscm/original.pkl")
        cluster_analysis(ORIGINAL_PATH, _eps=round(0.25, 3), _min_points=4,
                         _x_lab=feature_list[0], _y_lab=feature_list[1],
                         _grid=True, _model='dbscan', _lab='lab', _s=50, duration=_duration)

        # ---------------------------------------------------------
        m = FSCM(map_shape=tuple(_config["dim"]),
                 epoch=_config["epoch"],
                 training_rate=_config["training_rate"],
                 norm=_config["norm"],
                 make_jupiter_report=_config["make_jupiter_report"],
                 neighborhood_size=_config["neighborhood_size"],
                 grid_features=_config["grid_features"])

        m.fit(data, y, init_method='rrg', fix_boundary=True)
        # Validate the model---------------------------------
        for item_key, item_value in m.get_goodness_of_fit(full_list=True, k=100).items():
            print(item_key, ":", item_value)

        m.plot_init_weight_positions(0, 1, neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation='id', neuron_number=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=False, _scatter=True)
        m.plot_weight_positions(0, 1, point_annotation=None, neuron_number=True, _scatter=False)
        for inu, nu in enumerate(m.neurons):
            print(inu, ":", nu)

        m.plot_weight_positions(0, 1, point_annotation='bmu')
        m.plot_weight_positions(0, 1, point_annotation='id')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='V')
        m.plot_splines(0, 1, point_annotation=None, neuron_number=False, _scatter=False, vh='H')
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_mass_hit()

        # Supper Clustering --------------------------------
        lb = m.self_cluster()
        m.plot_cluster_association(0, 1, lb)
        m.plot_cluster_map(lb, labels=True)

        m.plot_unified_distance_matrix(explicit=False, merge_clusters=False)
        m.plot_unified_distance_matrix(explicit=False, merge_range=10)
        m.plot_neighbor_distance_hexbin()
        m.plot_weight_planes(feature='all')

        # pickle for FSCF class test_configs
        fscm_path = 'outputs/fscm/fscf_test.pkl'
        outfile = open(fscm_path, 'wb')
        pickle.dump(m, outfile)
        outfile.close()


if __name__ == '__main__':
    __fscm_test(_dataset="iris")
