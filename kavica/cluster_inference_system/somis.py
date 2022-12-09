#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Self-Organizing Map Inference System

It is the base class to model the topological structuer of both an overall feature space and a cluster itself.

It is used in other two main classes:
    - Organization Component Analysis
    - Feature Space Curvature Map


Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
First update: 01/09/2021
Last update:18/10/2022
"""

#TODO: # Regular Hyperrectangle Grid (RHG)

import random
import itertools
import pickle
import sys
import alphashape
import vg
import matplotlib.lines as mlines
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from descartes import PolygonPatch
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.preprocessing import normalize, scale
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity
from kavica.utils._bcolors import BColors
from kavica.cluster_inference_system.som import SOM, normalize_pandas
from random import randint

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize, linewidth=200)
random.seed(4)
np.random.seed(4)
plt.rcParams.update({'font.size': 16})

__all__ = ['SOMIS', ]


class SOMIS(SOM):
    """ Self organization Map Inference System.

    """

    # Fixme: set the neighbourhood size as well
    def __init__(self, map_shape=(5, 5), training_rate=0.1, neighborhood_size=1, epoch=1000, norm=None, scaled=False,
                 make_jupiter_report=True, grid_features=None, lr_decay=.00001, radius_decay=.00001):
        super(SOMIS, self).__init__(map_shape, training_rate, neighborhood_size, epoch, norm, scaled,
                                    make_jupiter_report, grid_features, lr_decay, radius_decay)
        self.inertia_data = None
        self.normalized_inertia_data = None
        self.geodesic_inertia_data = {}
        self.geodesic_lift = None
        self.geodesic_support = None
        self.geodesic_paths = {}

    def _train_som(self, _plot=True, fix_boundary=False, _init_method=None):
        """ The stochastic SOM training algorithm

        We use the active feature set to train the SOM. The we update the weights of the illustrative feature over
        the map also.

        Args:
            _plot (boolean): If True, the model training progress will be plotted
            fix_boundary (boolean): If True, the boundary nodes wil not updated during training
            _init_method (str): indicates the initialization method

        Returns:
            A numpy ndarray includes the all neurons finally position in the feature space

        """
        # fixme: It has a problem with the feature wight plot (when training rate in high)
        # Todo: The fix boundary needs test_configs

        # Initiate the neurons
        instances_number, features_number, topology_size, neurons, neurons_xy_coordinate, \
        neighborhood_epoch, neighborhood_size = self.initialize_network(method=_init_method)

        _all_features = list(self.features_indexes.values())
        _active_features_sorted = sorted(self.get_feature_indexes(self.grid_features))
        _illustrative_features_sorted = sorted(list(set(_all_features) - set(_active_features_sorted)))

        self.neurons_energy.append(self.mean_distance_closest_unit(self.data.iloc[:, _active_features_sorted],
                                                                   neurons[:, _active_features_sorted]))

        # training epoch by epoch
        for epoch in range(self.epoch):
            # Todo: Add the training progress data

            # neighborhood size decreases in discrete neighborhood_epoch
            learn_rate = self.training_rate * np.exp(-epoch * self.lr_decay)
            neighborhood_size = self.neighborhood_size * np.exp(-epoch * self.radius_decay)

            # Pick a sample vector in random
            random_index = randint(0, instances_number - 1)
            random_vector = self.data.iloc[[random_index]]

            # competitive step (It could be vectorized)
            xk_m = np.outer(np.linspace(1, 1, topology_size), random_vector)
            diff = neurons - xk_m
            squ = diff[:, _active_features_sorted] * diff[:, _active_features_sorted]
            s = np.dot(squ, np.linspace(1, 1, len(_active_features_sorted)))
            o = np.argsort(s)
            c = o[0]

            # update the neurones weight vector (It could be vectorized)
            gamma_m = np.outer(self.gamma(c, neurons_xy_coordinate, topology_size, neighborhood_size, learn_rate),
                               np.linspace(1, 1, features_number))

            # It is controlling the boundary nodes movements
            if not fix_boundary:
                neurons = neurons - diff * gamma_m
            else:
                neurons[self.unboundary_neurons_indexes, :] = neurons[self.unboundary_neurons_indexes, :] - \
                                                              diff[self.unboundary_neurons_indexes] * \
                                                              gamma_m[self.unboundary_neurons_indexes]

            self.animation[epoch + 1, :, :] = neurons

            self.neurons_energy.append(self.mean_distance_closest_unit(self.data.iloc[:, _active_features_sorted],
                                                                       neurons[:, _active_features_sorted]))
        if _plot:
            self.plot_training_progress(log=False)
        self.neurons = neurons

    def geodesic_inertia_to_panel(self):
        """Converts the geodesic inertia dict to pandas data frame

        Returns:
            A pandas data panel that any panel includes the information about one inertia.
        """
        inertia_frames = {}
        supports_frames = {}
        for inertia_item_key, inertia_item_value in self.geodesic_inertia_data.items():
            supports_frames.update({inertia_item_key: inertia_item_value.get('support')})
            inertia_frames.update({inertia_item_key: inertia_item_value.get('lift')})
        geodesic_lift_panel = pd.concat(inertia_frames.values(), keys=list(inertia_frames.keys()))
        geodesic_support_df = pd.DataFrame(supports_frames)
        return geodesic_lift_panel, geodesic_support_df

    @staticmethod
    def get_adjacency_matrix(input_matrix, normalizing=False, norm='l2'):
        """Computes the distance matrix.

        Args:
            input_matrix (numpy): Includes the input data
            normalizing (boolean): If True, the data is normalized before computing the distance matrix.
            norm (str): Indicate The norm to use to normalize each non-zero sample (‘l1’, ‘l2’, or ‘max’)

        Returns:
            A ndarray includes the distance between the neurons.

        """
        if normalizing:
            input_matrix = normalize(input_matrix, axis=0, norm=norm)
        distance_matrix = euclidean_distances(input_matrix, input_matrix)
        return distance_matrix

    def plot_concave_hull(self, points, inter_points=False, fill=True, diagonals_x=None, diagonals_y=None,
                          diagonals_direction=[0, 0], x_label=None, y_label=None, title='SOM Concave Hull',
                          predefined_alpha=0.2, optimal_alpha_value=True, geodesic_paths=False):

        """ Plot the concave hall boundaries

        Args:
            diagonals_direction (binary list): Indicates the direction of the arrows (bottom to top)
            diagonals_y (list): Includes the primary and secondary diagonals Y coordinates
            diagonals_x (list): Includes the primary and secondary diagonals X coordinates
            points (ndarray): Includes the data points.
            inter_points (boolean): If True, the internal points are plotted too.
            fill (boolean): If True, the Alpha shape is filled by the color.
            x_label (str): Indicates the plot x-axis label
            y_label (str): Indicates the plot y-axis label
            title (str): Indicates the plot title
            predefined_alpha (float): Indicates the concave hull radian.
            optimal_alpha_value (boolean): If True, alpha parameter can be solved automatically,
                                           but with large datasets this can take a long time to calculate.
            geodesic_paths (boolean): If True, the geodesic paths are plotted too.
        Returns:

        Note:
            - Alpha shape: https://pypi.org/project/alphashape/
        """

        # Todo: estimate the optimal alpha value.
        def get_alpha_shape(point_set, alpha=None, only_outer=True):
            # Fixme: The optimal value of the alpha has to be selected then this function is useful.
            """Compute the alpha shape (concave hull) of a set of points.

            Args:
                point_set (ndarray): np.array of shape (n,2) points.
                alpha (float): alpha value.
                only_outer (boolean): boolean value to specify if we keep only the outer border or also inner edges.
            Returns:
                A set of (i,j) pairs representing edges of the alpha-shape. (i,j) are the indices in the points array.
                A list of the outer points.
            Notes:
                See also www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle

            """

            assert point_set.shape[0] > 3, "At least 4 data points are needed"

            def add_edge(edge_set, _source, _target):
                """
                Add an edge between the i-th and j-th points, if not in the list already.

                Args:
                    edge_set (set): Indicates a set of edges in graph
                    _source (int): Indicates a source node to add a new edge.
                    _target (int): Indicates a target node to add a new edge.

                """
                if (_source, _target) in edge_set or (_target, _source) in edge_set:
                    assert (_target, _source) in edge_set, "Can't go twice over same directed edge right?"
                    if only_outer:
                        # if both neighboring triangles are in shape, it's not a boundary edge
                        edge_set.remove((_target, _source))
                    return
                edge_set.add((_source, _target))

            tri = Delaunay(point_set)
            edges_subset = set()
            neuron_chain = []
            # Loop over triangles to compute the edges set.
            # <ia, ib, ic> is indices of corner points of the triangle
            for ia, ib, ic in tri.vertices:
                pa = point_set[ia]
                pb = point_set[ib]
                pc = point_set[ic]
                # Computing radius of triangle circumcircle
                a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
                b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
                c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
                s = (a + b + c) / 2.0
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                circumcircle_r = a * b * c / (4.0 * area)
                if circumcircle_r < alpha:
                    add_edge(edges_subset, ia, ib)
                    add_edge(edges_subset, ib, ic)
                    add_edge(edges_subset, ic, ia)

            # Compute the neuron chain
            for edge_item in edges_subset:
                neuron_chain.append(edge_item[0])
                neuron_chain.append(edge_item[1])

            return edges_subset, neuron_chain

        # Plot
        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
        ax.set_facecolor('lightgrey')
        if fill:
            try:
                if optimal_alpha_value:
                    alpha_shape = alphashape.alphashape(points)
                else:
                    alpha_shape = alphashape.alphashape(points, predefined_alpha)
                ax.scatter(*zip(*points))
                ax.add_patch(PolygonPatch(alpha_shape, alpha=0.3))
            except (ZeroDivisionError, ValueError):
                alpha_shape = alphashape.alphashape(points, 0.)
                ax.scatter(*zip(*points))
                ax.add_patch(PolygonPatch(alpha_shape, alpha=0.3))

            # Type the neuron numbers for point_index, coordinates in enumerate(points):
            # ax.annotate(point_index, coordinates)

            # plot geodesic_paths
            if geodesic_paths:
                for point_index, coordinates in enumerate(points):
                    if point_index in self.geodesic_paths.get('primary'):
                        ax.scatter(coordinates[0], coordinates[1], marker='o', color='r', label='primary')
                        ax.annotate(point_index, coordinates)
                    elif point_index in self.geodesic_paths.get('secondary'):
                        ax.scatter(coordinates[0], coordinates[1], marker='o', color='b', label='secondary')
                        ax.annotate(point_index, coordinates)
                    else:
                        pass
                suffix = ""
            else:
                suffix = "_without_path"
        else:
            # fixme: the alph value has to be calculated.
            # Computing the alpha shape
            edges, neurons = get_alpha_shape(points, alpha=predefined_alpha, only_outer=True)
            if inter_points:
                ax.plot(points[:, 0], points[:, 1], '.', c='black')

            for i, j in edges:
                ax.plot(points[[i, j], 0], points[[i, j], 1], color='gray', linestyle='-', linewidth=1,
                        marker='o', markerfacecolor='blue', markeredgecolor='blue', markersize=7)
                ax.annotate(i, (points[i, :]))
                ax.annotate(j, (points[j, :]))

        # Arrow Direction
        ax.arrow(diagonals_x[0 + diagonals_direction[0]],
                 diagonals_y[0 + diagonals_direction[0]],
                 diagonals_x[1 - diagonals_direction[0]] - diagonals_x[0 + diagonals_direction[0]],
                 diagonals_y[1 - diagonals_direction[0]] - diagonals_y[0 + diagonals_direction[0]],
                 fc='red', ec='red',
                 length_includes_head=True, width=0.0001, head_width=0.0003)
        ax.arrow(diagonals_x[2 + diagonals_direction[1]],
                 diagonals_y[2 + diagonals_direction[1]],
                 diagonals_x[3 - diagonals_direction[1]] - diagonals_x[2 + diagonals_direction[1]],
                 diagonals_y[3 - diagonals_direction[1]] - diagonals_y[2 + diagonals_direction[1]],
                 fc='blue', ec='blue',
                 length_includes_head=True, width=0.0001, head_width=0.0003)

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.gca().set_xlim()
        plt.gca().set_ylim()
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/concave_hull_plot{}.png'.format(suffix))
        else:
            plt.show()
        plt.close()

    def normalize_inertia_vectors(self, _inertia, _round=3):
        """ Normalize the Inertia table

        Args:
            _inertia (pandas): Includes the geometrical inertia data.
            _round (int): Indicates the rounding decimal number

        Returns:
            A pandas includes the normalized inertia data.
        """
        # Todo: convert to a static function
        # Todo: test_configs the scale also
        data_min = self.data.min(axis=0)
        data_max = self.data.max(axis=0)
        for feature_item in data_max.index:
            for diagonal_item in _inertia.index:
                if diagonal_item == 'Primary_Inertia':
                    _inertia.loc[diagonal_item, feature_item] = \
                        _inertia.loc['Primary_neuron(+)', feature_item] - \
                        _inertia.loc['Primary_neuron(-)', feature_item]
                elif diagonal_item == 'Secondary_Inertia':
                    _inertia.loc[diagonal_item, feature_item] = \
                        _inertia.loc['Secondary_neuron(+)', feature_item] - \
                        _inertia.loc['Secondary_neuron(-)', feature_item]
                else:
                    _inertia.loc[diagonal_item, feature_item] = abs(
                        _inertia.loc[diagonal_item, feature_item] - data_min[feature_item]) / (
                                                                        data_max[feature_item] -
                                                                        data_min[feature_item])
        self.normalized_inertia_data = _inertia.round(_round)
        return _inertia.round(_round)

    def print_inertia(self, original_coordinate=[], _inertia=None, divergence_="euclidean",
                      normalized=True, inverse=True):
        """ Prints via terminal the main inertia.

        The report includes:
            - The definition of the original Two-dimensional space.
            - The representative point the alpha shape
            - The main inertia vectors
            - The most important feature in any direction
            - The feature important list and score

        Args:
            inverse (boolean): If True, the inverse of the similarity measurement takes to account.
            divergence_ (str): Indicates the type of inertia computation. ()
            normalized (boolean): If True, the data is normalized before the print.
            original_coordinate (list): Indicates the two coordination inertia original space
            _inertia (pandas): Includes the main inertia's neurons and vectors.

        Returns:

        """
        if divergence_.lower() == "euclidean":
            if normalized:
                _inertia = self.normalize_inertia_vectors(_inertia)
            # Space Coordinates
            print(BColors.OKGREEN + "\nThe original Two-dimensional space X: {} and Y:{}'".format(
                original_coordinate[0], original_coordinate[1]) + BColors.ENDC)

            # Main Neurons
            print(BColors.OKGREEN + "\nThe most important neurons are:" + BColors.ENDC)
            print(BColors.OKGREEN + tabulate(_inertia.drop(['Primary_Inertia', 'Secondary_Inertia'], axis=0).T,
                                             tablefmt="fancy_grid", headers="keys",
                                             numalign="center") + BColors.ENDC)

            # Main Inertia Vectors
            print(BColors.OKBLUE + "\nThe Primary and Secondary cluster Inertia Vectors are: " + BColors.ENDC)
            print(BColors.OKBLUE + tabulate(_inertia.loc[['Primary_Inertia', 'Secondary_Inertia'], :].T,
                                            tablefmt="fancy_grid", headers="keys",
                                            numalign="center") + BColors.ENDC)
        elif divergence_.lower() == "geodesic":
            idx = pd.IndexSlice
            lifts = self.geodesic_lift.loc[idx[:, ['lift']], :]
            lifts.index = lifts.index.droplevel(1)
            if inverse:
                lifts = lifts.pow(-1)

            # Inertia Vectors geodesic supports
            print(BColors.OKBLUE + "\nThe cluster Inertia's Geodesic Support Vectors: " + BColors.ENDC)
            print(BColors.OKBLUE + tabulate(self.geodesic_support.T, tablefmt="fancy_grid", headers="keys",
                                            numalign="center") + BColors.ENDC)

            # Inertia Vectors geodesic lift
            print(BColors.OKBLUE + "\nThe cluster Inertia's Geodesic Lift Vectors (1 / Similarity): " + BColors.ENDC)
            print(BColors.OKBLUE + tabulate(lifts.T, tablefmt="fancy_grid", headers="keys",
                                            numalign="center") + BColors.ENDC)

    def __get_euclidean_divergence(self, primary_start, primary_end, secondary_start, secondary_end):
        """ Computes the end_to_end (rough) divergence between two neurons.

        It computes the deviation over any feature that is aliened with the vector between two neurons

        Args:
            primary_start (int): The starting neuron of the Primary Inertia vector
            primary_end (int): The ending neuron of the Primary Inertia vector
            secondary_start (int): The starting neuron of the Secondary Inertia vector
            secondary_end (int): The ending neuron of the Secondary Inertia vector

        Returns:
            A pandas includes the both main inertia vectors divergence.
        """

        inertia_data = pd.DataFrame(np.concatenate((np.array([[primary_end],
                                                              [primary_start],
                                                              [int(secondary_start)],
                                                              [int(secondary_end)]]),
                                                    self.neurons[[primary_end,
                                                                  primary_start,
                                                                  int(secondary_start),
                                                                  int(secondary_end)], :]), axis=1),
                                    index=['Primary_neuron(-)', 'Primary_neuron(+)', 'Secondary_neuron(-)',
                                           'Secondary_neuron(+)'],
                                    columns=['diagonal_neuron'] + list(self.features_indexes.keys()))

        inertia_data.loc['Primary_Inertia'] = inertia_data.loc['Primary_neuron(+)'] - inertia_data.loc[
            'Primary_neuron(-)']

        inertia_data.loc['Secondary_Inertia'] = inertia_data.loc['Secondary_neuron(+)'] - inertia_data.loc[
            'Secondary_neuron(-)']

        inertia_data.loc[['Primary_Inertia'], ['diagonal_neuron']] = "{}--->{}".format(
            int(inertia_data.loc['Primary_neuron(-)', 'diagonal_neuron']),
            int(inertia_data.loc['Primary_neuron(+)', 'diagonal_neuron']))

        inertia_data.loc[['Secondary_Inertia'], ['diagonal_neuron']] = "{}--->{}".format(
            int(inertia_data.loc['Secondary_neuron(-)', 'diagonal_neuron']),
            int(inertia_data.loc['Secondary_neuron(+)', 'diagonal_neuron']))

        # Save as an object
        inertia_data.to_pickle("./outputs/somis/temp/inertia_data.pkl")
        self.inertia_data = inertia_data

        return inertia_data

    def get_geodesic_lift_df(self, inverse=True, normalizing=True, _norm='l2', features=None, diagonal_header=False):
        """ Computes the lift data frame

        Args:
            diagonal_header (boolean): If True, the inertia vector's neurons are is added as first row of the dataframe
            _norm (str): Indicates the norm L1 , L2, or Max
            normalizing (boolean): If True, it normalized the neurons features matrix before distance matrix.
            inverse (boolean): If True, it calculate the 1 / similarity.
            features (list): Includes the features (name/index) that are represent the clustering space.

        Returns:
            A pandas includes the inertia's lift
        """
        idx = pd.IndexSlice

        inertia_vectors = self.geodesic_lift.loc[idx[:, ['lift']], :]
        inertia_vectors.index = inertia_vectors.index.droplevel(1)

        if inverse:
            inertia_vectors = inertia_vectors.pow(-1)

        # Drop the reference features (They are represented over the inertia support)
        df = inertia_vectors.T
        if features is not None:
            df.drop(features, inplace=True)

        if normalizing:
            df = normalize_pandas(df, _axis=0, _norm=_norm)

        if diagonal_header:
            header_df = pd.DataFrame([["{}--->{}".format(
                int(self.geodesic_inertia_data.get('Primary_Inertia').get('start_neuron')),
                int(self.geodesic_inertia_data.get('Primary_Inertia').get('end_neuron'))),
                "{}--->{}".format(
                    int(self.geodesic_inertia_data.get('Secondary_Inertia').get('start_neuron')),
                    int(self.geodesic_inertia_data.get('Secondary_Inertia').get('end_neuron')))]],
                index=['diagonal_neuron'], columns=df.columns)
            df = header_df.append(df)

        return df

    def __get_geodesic_divergence(self, inertia_name, start_neuron, end_neuron, weighting_features,
                                  distances='euclidean', _norm='l2', differentials=None):
        """ Computes the geodesic divergence between two neurons.

        It computes the geodesic divergence for any feature. It calculates the gradual gradient over the shortest path
        between two neurons.

        Args:
            inertia_name (str): Indicate the name/number of the inertia
            differentials (str): If Ture, the absolute values differentials for any jump via S.Path will be used.
                - 'None': The chain of original values of features
                - 'diff': The chain of first order of differentiation
                - 'diff_abc': The chain of absolute first order of differentiation
                - 'diff_pro': The chain of absolute first order of differential
                - 'diff_pro_abs': The chain of absolute first order of differential
            _norm (str): Indicates the neurones data normalization method.
            distances (str): Indicates the similarity distance calculation
                - cosine: The cosine distance
                - euclidean: The euclidean distance
                - cosine_similarity: The cosine similarity
            start_neuron (int): The starting neuron of the Primary Inertia vector
            end_neuron (int): The ending neuron of the Primary Inertia vector
            weighting_features (str/list): Includes a features that we would like to compute the edges weight over them

        Returns:
            A pandas includes the both main inertia vectors divergence.
        """
        # compute the shortest path
        shortest_path = self.get_shortest_geodesic_path(start_neuron,
                                                        end_neuron,
                                                        weighting_features).get('shortest_path')
        # Generate the shortest path
        self.geodesic_paths.update({inertia_name: shortest_path})
        print("Shortest path:", shortest_path, ", Distances metric:", distances, ", Normalization:", _norm)

        # computing the deferential
        # Fixme: how to compute the deferential (sure about the concept of the differentials)
        if not differentials:
            shortest_path_data = self.neurons[shortest_path, :]
        elif differentials.lower() == 'diff_abc':
            shortest_path_data = np.absolute(np.diff(self.neurons[shortest_path, :], axis=0))
        elif differentials.lower() == 'diff':
            shortest_path_data = np.diff(self.neurons[shortest_path, :], axis=0)
        elif differentials.lower() == 'diff_pro':
            shortest_path_data = np.divide(np.diff(self.neurons[shortest_path, :], axis=0),
                                           self.neurons[shortest_path[:-1], :])
        elif differentials.lower() == 'diff_pro_abs':
            shortest_path_data = np.divide(np.diff(self.neurons[shortest_path, :], axis=0),
                                           self.neurons[shortest_path[:-1], :])
        else:
            raise ValueError(BColors.FAIL + "The differentials value is not valid." + BColors.ENDC)

        # Fixme: the norm problem has to be solve.
        # Normalization:
        if _norm:
            shortest_path_data = normalize(shortest_path_data, axis=0, norm=_norm)
            norm_neurons = normalize(self.neurons, axis=0, norm=_norm)
        else:
            shortest_path_data = shortest_path_data
            norm_neurons = self.neurons

        # Sequence similarity
        cos1 = self.angle_between_vectors([1, 0],
                                          norm_neurons[end_neuron, weighting_features] -
                                          norm_neurons[start_neuron, weighting_features])
        cos2 = self.angle_between_vectors([0, 1],
                                          norm_neurons[end_neuron, weighting_features] -
                                          norm_neurons[start_neuron, weighting_features])

        # TODO: the inertia wight set
        cos1_n = round(cos1 / (cos1 + cos2), 3)  # Angle with the x_axis
        cos2_n = round(cos2 / (cos1 + cos2), 3)  # Angle with the y_axis

        # Todo: use the Cosine distance similarity value to compare the changes over the active feature and others.
        # cosine_similarity / euclidean_distances
        if distances.lower() == 'euclidean':
            feature_load = pd.DataFrame(euclidean_distances(shortest_path_data.T, shortest_path_data.T),
                                        index=list(self.features_indexes.keys()),
                                        columns=list(self.features_indexes.keys()))
        elif distances.lower() == 'cosine':
            feature_load = pd.DataFrame(cosine_distances(shortest_path_data.T, shortest_path_data.T),
                                        index=list(self.features_indexes.keys()),
                                        columns=list(self.features_indexes.keys()))
        elif distances.lower() == 'cosine_similarity':
            feature_load = pd.DataFrame(cosine_similarity(shortest_path_data.T, shortest_path_data.T),
                                        index=list(self.features_indexes.keys()),
                                        columns=list(self.features_indexes.keys()))

        # Similarity between any feature and the clustering feature set. (Directional similarity)
        feature_load.loc['lift'] = feature_load.loc[self.get_feature_name(weighting_features[1])] * cos2_n + \
                                   feature_load.loc[self.get_feature_name(weighting_features[0])] * cos1_n

        return {"support": dict(zip(self.get_feature_name(weighting_features), [cos1_n, cos2_n])),
                "lift": feature_load.loc[[self.get_feature_name(weighting_features[0]),
                                          self.get_feature_name(weighting_features[1]),
                                          'lift'], :],
                "start_neuron": start_neuron,
                "end_neuron": end_neuron}

    def get_inertia(self, features=None, alpha=0.001, plot_concave_hull=True, optimal_alpha_value=False,
                    print_table=False, normalized=True, _norm='l2', by='distance', divergence='euclidean'):
        """ Computes the main data inertia.

        It finds the two sub set of the neurons. In any sub set, there are two neurons that have the maximum distance
        from each other and from the neurons in others sub set. The distance are calculated in 2D space
        represented by features parameter.

        Args:
            optimal_alpha_value (boolean): If True, alpha parameter can be solved automatically, but with large datasets
                                           this can take a long time to calculate.
            by (str): Indicate the method that will use in order to compute the second component:
                - distance_angle: First, it finds the point from both plus and minos points. Then,it finds second point.
                  Where the vector between first and second point has the max angle with the first component.
                - angel:
                - cosine_Similarity:
                - distance:

            _norm (str): Indicates the norm L1 , L2, or Max
            normalized (boolean): If True, it normalized the neurons features matrix before distance matrix.
            print_table (boolean): If True, it prints out the inertia table
            plot_concave_hull (boolean): If True, it plots down the inertia table
            alpha (float): Indicate the alpha value in order to compute the concave hull.
            features (list): Includes the features (name/index) that are represent the clustering space.
            divergence (str): Indicates how to compute the shortest path between two nodes of any inertia vector.
                              It accepts:
                                - "euclidean": Compute a rough deviation between the star and end neurones.
                                - "geodesic": Compute the gradient over the shortest path between star and end neurones.
        Returns:
            A pandas includes the Primary and Secondary inertia vectors and associated neurons.
        """
        assert len(features) > 1, "The feature list is empty."

        # Find the major inertia
        features = self.get_feature_indexes(features)
        distance_matrix = self.get_adjacency_matrix(self.neurons[:, features], normalizing=normalized, norm=_norm)
        cosine_matrix = cosine_similarity(self.neurons[:, features], self.neurons[:, features])

        neuron_plus, neuron_minus = list(np.unravel_index(distance_matrix.argmax(), distance_matrix.shape))
        secondary_diagonal_alternatives = list(set(range(self.x_dim * self.y_dim)).difference({neuron_plus,
                                                                                               neuron_minus}))
        topsis_matrix = pd.DataFrame(list(itertools.combinations(secondary_diagonal_alternatives, 2)),
                                     columns=['neuron_x', 'neuron_y'])

        topsis_matrix[['distance_x_minus',
                       'distance_x_plus',
                       'distance_y_minus',
                       'distance_y_plus',
                       'distance_x_y',
                       'similarity_x_minus',
                       'similarity_x_plus',
                       'similarity_y_minus',
                       'similarity_y_plus',
                       'similarity_x_y']] = topsis_matrix.apply(
            lambda row: [distance_matrix[row.neuron_x, neuron_minus],
                         distance_matrix[row.neuron_x, neuron_plus],
                         distance_matrix[row.neuron_y, neuron_minus],
                         distance_matrix[row.neuron_y, neuron_plus],
                         distance_matrix[row.neuron_x, row.neuron_y],
                         cosine_matrix[row.neuron_x, neuron_minus],
                         cosine_matrix[row.neuron_x, neuron_plus],
                         cosine_matrix[row.neuron_y, neuron_minus],
                         cosine_matrix[row.neuron_y, neuron_plus],
                         cosine_matrix[row.neuron_x, row.neuron_y]], axis=1, result_type="expand")

        if by == 'distance':
            topsis_matrix['sum_distance_x'] = topsis_matrix['distance_x_minus'] + topsis_matrix['distance_x_plus']
            aux_neuron_1 = int(topsis_matrix.loc[topsis_matrix['sum_distance_x'].idxmax(), ['neuron_x']])

            # Defined the search space for the auxiliary_2
            sub_topsis_matrix = topsis_matrix[
                (topsis_matrix['neuron_x'] == aux_neuron_1) | (topsis_matrix['neuron_y'] == aux_neuron_1)]
            sub_topsis_matrix['aux_neuron_2'] = sub_topsis_matrix.apply(
                lambda row: int(row.neuron_y) if row.neuron_x == aux_neuron_1 else int(row.neuron_x), axis=1)
            sub_topsis_matrix['auxiliary_distance_plus'] = sub_topsis_matrix.apply(
                lambda row: row.distance_y_plus if row.neuron_x == aux_neuron_1 else row.distance_x_plus, axis=1)
            sub_topsis_matrix['auxiliary_distance_minus'] = sub_topsis_matrix.apply(
                lambda row: row.distance_y_minus if row.neuron_x == aux_neuron_1 else row.distance_x_minus, axis=1)

            sub_topsis_matrix['min_plus_minus'] = sub_topsis_matrix[['auxiliary_distance_plus',
                                                                     'auxiliary_distance_minus']].min(axis=1)
            sub_topsis_matrix['max_plus_minus'] = sub_topsis_matrix[['auxiliary_distance_plus',
                                                                     'auxiliary_distance_minus']].max(axis=1)

            # Fixme: It is needed to find a better measurement in order to select the second aux node.
            sub_topsis_matrix['sum_distance_x'] = sub_topsis_matrix['auxiliary_distance_plus'] + \
                                                  sub_topsis_matrix['auxiliary_distance_minus']

            sub_topsis_matrix['sum_distance_x'] = sub_topsis_matrix['distance_x_y'] + \
                                                  sub_topsis_matrix['sum_distance_x'] * (
                                                          sub_topsis_matrix['min_plus_minus'] /
                                                          sub_topsis_matrix['max_plus_minus']) * 0.5

            aux_neuron_2 = int(sub_topsis_matrix.loc[sub_topsis_matrix['sum_distance_x'].idxmax(), ['aux_neuron_2']])
        elif by == 'cosine_Similarity':
            topsis_matrix['sum_distance_x'] = topsis_matrix['distance_x_minus'] + topsis_matrix['distance_x_plus']
            aux_neuron_1 = int(topsis_matrix.loc[topsis_matrix['sum_distance_x'].idxmax(), ['neuron_x']])

            # Defined the search space for the auxiliary_2
            sub_topsis_matrix = topsis_matrix[
                (topsis_matrix['neuron_x'] == aux_neuron_1) | (
                        topsis_matrix['neuron_y'] == aux_neuron_1)]

            sub_topsis_matrix.loc[:, 'aux_neuron_2'] = sub_topsis_matrix.apply(
                lambda row: int(row.neuron_y) if row.neuron_x == aux_neuron_1 else int(row.neuron_x), axis=1)

            sub_topsis_matrix.loc[:, 'auxiliary_similarity_plus'] = sub_topsis_matrix.apply(
                lambda row: row.distance_y_plus if row.neuron_x == aux_neuron_1 else row.similarity_x_plus, axis=1)
            sub_topsis_matrix.loc[:, 'auxiliary_similarity_minus'] = sub_topsis_matrix.apply(
                lambda row: row.distance_y_minus if row.neuron_x == aux_neuron_1 else row.similarity_x_minus, axis=1)

            sub_topsis_matrix.loc[:, 'min_plus_minus'] = sub_topsis_matrix[['auxiliary_similarity_plus',
                                                                            'auxiliary_similarity_minus']].min(axis=1)
            sub_topsis_matrix.loc[:, 'max_plus_minus'] = sub_topsis_matrix[['auxiliary_similarity_plus',
                                                                            'auxiliary_similarity_minus']].max(axis=1)

            # Fixme: It is needed to find a better measurement in order to select the second aux node.
            # Todo: use the cosin dissimilarity
            sub_topsis_matrix.loc[:, 'sum_similarity_x'] = sub_topsis_matrix['auxiliary_similarity_plus'] + \
                                                           sub_topsis_matrix['auxiliary_similarity_minus']

            sub_topsis_matrix.loc[:, 'sum_similarity_x'] = sub_topsis_matrix['similarity_x_y'] + \
                                                           sub_topsis_matrix['sum_similarity_x'] * (
                                                                   sub_topsis_matrix['min_plus_minus'] /
                                                                   sub_topsis_matrix['max_plus_minus'])

            aux_neuron_2 = int(sub_topsis_matrix.loc[sub_topsis_matrix['sum_similarity_x'].idxmin(), ['aux_neuron_2']])
        elif by == 'distance_angle':
            topsis_matrix['sum_distance_x'] = topsis_matrix['distance_x_minus'] + topsis_matrix['distance_x_plus']

            topsis_matrix['angle_between_main_components'] = topsis_matrix.apply(
                lambda row: self.angle_between_vectors(
                    self.neurons[neuron_minus, features] - self.neurons[neuron_plus, features],
                    self.neurons[int(row.neuron_x), features] - self.neurons[int(row.neuron_y), features]), axis=1)

            aux_neuron_1 = int(topsis_matrix.loc[topsis_matrix['sum_distance_x'].idxmax(), ['neuron_x']])

            # Defined the search space for the auxiliary_2
            sub_topsis_matrix = topsis_matrix[
                (topsis_matrix['neuron_x'] == aux_neuron_1) | (
                        topsis_matrix['neuron_y'] == aux_neuron_1)]

            sub_topsis_matrix['aux_neuron_2'] = sub_topsis_matrix.apply(
                lambda row: int(row.neuron_y) if row.neuron_x == aux_neuron_1 else int(row.neuron_x), axis=1)
            sub_topsis_matrix['auxiliary_distance_plus'] = sub_topsis_matrix.apply(
                lambda row: row.distance_y_plus if row.neuron_x == aux_neuron_1 else row.distance_x_plus, axis=1)
            sub_topsis_matrix['auxiliary_distance_minus'] = sub_topsis_matrix.apply(
                lambda row: row.distance_y_minus if row.neuron_x == aux_neuron_1 else row.distance_x_minus, axis=1)

            aux_neuron_2 = int(sub_topsis_matrix.loc[sub_topsis_matrix['angle_between_main_components'].idxmin(),
                                                     ['aux_neuron_2']])
        elif by == 'angle':
            # Todo: other criterion for selecting the aux nodes
            topsis_matrix['quadrilateral_perimeter'] = topsis_matrix[
                topsis_matrix.columns.difference(['neuron_x', 'neuron_y'])].sum(axis=1, skipna=True)

            topsis_matrix['quadrilateral_area'] = topsis_matrix.apply(
                lambda row: Polygon([self.neurons[neuron_minus, features],
                                     self.neurons[neuron_plus, features],
                                     self.neurons[int(row.neuron_x), features],
                                     self.neurons[int(row.neuron_y), features]]).area, axis=1)

            topsis_matrix['quadrilateral_angle_plus'] = topsis_matrix.apply(
                lambda row: vg.angle(self.neurons[[int(row.neuron_x), neuron_plus, int(row.neuron_y)], features[0]],
                                     self.neurons[[int(row.neuron_x), neuron_plus, int(row.neuron_y)], features[1]]),
                axis=1)

            topsis_matrix['quadrilateral_angle_minus'] = topsis_matrix.apply(
                lambda row: vg.angle(self.neurons[[int(row.neuron_x), neuron_minus, int(row.neuron_y)], features[0]],
                                     self.neurons[[int(row.neuron_x), neuron_minus, int(row.neuron_y)], features[1]]),
                axis=1)

            topsis_matrix['quadrilateral_angle_sum'] = topsis_matrix['quadrilateral_angle_minus'] + \
                                                       topsis_matrix['quadrilateral_angle_plus']

            aux_neuron_1, aux_neuron_2 = topsis_matrix.loc[topsis_matrix['quadrilateral_angle_sum'].idxmax(),
                                                           ['neuron_x', 'neuron_y']]
        else:
            raise ValueError(BColors.FAIL + "The second component selection measurement is not valid." + BColors.ENDC)
        # compute the divergence
        if divergence.lower() == "euclidean":
            inertia_data = self.__get_euclidean_divergence(neuron_plus, neuron_minus, aux_neuron_1, aux_neuron_2)
            # Fixme: There is a conflict between the normalize and the print table. I need to check the norm situation.
            # Print out the report
            if print_table:
                self.print_inertia(self.get_feature_name(features), inertia_data, divergence)
            else:
                # Normalize
                inertia_data = self.normalize_inertia_vectors(inertia_data)

            # Plot the concave hull
            if plot_concave_hull:
                diagonal_x = self.neurons[[neuron_minus, neuron_plus, int(aux_neuron_1),
                                           int(aux_neuron_2)], features[0]]
                diagonal_y = self.neurons[[neuron_minus, neuron_plus, int(aux_neuron_1),
                                           int(aux_neuron_2)], features[1]]

                # Set the arrows directions
                diagonal_direction = [0, 0]
                if diagonal_y[0] > diagonal_y[1]:
                    diagonal_direction[0] = 1
                if diagonal_y[2] > diagonal_y[3]:
                    diagonal_direction[1] = 1

                # plot the alpha shape of the SOM.
                self.plot_concave_hull(points=self.neurons[:, features],
                                       diagonals_x=diagonal_x,
                                       diagonals_y=diagonal_y,
                                       diagonals_direction=diagonal_direction,
                                       x_label=self.get_feature_name(features[0]),
                                       y_label=self.get_feature_name(features[1]),
                                       predefined_alpha=alpha,
                                       optimal_alpha_value=optimal_alpha_value)
            return inertia_data
        elif divergence.lower() == "geodesic":
            # Compute the inertia
            diagonal_x = self.neurons[[neuron_minus, neuron_plus, int(aux_neuron_1),
                                       int(aux_neuron_2)], features[0]]
            diagonal_y = self.neurons[[neuron_minus, neuron_plus, int(aux_neuron_1),
                                       int(aux_neuron_2)], features[1]]

            # Set the arrows directions
            diagonal_direction = [0, 0]
            if diagonal_y[0] > diagonal_y[1]:
                diagonal_direction[0] = 1
            if diagonal_y[2] > diagonal_y[3]:
                diagonal_direction[1] = 1

            primary_geodesic_inertia = self.__get_geodesic_divergence('primary',
                                                                      neuron_plus,
                                                                      neuron_minus,
                                                                      features)
            secondary_geodesic_inertia = self.__get_geodesic_divergence('secondary',
                                                                        aux_neuron_1,
                                                                        aux_neuron_2,
                                                                        features)
            # Save as an object
            self.geodesic_inertia_data.update({"Primary_Inertia": primary_geodesic_inertia,
                                               "Secondary_Inertia": secondary_geodesic_inertia})

            # Save as pandas panel
            geodesic_lift_panel, geodesic_support_df = self.geodesic_inertia_to_panel()
            self.geodesic_lift = geodesic_lift_panel
            self.geodesic_support = geodesic_support_df

            with open('./outputs/somis/temp/geodesic_inertia_data.pkl', 'wb') as handle:
                pickle.dump(self.geodesic_inertia_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if print_table:
                self.print_inertia(self.get_feature_name(features), self.geodesic_inertia_data, divergence)

            # plot the alpha shape of the SOM.
            self.plot_concave_hull(points=self.neurons[:, features],
                                   diagonals_x=diagonal_x,
                                   diagonals_y=diagonal_y,
                                   diagonals_direction=diagonal_direction,
                                   x_label=self.get_feature_name(features[0]),
                                   y_label=self.get_feature_name(features[1]),
                                   predefined_alpha=alpha,
                                   optimal_alpha_value=optimal_alpha_value,
                                   geodesic_paths=True)

            return self.get_geodesic_lift_df(normalizing=normalized,
                                             _norm=_norm,
                                             features=self.get_feature_name(features),
                                             diagonal_header=True)
        else:
            raise ValueError(BColors.FAIL + "The divergence value is not valid." + BColors.ENDC)

    def plot_inertia_drivers_slopes(self, inertia_vectors=None, inertia='primary'):
        """ Plots slopes of Organization Components

        Args:
            inertia_vectors (pandas): Includes the inertia vectors data
            inertia (str): Indicates which organization component will be plotted (All, primary or secondary)

        """

        # Fixme: It is not working with the geodesic appropriately.
        # Draw line slope line
        def draw_slope_line(_ax, _start_point, _end_point):
            """

            Args:
                _ax
                _start_point:
                _end_point:

            Returns:

            """
            _ax = plt.gca()
            l = mlines.Line2D([_start_point[0], _end_point[0]], [_start_point[1], _end_point[1]],
                              color='red' if _start_point[1] - _end_point[1] > 0 else 'green',
                              marker='o',
                              markersize=6)
            _ax.add_line(l)
            return l

        assert inertia.lower() in ['primary', 'secondary'], "plot_inertia_slopes support 'primary' & 'secondary'"

        if inertia.lower() == 'primary':
            lower_neuron = 'Primary_neuron(-)'
            upper_neuron = 'Primary_neuron(+)'

        elif inertia.lower() == 'secondary':
            lower_neuron = 'Secondary_neuron(-)'
            upper_neuron = 'Secondary_neuron(+)'
        else:
            raise ValueError("The {} vector has not been define.".format(inertia))

        if inertia_vectors is None:
            inertia_vectors = self.normalized_inertia_data

        df = inertia_vectors.T
        df = df.rename_axis('HWC').reset_index()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        neurons = df.loc[0, :]
        df.drop([0], inplace=True)

        neurons = [str('Neuron_{}'.format(neuron_item)) for neuron_item in neurons.loc[[lower_neuron,
                                                                                        upper_neuron]].astype(int)]

        fig, ax = plt.subplots(1, 1, figsize=(14, 14), dpi=100)

        # Draw Vertical Lines
        ax.vlines(x=1, ymin=0, ymax=1, color='black', alpha=0.7, linewidth=1, linestyles='dotted')
        ax.vlines(x=3, ymin=0, ymax=1, color='black', alpha=0.7, linewidth=1, linestyles='dotted')

        # Draw Points
        ax.scatter(y=df[lower_neuron], x=np.repeat(1, df.shape[0]), s=10, color='black', alpha=0.7)
        ax.scatter(y=df[upper_neuron], x=np.repeat(3, df.shape[0]), s=10, color='black', alpha=0.7)

        # Draw Line Segmented Annotation
        for start_point, end_point, c in zip(df[lower_neuron], df[upper_neuron], df['HWC']):
            draw_slope_line(ax, [1, start_point], [3, end_point])
            ax.text(1 - 0.05, start_point, c + ', ' + str(round(start_point, 2)),
                    horizontalalignment='right', verticalalignment='center', fontdict={'size': 14})
            ax.text(3 + 0.05, end_point, c + ', ' + str(round(end_point, 2)),
                    horizontalalignment='left', verticalalignment='center', fontdict={'size': 14})

        # Neuron(-) and Neuron(+) Annotations
        ax.text(1 - 0.05, 1.05, 'Neuron(-)', horizontalalignment='right', verticalalignment='center',
                fontdict={'size': 18, 'weight': 700, 'color': 'Blue'})
        ax.text(3 + 0.05, 1.05, 'Neuron(+)', horizontalalignment='left', verticalalignment='center',
                fontdict={'size': 18, 'weight': 700, 'color': 'Blue'})

        # Decoration
        ax.set_title("Comparing HWC values aliened with Cluster's \n \n {} Inertia".format(inertia.upper()),
                     horizontalalignment='center', verticalalignment='center', fontdict={'size': 18, 'weight': 700})
        ax.set(xlim=(0, 4), ylim=(-0.05, 1.1), ylabel='Normalized Value')
        ax.set_xticks([1, 3])
        ax.set_xticklabels(neurons[0:2], horizontalalignment='center', verticalalignment='top',
                           fontdict={'size': 18, 'weight': 700, 'color': 'Blue'})
        plt.yticks(np.arange(0, 1, 0.1), fontsize=16)

        # Lighten borders
        plt.gca().spines["top"].set_alpha(.0)
        plt.gca().spines["bottom"].set_alpha(.0)
        plt.gca().spines["right"].set_alpha(.0)
        plt.gca().spines["left"].set_alpha(.0)
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/{}_inertia_drivers_slopes_plot.png'.format(inertia))
        else:
            plt.show()
        plt.close()

    def plot_inertia_drivers_symmetric_bar(self, inertia_vectors=None):
        """ Bar plot the first two Organization Components Factor load

        Args:
            inertia_vectors (pandas): Includes the Organization Components

        Returns:
        """
        # Fixme: It is not working with the geodesic appropriately.
        if inertia_vectors is None:
            inertia_vectors = self.normalized_inertia_data

        df = inertia_vectors.T
        df = df.rename_axis('HWC').reset_index()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        df.drop([0], inplace=True)

        HWC = [''] + df.HWC.to_list()
        Primary_Inertia = [0] + df.Primary_Inertia.astype(float).round(2).to_list()
        Secondary_Inertia = [0] + df.Secondary_Inertia.astype(float).round(2).to_list()
        ind = np.arange(len(HWC))
        width = 0.3

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=60)
        ax.barh(ind, Primary_Inertia, width, align='center', alpha=0.7, color='r', label='OC1')
        ax.barh(ind - width, Secondary_Inertia, width, align='center', alpha=0.7, color='b', label='OC2')
        ax.set(yticks=ind - width / 2, yticklabels=HWC, ylim=[2 * width - 1, len(HWC)])

        for i, v in enumerate(Primary_Inertia):
            if i == 0:
                pass
            elif v < 0:
                ax.text(v - 0.15, i - 0.05, str(v), color='red', fontsize=12)
            else:
                ax.text(v + 0.05, i - 0.05, str(v), color='red', fontsize=12)
        for i, v in enumerate(Secondary_Inertia):
            if i == 0:
                pass
            elif v < 0:
                ax.text(v - 0.15, i - 0.4, str(v), color='blue', fontsize=12)
            else:
                ax.text(v + 0.05, i - 0.4, str(v), color='blue', fontsize=12)
        plt.grid()
        plt.xlim(-1.2, 1.2)
        plt.legend(fontsize=14)
        plt.xlabel('Normalized Feature Load')
        plt.title('The Organization Inertia Factor Load', fontsize=14, fontweight='bold')
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/inertia_drivers_hbar_plot.png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_inertia_drivers_asymmetric_bar(self, features=None, inertia_vectors=None, inverse=True,
                                            normalizing=True, _norm='l2'):
        """ Bar plot the first two Organization Components Factor lift

        Args:
            inertia_vectors (pandas): Includes the Organization Components
            inverse (boolean): If True, the inverse of the similarity measurement takes to account.
            features (list): Includes the features (name/index) that are represent the clustering space.
            normalizing (boolean): If True, the data is normalized before computing the distance matrix.
            _norm (str): Indicate The norm to use to normalize each non_zero sample (‘l1’, ‘l2’, or ‘max’)

        Returns:
            Bar Blot

        Test Data:
            Cl3
            d = {'Primary_Inertia': [0.02,0.259,0.148,0.184,0.184,0.361,0.105,0.753,0.191,0.293,0.353,0.212,0.148,0.235,0.268,0.092,0.138],
                 'Secondary_Inertia': [0.01,0.077,0.095,0.138,0.138,0.446,0.067,0.085,0.144,0.358,0.246,0.171,0.103,0.208,0.267,0.056,0.089]}
            Cl_2
            d = {'Primary_Inertia': [0.094,0.069,0.202,0.06,0.049,0.059,0.09,0.061,0.06,0.944,0.053,0.059,0.072,0.058,0.049,0.089,0.07],
                 'Secondary_Inertia': [0.135,0.238,0.131,0.219,0.216,0.182,0.149,0.1,0.182,0.697,0.203,0.187,0.167,0.191,0.203,0.145,0.162]}
            Cl1
            d = {'Primary_Inertia': [0.196,0.688,0.247,0.218,0.169,0.176,0.183,0.196,0.175,0.164,0.171,0.174,0.177,0.172,0.173,0.187,0.179],
                 'Secondary_Inertia': [0.248,0.162,0.305,0.462,0.206,0.219,0.253,0.206,0.222,0.201,0.215,0.221,0.238,0.226,0.202,0.248,0.234]}
            df = pd.DataFrame(data=d, index=df.index)
        """
        if not inertia_vectors:
            df = self.get_geodesic_lift_df(inverse=inverse, normalizing=normalizing, _norm=_norm, features=features)

        df = df.rename_axis('HWC').reset_index()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

        HWC = [''] + df.HWC.to_list()
        Primary_Inertia = [0] + df.Primary_Inertia.astype(float).round(2).to_list()
        Secondary_Inertia = [0] + df.Secondary_Inertia.astype(float).round(2).to_list()
        ind = np.arange(len(HWC))
        width = 0.3

        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=60)
        ax.barh(ind, Primary_Inertia, width, align='center', alpha=0.7, color='r', label='OC1')
        ax.barh(ind - width, Secondary_Inertia, width, align='center', alpha=0.7, color='b', label='OC1')
        ax.set(yticks=ind - width / 2, yticklabels=HWC, ylim=[2 * width - 1, len(HWC)])

        for i, v in enumerate(Primary_Inertia):
            if i == 0:
                pass
            elif v < 0:
                ax.text(v - 0.15, i - 0.05, str(v), color='red', fontsize=12)
            else:
                ax.text(v + 0.05, i - 0.05, str(v), color='red', fontsize=12)
        for i, v in enumerate(Secondary_Inertia):
            if i == 0:
                pass
            elif v < 0:
                ax.text(v - 0.15, i - 0.4, str(v), color='blue', fontsize=12)
            else:
                ax.text(v + 0.05, i - 0.4, str(v), color='blue', fontsize=12)
        plt.grid()
        plt.xlim(0, max(Primary_Inertia + Secondary_Inertia) * 1.1)
        plt.legend(fontsize=14)
        plt.xlabel('Normalized Feature Load')
        plt.title('The Organization Inertia Factor Load', fontsize=14, fontweight='bold')
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/inertia_drivers_hbar_plot.png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    # -----------------------------------------------------------------------------------------------
    # TODO
    # -----------------------------------------------------------------------------------------------

    def supper_clustering(self):
        """ Hybrid clustering """
        pass

    def inference_rough_set(self):
        """Use the <Rough set theory> to interpret the shape of the cluster """
        pass

    def plot_radar(self):
        """ Plots the radars"""
        pass

    def quiver_plot(self):
        """vector field plots

        See:
            https://matplotlib.org/gallery/images_contours_and_fields/quiver_demo.html#sphx-glr-gallery-images-contours-and-fields-quiver-demo-py
        """
        pass

    def quiver_key_plot(self):
        """vector field plots"""
        pass


def __test_me(dim=(10, 10), epo=8000, dataset='iris', _which_test=['organization_component_analysis']):
    def __som_test_iris(som_object, do_som=False):
        som_object.plot_weight_positions("sepal length (cm)", "petal width (cm)")
        som_object.embedding_index_test()
        print(som_object.neurons)
        print(som_object.get_goodness_of_fit(full_list=True))
        print(pd.DataFrame(som_object.neurons))
        print(som_object.data)
        print(som_object.neurons_association_list)
        print(m.som2networkx(_weight_features=["sepal length (cm)", "petal width (cm)"]))
        som_object.plot_init_weight_positions("sepal length (cm)", "petal width (cm)")
        som_object.plot_unified_distance_matrix()
        som_object.plot_sample_hit()
        som_object.plot_weight_planes(feature='all')
        som_object.feature_importance()
        som_object.plot_comparative_density()
        som_object.plot_neurons_guid()
        lb = som_object.self_cluster(_merg=True, merge_range=-0.5)
        som_object.plot_cluster_association("sepal length (cm)", "petal width (cm)", lb)
        som_object.plot_cluster_map(lb, labels=True)
        som_object.plot_unified_distance_matrix(merge_clusters=True, merge_range=-0.5)
        plt.show()

        print(som_object.get_intra_cluster_distances(som_object.unique_centroids['position_x'][0],
                                                     som_object.unique_centroids['position_y'][0],
                                                     som_object.centroids,
                                                     som_object.unified_distance_matrix))

        print(som_object.get_intra_clusters_distances(som_object.centroids,
                                                      som_object.unique_centroids,
                                                      som_object.unified_distance_matrix))

        print(som_object.get_inter_clusters_distance(som_object.centroids,
                                                     som_object.unique_centroids,
                                                     som_object.unified_distance_matrix))

        print(som_object.get_mean_intra_cluster_sparsity(som_object.unique_centroids['position_x'][0],
                                                         som_object.unique_centroids['position_y'][0],
                                                         som_object.unified_distance_matrix,
                                                         som_object.centroids, ))

        if do_som:
            som_object.starburst(explicit=True)  # starburst representation of clusters
            som_object.feature_importance()
            print(som_object.get_goodness_of_fit())

    def __sonis_test_iris(son_object):
        son_object.plot_unified_distance_matrix(explicit=False, merge_range=10)
        son_object.plot_sample_hit()

        # Validate the model
        for item_key, item_value in son_object.get_goodness_of_fit(full_list=True).items():
            print(item_key, ":", item_value)

        son_object.plot_weight_positions("sepal length (cm)", "petal width (cm)")

        son_object.get_inertia(["sepal length (cm)", "petal width (cm)"], divergence='geodesic',
                               print_table=True, by='distance')

        son_object.plot_inertia_drivers_asymmetric_bar()

        m.plot_unified_distance_matrix()

        print(son_object.get_goodness_of_fit(full_list=True))
        son_object.plot_neighbor_distance_hexbin()
        son_object.plot_weight_planes(feature='all')
        son_object.plot_weight_positions("sepal length (cm)", "petal width (cm)")
        son_object.feature_importance()
        son_object.plot_neurons_guid()

        # m.plot_inertia_drivers_symmetric_bar()

        # print(m.get_goodness_of_fit(full_list=True))
        # print(pd.DataFrame(m.neurons))
        # print(m.data)
        # print(m.neurons_association_list)

    def __somis_test_iris_clustering(son_object):
        son_object.plot_unified_distance_matrix(explicit=False, merge_range=10, labels=True)

    def __scatter_plot(__data, _x, _y):
        plt.figure()
        plt.plot(_x, _y, 'o', color='lime', data=__data)
        plt.xlabel(_x)
        plt.ylabel(_y)
        plt.title('Scatter plot')
        plt.show()

    if dataset == 'iris':
        from sklearn import datasets
        iris = datasets.load_iris()
        labels = iris.target
        data = pd.DataFrame(iris.data[:, :4])
        data.columns = iris.feature_names
        __scatter_plot(__data=data, _x='sepal length (cm)', _y='petal width (cm)')
    else:
        raise ValueError("The data is not valid")

    for test_item in _which_test:
        if test_item == 'som':
            m = SOM(map_shape=dim, epoch=epo, norm=None, make_jupiter_report=False,
                    training_rate=0.1, neighborhood_size=1,
                    grid_features=["petal width (cm)", "sepal length (cm)"])
            m.fit(data[["petal width (cm)", "sepal length (cm)"]], labels, init_method='rrg')
            __som_test_iris(m)
        elif test_item == 'organization_component_analysis':
            m = SOMIS(map_shape=dim, epoch=epo, norm=None, make_jupiter_report=False,
                      training_rate=0.1, neighborhood_size=1,
                      grid_features=["sepal length (cm)", "petal width (cm)"])
            m.fit(data, labels, init_method='rrg')
            __sonis_test_iris(m)
        elif test_item == 'clustering':
            m = SOMIS(map_shape=dim, epoch=epo, norm=None, make_jupiter_report=False)
            m.fit(data, labels)
            __somis_test_iris_clustering(m)


if __name__ == '__main__':
    __test_me()
