#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Self Organization Neural Network Inference System

These module includes both
    - The Self Organization Map inference System
    - The Fast version of the Self Organization Map.

The son program package includes the following main functions:
    1) model construction
    2) model evaluation
    3) model visualization is available in clusters_plot.py

See:
    https://github.com/njali2001/popsom
    https://help.rockware.com/rockworks17/WebHelp/maps_star.htm
    https://clevertap.com/blog/sankey-chart-vs-sunburst-chart/

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
last update: 24/10/2022
"""

import random
import math
import sys
import warnings
import os
import shutil
import imageio
import networkx as nx
import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sms
import statistics as stat  # F-test_configs
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from itertools import combinations, product
from random import randint
from matplotlib import cm
from matplotlib.patches import RegularPolygon
from scipy import stats  # KS Test
from scipy.spatial.distance import euclidean
from scipy.stats import f  # F-test_configs
from shapely.geometry import Polygon
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize, scale
from celluloid import Camera
from multiprocessing import Process, cpu_count
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
from tkinter import Tcl
from itertools import count
from numpy import dot  # cross
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from sklearn.cluster import DBSCAN
from numpy.linalg import norm
from sklearn import metrics
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex, to_rgba
from matplotlib.patches import Rectangle
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from kneed import KneeLocator
from mpl_toolkits.mplot3d import axes3d
from sklearn.metrics import silhouette_samples, silhouette_score
from kavica.utils._bcolors import BColors
from kavica.utils._plots import (MARKER, PARAVER_STATES_COLOR, get_color_map, get_color_palette, palette_size,
                                 get_listed_cmap)
from kavica.utils._progress_bar import progressbar
from kavica.utils._util import normalize_pandas, record_log
from scipy.spatial import distance as scipy_spatial

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize, linewidth=200)
random.seed(4)
np.random.seed(4)
plt.rcParams.update({'font.size': 16})

PARAVER_STATES_COLOR_PALETTE = get_color_palette(PARAVER_STATES_COLOR, None, True)

__all__ = [
    'SOM',
    'get_rrg'
]


def get_rrg(bound, size, vector_size, margin=0.2, grid_features=[0, 1]):
    """ Computes a multidimensional Regular Rectangular Grid based on the boundaries of dimensions

    Args:
        margin (float): Proportional margin of the grid from the boundary data points
        bound (2-D array_like, of length 2 x vector_size): Includes the upper and lower bound of any feature
        vector_size (int): Sample vector size
        size (int): Number of the sample
        grid_features (list): A list of features to describe the RRG structure

    Returns:
        ndarray in shape of (size, vector_size)
    """

    # Fixme: it generate a grid with the swapped features x,y
    def __mesh_grid(permutation_list):
        """ Creates the mesh_grid from ndarray

        Args:
            permutation_list (ndarray): Includes the list of the maker point in each axis

        Returns:
            ndarray
        """
        return np.array(np.meshgrid(*permutation_list.T)).T.reshape(-1, permutation_list.shape[1])

    _columns_map = dict(zip(bound.columns, range(len(bound.columns))))
    _bound = bound.copy()

    dim = int(math.sqrt(size))
    permutations = np.zeros((dim, len(grid_features)))
    compulsory_bound = bound.drop(grid_features, axis='columns')
    _compulsory_columns = list(compulsory_bound.columns)
    _compulsory_indexes = sorted(list(map(_columns_map.get, _compulsory_columns)))

    for item, feature_item in enumerate(bound[grid_features].T.iterrows()):
        bound = feature_item[1].loc['max'] - feature_item[1].loc['min']
        lower_bound = feature_item[1].loc['min'] - (bound * margin)
        upper_bound = feature_item[1].loc['max'] + (bound * margin)
        permutations[:, item] = np.linspace(lower_bound, upper_bound, endpoint=True, num=dim)

    grid_base = __mesh_grid(permutations)

    # add the non grid features
    for _comp_item in _compulsory_indexes:
        _zero_column = np.full((1, grid_base.shape[0]), _bound.ix['mean', _comp_item])
        grid_base = np.insert(grid_base, _comp_item, _zero_column, axis=1)

    return grid_base


def get_random_perturbation_around_mean(c, size, cov=None):
    """ Random perturbation around mean values

    It initializes the cluster centers by slightly perturbing the mean or the global centroid

    Args:
        c (1D array, of length N): Indicates the mean center vector
        cov (2D array, of shape (N, N)): Covariance matrix of the distribution.
                                         It must be symmetric and positive-semi-definite for proper sampling
        size (int): Output number

    Returns:
        ndarray in shape of (size, c)
    """
    if cov is None:
        num_features = len(c)
        cov = np.zeros((num_features, num_features), float)
        np.fill_diagonal(cov, 1)
    print(BColors.OKGREEN + "\n Random perturbation around {} is initiated SOM".format(c.tolist()) + BColors.ENDC)
    return np.random.multivariate_normal(c, cov, size)


def get_random_component(bound, size, vector_size):
    """ Random component
    Each component of a reference vector is chosen randomly from the range of values observed in the data.

    Args:
        bound (2-D array_like, of length 2 x vector_size): Includes the upper and lower bound of any feature
        vector_size (int): Sample vector size
        size (int): Number of the sample

    Returns:
        ndarray in shape of (size, vector_size)
    """
    vector = np.zeros((size, vector_size))
    for item, feature_item in enumerate(bound.T.iterrows()):
        vector[:, item] = np.random.uniform(feature_item[1].loc['min'], feature_item[1].loc['max'], size)
    return vector


class SOM(object):
    """ Base class of the Self Organizing Map.

    Parameters
    ----------
    map_shape (tuple):
        A tuple indicates the map size. For example, (8, 8) will create
        a 8 * 8 map with 64 neurons.
    training_rate (float):
        The training rate, should be a positive non-zero real number
    epoch (int):
        The training iterations number
    som_method (str) :
        There is two alternative algorithm the som and som_f
    initializer (str): function  #TODO
        A function which takes in the input data and weight matrix and Returns
        an initialized weight matrix. The initializers are defined in
        somber.components.initializers. Can be set to None.
    scaled (boolean): initialized scaled instance  #TODO
        An initialized instance of scaled() which is used to scale the data
        to have mean 0 and SD 1.
    normalize (boolean):
        normalize the input data space
    """

    def __init__(self, map_shape=(5, 5), training_rate=0.1, neighborhood_size=1, epoch=1000, norm=None, scaled=False,
                 make_jupiter_report=True, grid_features=None, lr_decay=.00001, radius_decay=.00001):
        """ Initialize the Model

        Args:
            map_shape:
            training_rate:
            epoch:
            norm (str):
                The norm to use to normalize each non-zero sample (or each non-zero feature if axis is 0).
                ‘l1’, ‘l2’, or ‘max’, 'None' optional (‘None’ by default)
            scaled:
        """

        def __validate_parameters():
            assert norm in ['l1', 'l2', 'max', None], "The norm {} is not defined".format(norm)
            assert map_shape >= (3, 3), "The map size {} is too small".format(map_shape)

        if not norm:
            norm = None

        __validate_parameters()

        self.x_dim = map_shape[0]
        self.y_dim = map_shape[1]
        self.boundary_neurons_indexes = self.get_boundary(map_shape[0])
        self.unboundary_neurons_indexes = self.get_unboundary(map_shape[0])
        self.training_rate = training_rate
        self.epoch = epoch
        self.norm = norm
        self.scaled = scaled
        self.data = None
        self.neurons_cluster_id = None
        self.labels = None
        self.features_indexes = None
        self.neurons_association_list = []
        self.features_goodness_of_fit = None
        self.distance_matrix = None
        self.unified_distance_matrix = None
        self.centroids = None
        self.centroids = None
        self.unique_centroids = None
        self.neurons_energy = []
        self.animation = None
        self.neurons = None
        self.make_jupiter_report = make_jupiter_report
        self.graph_weighting_features = None
        self.graph = None
        self.initial_neurons_wight = None
        self.animation_temp_path = 'outputs/animation_temp/'
        self.neurons_labels = None
        self.cluster_plot_cmap = 'tab20'
        self._figsize = (16, 16)
        self._dpi = 100
        self.hyper_parameters = {}  # Initial state of training algorithm parameters
        self.grid_dim = 2  # 2D or 3D grid
        self.grid_features = grid_features
        self.neighborhood_size = neighborhood_size
        self.lr_decay = lr_decay
        self.radius_decay = radius_decay

    # TODO: adjusting with the non labeled data
    def fit(self, data, labels=None, _plot=True, fix_boundary=False, init_method='rs'):
        """Train the Model.

        Args:
            data (pandas df): includes the data
            labels (pandas df): includes the label of any instance in dataset
            _plot (boolean): If True, the model training progress will be plotted
            fix_boundary (boolean): If True, the boundary nodes wil not updated during training
            init_method (str): indicates the initialization method
                - rpaz: random perturbation around zero vector
                - rpam: random perturbation around mean values
                - rc: random component
                - rs: random selection
                - rrg: regular rectangular grid (our method I)
                - irrg: irregular rectangular grid(our method II)
                - pca_rrg: linear projection method with regular rectangular grid
                - pca_irrg: linear projection method with irregular rectangular grid
                - kmean: K_mean approach
        Returns:
            Self
        """

        if self.scaled:
            data.loc[:, :] = scale(data)
        elif self.norm:
            data.loc[:, :] = normalize(data, norm=self.norm)
        else:
            pass
        self.data = data
        self.labels = labels
        self.features_indexes = dict(zip(list(self.data.columns), range(self.data.shape[1])))

        # Initialize the animation matrix
        self.animation = np.zeros((self.epoch + 1, self.x_dim * self.y_dim, self.data.shape[1]))

        self._train_som(_plot=_plot, fix_boundary=fix_boundary, _init_method=init_method)
        self._associate_to_neurons()

        return self

    def _associate_to_neurons(self):
        """ Computes the neurons association of each data point

        Returns:
            A list indicates which training data is associated with each of the neurons
        """
        for index, row in self.data.iterrows():
            b = self.get_associated_neuron(row)
            self.neurons_association_list.extend([b])
        return self.neurons_association_list

    def get_bmu(self, data_point, bmu_number=3):
        """ Computes the N first BMUs of a data point

        Returns:
            A list indicates which BMUs are associated with a single data point
        """
        return self.get_associated_neuron(data_point, full_list=True)[0:bmu_number]

    def get_bmus(self, bmu_number=1):
        """ Computes the N first BMUs for each data point

        Returns:
            A list of list indicates which BMUs are associated with a single data point
        """
        _bmus = np.empty((0, bmu_number), int)
        for index, row in self.data.iterrows():
            b = self.get_associated_neuron(row, full_list=True)[0:bmu_number]
            _bmus = np.append(_bmus, np.array([b]), axis=0)
        return _bmus

    def get_neuron_direct_neighbours(self, neuron, star_topology=False):
        """ Computes the topological neighbors of a neuron.

        Args:
            neuron (int): The neuron index in the map.
            star_topology (boolean): If true the map has star topology
        Returns:
            A list of neighbors indexes
        """
        return np.nonzero(self.get_direct_neighbours_matrix(self.x_dim, self.y_dim, star_flag=star_topology)[neuron])[0]

    def __initialize_neurons(self, _method='rs', _features_number=None):
        """ Set the initial position of any neuron

        Args:
            _method (str): indicates the initialization method:
                - rpaz: random perturbation around zero vector
                - rpam: random perturbation around mean values
                - rc: random component
                - rs: random selection
                - rrg: regular rectangular grid (our method I)
                - irrg: irregular rectangular grid(our method II) (todo)
                - pca_rrg: linear projection method with regular rectangular grid (todo)
                - pca_irrg: linear projection method with irregular rectangular grid (todo)
                - kmean: K_mean approach (todo)

        Returns:
            A numpy array where each row represents a neuron, each column represents a dimension.
        """
        # Todo: It is needed to compute the neurone optimal initial states.
        # Todo: Principal component initialization and Kmean are alternative method.
        assert _method in ['rpaz', 'rpam', 'rc', 'rs', 'rrg'], "Error: the initialization method does not exist"
        if _method in ['kmean', 'pca_rrg', 'irrg', 'pca_irrg']:
            raise ValueError("Error: the init method has not been defined yet.")

        # Indicates the number of the neurones
        topology_size = self.x_dim * self.y_dim
        instances_number, features_number = self.data.shape
        feature_bound = self.data.describe().loc[['min', 'max', 'mean']]

        # Neurons holders
        if _method == 'rpaz':
            vector = np.random.uniform(-1, 1, topology_size * features_number)
            init_vectors = np.transpose(np.reshape(vector, (features_number, topology_size)))
        elif _method == 'rpam':
            c = np.mean(self.data, axis=0)
            init_vectors = get_random_perturbation_around_mean(c, topology_size)
        elif _method == 'rc':
            init_vectors = get_random_component(feature_bound, topology_size, features_number)
        elif _method == 'rs':
            init_vectors = self.data.sample(n=topology_size, replace=True).to_numpy()
        elif _method == 'rrg':
            init_vectors = get_rrg(feature_bound, topology_size, features_number,
                                   margin=-0.1, grid_features=self.grid_features)

        self.initial_neurons_wight = init_vectors.copy()
        self.animation[0, :, :] = init_vectors
        return init_vectors

    def initialize_network(self, method=None):
        """ Initializes the SOM network

        Args:
            method (str): indicates the initialization method:
                - rpaz: random perturbation around zero vector
                - rpam: random perturbation around mean values
                - rc: random component
                - rs: random selection
                - rrg: regular rectangular grid (our method I)
                - irrg: irregular rectangular grid(our method II) (todo)
                - pca_rrg: linear projection method with regular rectangular grid (todo)
                - pca_irrg: linear projection method with irregular rectangular grid (todo)
                - kmean: K_mean approach (todo)

        Returns:
            instances_number,
            features_number,
            topology_size,
            neurons,
            neurons_xy_coordinate,
            neighborhood_epoch,
            neighborhood_size
        """
        # The vector size of the neurons and instances are equal.
        instances_number, features_number = self.data.shape
        topology_size = self.x_dim * self.y_dim

        # Compute the initial neighborhood size and epoch
        neighborhood_size = self.neighborhood_size  # max(self.x_dim, self.y_dim) + 1
        neighborhood_epoch = np.ceil(self.epoch / neighborhood_size)

        neurons = self.__initialize_neurons(_method=method)

        # Constants for the Gamma function
        neurons_xy_coordinate = np.matrix.transpose(
            self._get_neurons_xy_coordinates(range(0, topology_size)).reshape(2, topology_size))

        self.hyper_parameters = {'neighborhood_epoch': neighborhood_epoch,
                                 'neighborhood_size': neighborhood_size,
                                 'training_rate': self.training_rate}
        return (instances_number, features_number, topology_size, neurons,
                neurons_xy_coordinate, neighborhood_epoch, neighborhood_size)

    def gamma(self, c, neurons_xy_coordinates, topology_size, neighborhood_size, training_rate):
        """ Computes neighborhood of a neuron

        Args:
            neurons_xy_coordinates (ndarray 2d/3d): Includes the neurons xyz coordinates
            topology_size (int): Indicates the number of the neurons
            c (int): Indicates a neuron index
            neighborhood_size (float): Indicates the neuron neighborhood size

        Returns:
            A list of the neighborhood size
        """
        # TODO: it has to vectorized
        # lookup the 2D map coordinate for c
        c_xy_coordinate = neurons_xy_coordinates[c,]

        # a matrix with each row equal to c 2D
        c_xy_coordinates = np.outer(np.linspace(1, 1, topology_size), c_xy_coordinate)

        # distance vector of each neuron from c in terms of map coordinates!
        neuron_distance_vector = np.sqrt(np.dot((c_xy_coordinates - neurons_xy_coordinates) ** 2, [1, 1]))

        # if m on the grid is in neigh then alpha else 0.0
        neighborhood = np.where(neuron_distance_vector < neighborhood_size * 1.5, training_rate, 0.0)
        return neighborhood

    def _get_neurons_xy_coordinates(self, row_index):
        """ Converts a 1D row_index into a xy map coordinate

        Args:
            row_index (list): Includes the neurones sequential number. (sorted)

        Returns:
            A numpy array where any element is [xi,yi]
        """
        x = np.array(row_index) % self.x_dim
        y = np.array(row_index) // self.x_dim
        return np.concatenate((x, y))

    @staticmethod
    def mean_distance_closest_unit(x, y):
        """Mean distance to the closest unit

        Considering the rows of X (and Y=X) as vectors, compute the distance matrix between each pair of vectors.
        The computing the mean of the minimum distance of any vector.

        Args:
            x (numpy): Includes the data points (vectors)
            y (numpy): Includes the data units (vectors)

        Returns:
            A float indicate the mean of min distances.

        Note: The first parameter (x) have to be the instances and the y is unites
        """
        return np.array(euclidean_distances(x, y), dtype=float).min(1).mean()

    def _train_som(self, _plot=True, fix_boundary=False, _init_method=None):
        """ The stochastic SOM training algorithm

        Args:
            _plot (boolean): If True, the model training progress will be plotted
            fix_boundary (boolean): If True, the boundary nodes wil not updated during training
            _init_method (str): indicates the initialization method
        """
        # fixme: It has a problem with the feature wight plot (when training rate in high)
        # Todo: The fix boundary needs test_configs

        # Initiate the neurons
        instances_number, features_number, topology_size, neurons, neurons_xy_coordinate, \
        neighborhood_epoch, neighborhood_size = self.initialize_network(method=_init_method)

        self.neurons_energy.append(self.mean_distance_closest_unit(self.data, neurons))
        step_counter = 0  # counts the number of epochs per neighborhood_epoch

        # training epoch by epoch
        for epoch in range(self.epoch):
            # Todo: Add the training progress data
            """
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
            # TODO: it have to be vectorized
            xk_m = np.outer(np.linspace(1, 1, topology_size), random_vector)
            diff = neurons - xk_m
            squ = diff * diff
            s = np.dot(squ, np.linspace(1, 1, features_number))
            o = np.argsort(s)
            c = o[0]

            # update the neurones weight vector
            # TODO: it have to be vectorized
            gamma_m = np.outer(self.gamma(c, neurons_xy_coordinate, topology_size, neighborhood_size, learn_rate),
                               np.linspace(1, 1, features_number))

            # It is controlling the boundary nodes movements
            if not fix_boundary:
                neurons = neurons - diff * gamma_m
            else:
                neurons[self.unboundary_neurons_indexes] = neurons[self.unboundary_neurons_indexes] - \
                                                           diff[self.unboundary_neurons_indexes] * \
                                                           gamma_m[self.unboundary_neurons_indexes]

            self.animation[epoch + 1, :, :] = neurons

            self.neurons_energy.append(self.mean_distance_closest_unit(self.data, neurons))
        if _plot:
            self.plot_training_progress(log=False)
        self.neurons = neurons

    def plot_comparative_density(self, feature='all'):
        """ Computes and draws kernel density estimate.

       It is a smoothed version of the histogram. This is a useful alternative to the histogram for continuous data
       that comes from an underlying smooth distribution both the neurons and data.

        Args:
            feature (str or list of string): Indicates the feature name (by default set to all )

        Returns:
            Plot the density.
        """
        if feature == 'all':
            grid_size = (1, len(list(self.data)))
            fig = plt.figure(figsize=(16, 10))
            for feature_item in range(0, grid_size[1]):
                train = np.array(self.data)[:, feature_item]
                neurons = self.neurons[:, feature_item]
                vars()['ax{}'.format(feature_item)] = plt.subplot2grid(grid_size, (0, feature_item))
                sns.kdeplot(np.ravel(train), label="training data", shade=True, color="g",
                            ax=vars()['ax{}'.format(feature_item)])
                sns.kdeplot(neurons, label="neurons", shade=True, color="b",
                            ax=vars()['ax{}'.format(feature_item)])
                vars()['ax{}'.format(feature_item)].set_ylabel('Density')
                vars()['ax{}'.format(feature_item)].set_xlabel(list(self.data)[feature_item])
        elif type(feature) == str and feature in list(self.data):
            feature_index = list(self.data).index(feature)
            train = np.array(self.data)[:, feature_index]
            neurons = self.neurons[:, feature_index]
            plt.ylabel('Density')
            plt.xlabel(feature)
            sns.kdeplot(np.ravel(train), label="training data", shade=True, color="g")
            sns.kdeplot(neurons, label="neurons", shade=True, color="b")
            plt.legend(fontsize=15)
        elif type(feature) == int and len(list(self.data)) > feature >= 0:
            feature_index = feature
            train = np.array(self.data)[:, feature_index]
            neurons = self.neurons[:, feature_index]
            plt.ylabel('Density')
            plt.xlabel(list(self.data)[feature])
            sns.kdeplot(np.ravel(train), label="training data", shade=True, color="g")
            sns.kdeplot(neurons, label="neurons", shade=True, color="b")
            plt.legend(fontsize=15)
        elif type(feature) == list:
            grid_size = (1, len(feature))
            fig = plt.figure(figsize=(16, 10))
            fig_counter = 0
            for feature_item in feature:
                if type(feature_item) == int and len(list(self.data)) > feature_item >= 0:
                    feature_index = feature_item
                    feature_name = list(self.data)[feature_item]
                    train = np.array(self.data)[:, feature_index]
                    neurons = self.neurons[:, feature_index]
                elif type(feature_item) == str and feature_item in list(self.data):
                    feature_index = list(self.data).index(feature_item)
                    feature_name = feature_item
                    train = np.array(self.data)[:, feature_index]
                    neurons = self.neurons[:, feature_index]
                vars()['ax{}'.format(feature_item)] = plt.subplot2grid(grid_size, (0, fig_counter))
                sns.kdeplot(np.ravel(train), label="training data", shade=True, color="g",
                            ax=vars()['ax{}'.format(feature_item)])
                sns.kdeplot(neurons, label="neurons", shade=True, color="b",
                            ax=vars()['ax{}'.format(feature_item)])
                vars()['ax{}'.format(feature_item)].set_ylabel('Density')
                vars()['ax{}'.format(feature_item)].set_xlabel(feature_name)
                fig_counter = fig_counter + 1
        else:
            raise ValueError("The feature name/index is not the name of a training data frame dimension or index")
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/comparative_density_plot.png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()

    def get_goodness_of_fit(self, conf_int=.95, k=30, full_list=False, ks=True):
        """ Computes the goodness of fit for the SOM map.

        The variance of the training data that is captured by the SOM model.
        The goodness of fit measurement includes two items, likelihood level and topology accuracy.
        It should be higher than 90%. But the researcher should make a decision about the precise level.

        Args:
            conf_int (float): Indicates the confidence interval of the quality.
            k (int): Indicates the sample size that will be used in order to estimate the mean accuracy.
            full_list (boolean): If True, it returns both the goodness of fit and topology accuracy else their means
            ks (boolean):b If True, the kolgomorov_smirnov_test is applied else the embedding_index_test.

        Returns:
            A dict/float that indicates teh mean accuracy values of the map.
        """
        if ks:
            likelihood_test = self.kolgomorov_smirnov_test(conf_int, verb=False)
        else:
            likelihood_test = self.embedding_index_test(conf_int, verb=False)

        topology_accuracy = self.get_mean_topology_accuracy(k, conf_int, full_list=False, ci=True)

        if full_list:
            return {"likelihood_test": likelihood_test, "topology_accuracy": topology_accuracy}
        else:
            return 0.5 * likelihood_test + 0.5 * topology_accuracy.get('accuracy')

    def kolgomorov_smirnov_test(self, confidence_interval=0.95, verb=False):
        """ kolgomorov-smirnov test_configs

        It applies kolmogorov_Smirnov Test in order to check the whether that neurons and training both are drawing
        out of the same distribution or not.

        Args:
            confidence_interval (float): Indicates the confidence interval of the test_configs
            verb (bool): If True, it will return the variance that is preserved (for any feature) by SOM map.

        Returns:
            A float value that indicates the overall model goodness of fit based on the kolmogorov_Smirnov Test
        """
        # Initialize the self.feature goodness of fit.
        self.features_goodness_of_fit = pd.DataFrame(0, index=self.data.columns, columns=['statistic', 'p-value'])

        for i in range(self.data.shape[1]):
            self.features_goodness_of_fit.loc[i, 'statistic'], self.features_goodness_of_fit.loc[
                i, 'p-value'] = stats.mstats.ks_2samp(self.neurons[:, i], self.data.iloc[:, i])

        feature_importance_list = self.feature_importance(graphics=False)
        test_results = pd.concat([feature_importance_list, self.features_goodness_of_fit], axis=1, sort=False)

        # Variance captured by converged features
        if verb:
            test_results['perc'][test_results['p-value'] <= (1 - confidence_interval)] = 0
            return test_results['perc']
        else:
            return test_results[test_results['p-value'] > (1 - confidence_interval)]['perc'].sum()

    def f_test(self, df1, df2, conf=0.95):
        """ Computes F_test over two data frame column wise

        Args:
            df1 (pandas): Includes a data set
            df2 (pandas): Includes a data set
            conf (float): Indicates the confidence interval [0.0,1.0]

        Returns:
            A pandas includes the test_configs result
        """

        def __f_test(x, y, conf_level=conf):
            """ F_test kernel

            Args:
                x (pandas series): Includes a data set
                y (pandas series): Includes a data set
                conf_level (float): Indicates the confidence interval [0.0,1.0]

            Returns:
                Three float includes statistic_estimation, lower_limit, and upper_limit
            """
            x_degree_of_freedom = len(x) - 1
            y_degree_of_freedom = len(y) - 1
            statistic_estimation = stat.variance(x.tolist()) / stat.variance(y.tolist())
            beat_value = (1 - conf_level) / 2
            lower_limit = statistic_estimation / f.ppf(1 - beat_value, x_degree_of_freedom, y_degree_of_freedom)
            upper_limit = statistic_estimation / f.ppf(beat_value, x_degree_of_freedom, y_degree_of_freedom)
            return statistic_estimation, lower_limit, upper_limit

        assert df1.shape[1] == df2.shape[1], "Those data frames do not have the same number of features"

        # Initialize the self.feature goodness of fit.
        f_test_result = pd.DataFrame(0, index=self.data.columns, columns=['f_statistic',
                                                                          'f_lower_limit',
                                                                          'f_upper_limit'])

        # compute the F-test_configs on each feature in our populations
        for feature_item in range(df1.shape[1]):
            f_test_result.iloc[feature_item, :] = __f_test(df1.ix[:, feature_item],
                                                           df2.ix[:, feature_item],
                                                           conf_level=conf)

        return f_test_result

    def t_test(self, df1, df2, conf=0.95):
        """Compute t_test over two data frame column wise

        Args:
            df1 (pandas): Includes a data set
            df2 (pandas): Includes a data set
            conf (float): Indicates the confidence interval

        Returns:
            A pandas includes the test_configs result
        """

        def __t_test(x, y, conf_level=0.95):
            cm = sms.CompareMeans(sms.DescrStatsW(x), sms.DescrStatsW(y))
            lower_limit = cm.tconfint_diff(alpha=1 - conf_level, usevar='unequal')[0]
            upper_limit = cm.tconfint_diff(alpha=1 - conf_level, usevar='unequal')[1]
            return np.mean(x), np.mean(y), lower_limit, upper_limit

        assert df1.shape[1] == df2.shape[1], "Those data frames do not have the same number of features"

        # Initialize the self.feature goodness of fit.
        t_test_result = pd.DataFrame(0, index=self.data.columns, columns=['t_statistic_1',
                                                                          't_statistic_2',
                                                                          't_lower_limit',
                                                                          't_upper_limit'])

        # compute the F-test_configs on each feature in our populations
        for feature_item in range(df1.shape[1]):
            t_test_result.iloc[feature_item, :] = __t_test(df1.ix[:, feature_item],
                                                           df2.ix[:, feature_item],
                                                           conf_level=conf)

        # return a list with the ratios and conf intervals for each feature
        return t_test_result

    def embedding_index_test(self, conf_interval=.95, verb=False):
        """ mean variance test_configs

        It applies f_test and t_test simultaneously in order to check the whether that neurons and
        training both are drawing out of the same distribution or not.

        Args:
            conf_interval (float): Indicates the confidence interval of the test_configs
            verb (bool): If True, it will return the amount of the variance tha is preserved (for any feature) by SOM map.

        Returns:
            A float value that indicates the overall model goodness of fit based on the t_test & f_test.
        """
        # Todo: it is not an stable test_configs
        __f_test = self.f_test(pd.DataFrame(self.neurons, columns=self.data.columns), self.data, conf_interval)
        __t_test = self.t_test(pd.DataFrame(self.neurons, columns=self.data.columns), self.data, conf_interval)

        __results = pd.concat([self.feature_importance(graphics=False), __f_test, __t_test], axis=1, sort=False)

        __results['overall_statistic'] = 0
        __results['overall_statistic'][(__results['f_upper_limit'] >= 1.0) & (__results['f_lower_limit'] <= 1.0) &
                                       (__results['t_upper_limit'] >= 0.0) & (__results['t_lower_limit'] <= 0.0)] = 1

        # Variance captured by converged features

        if verb:
            __results['perc'][__results['overall_statistic'] == 0.0] = 0.0
            return __results['perc']
        else:
            return __results[__results['overall_statistic'] == 1.0]['perc'].sum()

    def get_mean_topology_accuracy(self, k=30, conf_int=.95, full_list=False, ci=True):
        """ Estimate the mean topological accuracy

        It does the sampling, and calculates the SOM topology accuracy over the samples.
        Then, it estimates the SOM mean accuracy.

        Args:
            k (int): Indicates the sampling size.
            conf_int (float): Indicates the confidence interval
            full_list (boolean): If True, it returns a list that includes the accuracy of all the samples
            ci (boolean): If True, the confidence interval will compute.

        Returns:
            A list/float/dict includes the mean topology accuracy.
        """
        # fixme: It needs to compute optimal k
        if k > self.data.shape[0]:
            k = self.data.shape[0]
            warnings.warn(BColors.WARNING +
                          "The sample bag size is bigger than the total number of the instances." +
                          BColors.ENDC)

        # Sampling
        test_subset = pd.DataFrame(self.data.sample(n=k).index.values.tolist(), columns=['row_index'])
        test_subset['accuracy'] = test_subset.apply(
            lambda row: self.get_topology_accuracy(self.data.iloc[row.row_index - 1], row.row_index), axis=1)

        if full_list:
            return test_subset['accuracy']
        else:
            overall_accuracy = test_subset['accuracy'].mean()
            if ci:
                confidence_interval = self.get_confidence_interval(conf_int, test_subset['accuracy'])
                return {'accuracy': overall_accuracy,
                        'lower_bound': round(confidence_interval['lower'], 3),
                        'upper_bound': round(confidence_interval['upper'], 3)}
            else:
                return {'accuracy': overall_accuracy,
                        'lower_bound': None,
                        'upper_bound': None}

    @staticmethod
    def get_confidence_interval(conf, data):
        """Computes the confidence interval for a given data

        Args:
            conf (float): Indicates the confidence level
            data (list): Includes the values that we would like to compute the mean confidence interval over them.

        Returns:
            A dict includes the lower and upper bound of the confidence interval.
        """
        _confs = sms.DescrStatsW(data).tconfint_mean(alpha=conf)
        return {'lower': 0 if _confs[0] < 0 else _confs[0], 'upper': 1 if _confs[0] > 1 else _confs[1]}

    def get_topology_accuracy(self, test_sample, instance_index, _mode='run'):
        """ Computes the topology accuracy.

        It computes the topology accuracy over a sample. Where it will be 1 for the well-structured SOM topology
        and 0 for the SOM topology.
        If SOM topology is accurate, the first two associated neurons will be topological neighbors.
        And, their distance should in [1, 1.42]

        Args:
            test_sample (pandas): Includes a test_configs data sample
            instance_index (int): Indicates an instance index that is just used for robustness test_configs.
            _mode (str): Indicates the mode that the SOM run:
                dev: developing mode and print some extera logs
                run: runtime mode and the log is not printed out

        Returns:
            An int, it will be 1 if the accuracy test_configs is passed and 0 for failed.
        """
        # Fixme: the is a incompatibility with the G calculation in FSCM module (may be rewrite function)
        # robustness test_configs
        distance_list = self.get_associated_neuron(test_sample, full_list=True)
        associated_neuron_coordinates = self.get_neuron_coordinate(distance_list[0])
        associated_neuron_coordinates_map = self.get_neuron_coordinate(
            self.neurons_association_list[instance_index - 1])

        if (associated_neuron_coordinates != associated_neuron_coordinates_map or distance_list[0] !=
                self.neurons_association_list[instance_index - 1]) and _mode == 'dev':
            print("BMU:{} <> Map_index:{}".format(distance_list[0], self.neurons_association_list[instance_index - 1]))

        if euclidean(self.get_neuron_coordinate(distance_list[0]), self.get_neuron_coordinate(distance_list[1])) < 2:
            return 1
        else:
            return 0

    def get_associated_neuron(self, instance, full_list=False, distance='euclidean'):
        """ Computes the must similar neuron.

        It computes the distance of the instance over all neurons and returns the closest ones' index.

        Args:
            instance (array): Indicates an instance vector.
            full_list (boolean): If True, it returns a sorted list of the neurons
            distance (str): Indicates the distance measurement.
        Returns:
            A list/int that indicates the neuron index number.
        """
        # Todo: the geodesic distance should be applied as well.
        distance_list = np.apply_along_axis(euclidean, 1, self.neurons, instance)
        sorted_distance_list = np.argsort(distance_list)
        if full_list:
            return sorted_distance_list
        else:
            return sorted_distance_list[0]

    def feature_importance(self, graphics=True, method='max_variance'):
        """ Computes the feature important and plot.

        Args:
            graphics (boolean): If true, it draws the feature important bar plot.
            method (str): Indicate the method for calculating the feature importance.
            method (str): Indicate the method for calculating the feature importance.

        Returns:
            A array of feature importance score.
        """
        # TODO: Add more methods (MCFS,LS,RIFS,SECT,PFS),
        method = method.lower()
        assert method in ['max_variance'], "The feature importance method {} is not defined.".format(method)

        if method == 'max_variance':
            feature_importance = pd.DataFrame(self.data.var(), columns=['score'])
            feature_importance['perc'] = feature_importance['score'] / feature_importance['score'].sum()

        # plot the significance
        if graphics:
            plt.figure()
            plt.title("Feature importance")
            plt.barh(feature_importance.index, feature_importance['perc'], color="r", align="center")
            plt.xlabel("% {} score".format(method))
            plt.ylabel("Feature Name")
            if self.make_jupiter_report:
                plt.savefig('OCA_Report/pic/feature_importance_plot.png', bbox_inches='tight')
                plt.close()
            else:
                plt.show()
                plt.close()
        else:
            return feature_importance

    def get_label_associated_neuron(self):
        """ Prints the association of labels with map elements

        Returns:
            A dataframe containing the projection onto the map for each observation
        """
        if self.labels is not None:
            association_list = pd.DataFrame(self.labels, columns=['labels'])
            association_list['row_index'] = association_list.index
            association_list[['x_coordinate', 'y_coordinate']] = association_list.apply(
                lambda row: pd.Series(self.get_neuron_coordinate(self.neurons_association_list[row.row_index])), axis=1)
            association_list = association_list.drop(['row_index'], axis=1)
            return association_list
        else:
            raise ValueError("The label list is empty.")

    def get_neuron(self, x, y):
        """ Returns a neuron at (x,y) on the map as a vector

        Args:
            x (int): Indicates the x-coordinate of neuron
            y (int): Indicates the y-coordinate of neuron

        Returns:
            A vector includes the feature loads.
        """
        return self.neurons[self.get_ordinal_index(x, y)]

    def get_neuron_coordinate(self, ordinal_index):
        """ Compute the xy coordinate of the ordinal neuron in topological map

        Args:
            ordinal_index (int): Indicates the neuron index number

        Returns:
            An array includes the neuron coordinates.
        """
        # fixme: the position of x and y seems to be swapped. check before change
        return [ordinal_index % self.x_dim, ordinal_index // self.x_dim]

    def get_ordinal_index(self, x, y):
        """ Computes the ordinal index of a neuron from the topological coordinates

        Args:
            x (int): Indicates the x-coordinate of neuron
            y (int): Indicates the y-coordinate of neuron

        Returns:
            An int that indicates the ordinal index of a neuron.

         """
        return x + y * self.x_dim

    @staticmethod
    def exponential_covariance(x1, x2, bandwidth=2, p=2):
        """ Exponential covariance function

        C(d)= exp(−(d / V) ^ p) is a stationary covariance function with smooth sample paths.
        where V is a scaling parameter, and d = d(x,y) is the distance between two points. Sample paths of a Gaussian
        process with the exponential covariance function are not smooth and p is the exponential time.

        Args:
            x1 (array 2d): Indiactes the mesh
            x2 (array 1d): Indicates the cell center
            bandwidth (float): Indicates the smoothing bandwidth
            p (float): Indicates the exponential time

        Returns:
            A pandas includes the cov data.
        """
        # TODO: check the original formulation.
        # TODO: Matérn covariance function should be available as an alternative.
        # TODO: Rational quadratic covariance function should be available as an alternative.
        assert bandwidth > 1, "The bandwidth have to be higher than one."
        return np.exp(-(euclidean_distances(x1 * (1 / bandwidth), x2 * (1 / bandwidth)) ** p))

    def get_fft_weight(self, x_dim, y_dim, bandwidth=2, dx=1, dy=1):
        """ Computes FFT of smoothing kernel object.

        Args:
            x_dim (int): Indicates the map's x dimension
            y_dim (int): Indicates the map's x dimension
            bandwidth (float): Indicates the smoothing bandwidth
            dx (float): Indicate the distance unit value over the x-axis
            dy (float): Indicate the distance unit value over the y-axis

        Returns:
            A dict includes the wgth object.
        """
        M = 2 * x_dim
        N = 2 * y_dim
        xg = np.column_stack(([*range(M)] * N, np.repeat([*range(N)], M)))

        # Fixme: it is just used for the validation and should be deleted in the last version
        if True:
            xg1 = []
            for i in range(N):
                for j in range(M):
                    xg1.extend([[j, i]])
            xg1 = np.array(xg)
            if np.array_equal(xg, xg1):
                pass
            else:
                raise IndexError("the weight matrix construction error.")

        mid_cell = np.array([[int(dx * M / 2 - 1), int((dy * N) / 2 - 1)]])
        exponential_covariance_matrix = self.exponential_covariance(xg, mid_cell, bandwidth=bandwidth)
        exponential_covariance_matrix = np.matrix.transpose(np.reshape(exponential_covariance_matrix, (N, M)))
        normalizing_matrix = np.zeros((M, N))
        normalizing_matrix[int(M / 2 - 1)][int(N / 2 - 1)] = 1
        _weight = np.fft.fft2(exponential_covariance_matrix) / (np.fft.fft2(normalizing_matrix) * M * N)
        return {"m": x_dim, "n": y_dim, "N": N, "M": M, "wght": _weight}

    def kernel_smoother(self, X, _weight=None, bandwidth=None):
        """ Kernel Smoother Function.

        Takes an image matrix and applies a kernel smoother to it. Missing values are handled using the
        Nadaraya/Watson normalization of the kernel.

        Args:
            X (ndarray): A matrix image. Missing values can be indicated by Nones.
            _weight (object): Fast Fourier Transform (FFT) of smoothing kernel. If this is NULL the default is
                              to compute this object.
            bandwidth (float): The bandwidth

        Returns:
            A ndarray includes the smooth image matrix.
        """
        if _weight is None:
            _weight = self.get_fft_weight(self.x_dim, self.y_dim, bandwidth, 1, 1)

        normalized_fft = np.zeros((_weight.get('M'), _weight.get('N')))
        normalized_fft[0:_weight.get('m'), 0:_weight.get('n')] = X
        normalized_fft = np.fft.ifft2(np.fft.fft2(normalized_fft) * _weight.get('wght')).real[0:_weight.get('m'),
                         0:_weight.get('n')]

        unifier_matrix = np.zeros((_weight.get('M'), _weight.get('N')))
        unifier_matrix[0:_weight.get('m'), 0:_weight.get('n')] = [[1] * self.y_dim] * self.x_dim
        unifier_matrix = np.fft.ifft2(np.fft.fft2(unifier_matrix) * _weight.get('wght')).real[0:_weight.get('m'),
                         0:_weight.get('n')]

        return normalized_fft / unifier_matrix

    def plot_unified_distance_matrix(self, explicit=False, smoothness=2, merge_clusters=False,
                                     merge_range=.25, labels=False):
        """ Visualizes the unified distance matrix.

        Args:
            explicit (boolean): controls the shape of the connected componentes
            smoothness (float): controls the smoothing level of the umat (NULL,0,>0)
            merge_clusters (boolean): a switch that controls if the clusters are merged together
            merge_range (float): a range that is used as a percentage of a certain distance in the code to determine
                                whether components are closer to their centroids or centroids closer to each other.
            labels (boolean): If True, the neurons labels are plotted at the center of any cell.

        Returns:

        """
        self.unified_distance_matrix = self.get_unified_distance_matrix(smoothness=smoothness)
        self.plot_heatmap(self.unified_distance_matrix, explicit=explicit, connect_components=True,
                          merge=merge_clusters, merge_range=merge_range, labels=labels)

    def get_distance_matrix(self):
        """Computes the distance matrix.

        Set the self distance to the values.
        Returns:
            A ndarray includes the distance between the neurons.
        """
        self.distance_matrix = euclidean_distances(self.neurons, self.neurons)
        return self.distance_matrix

    def get_unified_distance_matrix(self, smoothness=None):
        """Computes the unified distance matrix.

        The U-matrix (unified distance matrix) is a representation of a self-organizing map (SOM) where the Euclidean
        distance between the neurons. This is used to visualize the data in a high-dimensional space using a 2D plot.

        Args:
            smoothness (float): Indicates the smoothness of the unified distance matrix.

        Returns:
            A 2d array represents the unified distance matrix.
        """
        return self.get_heat_values(self.get_distance_matrix(), smoothness)

    @staticmethod
    def get_direct_neighbours_matrix(size_x, size_y, star_flag=False):
        """ Computes the direct neighbour's matrix(map)

        We have matrix nxn and a neighboring matrix (n*n)x(n*n), it computes a (n*n)x(n*n) binary matrix.
        If the jth cell is a direct neighbor of ith cell it will set to one else zero.

        Args:
            size_y (int): Indicates the matrix columns number
            size_x (int): Indicates the matrix rows number.
            star_flag (boolean): If True, The star connection to neighbours will be assumed

        Returns:
            A binary ndarray includes the direct neighbors map.
        """

        # Fixme: The star_flag should be checked again
        def __neighbours(cell, star=star_flag):
            """ Computes the neighbours of a cell.

            Args:
                cell (tuple): Indicates the x,y coordinates of the a original matrix cell.
                star (boolean): If True, The star connection to neighbours will be assumed

            Returns:
                A list of tuple includes all the direct neighbours of a cell.
            """
            if star:
                for _c in product(*(range(n - 1, n + 2) for n in cell)):
                    if _c != cell and 0 <= _c[0] < size_x and 0 <= _c[1] < size_y:
                        yield _c
            else:
                for _c in product(*(range(n - 1, n + 2) for n in cell)):
                    if _c != cell and 0 <= _c[0] < size_x and 0 <= _c[1] < size_y and \
                            math.sqrt(((_c[0] - cell[0]) ** 2 + (_c[1] - cell[1]) ** 2)) == 1:
                        yield _c

        neighbour_map = np.zeros((size_x * size_y, size_x * size_y))
        for row_item in range(0, size_x):
            for column_item in range(0, size_y):
                for neighbour_item in list(__neighbours((row_item, column_item))):
                    neighbour_map[column_item + row_item * size_y][neighbour_item[1] + neighbour_item[0] * size_y] = 1

        return neighbour_map

    def get_heat_values(self, d, smoothness=None):
        """Calculates the heat map values.

            Those values are used in order to represent the distance matrix.

        Args:
            d (ndarray): Includes the distance matrix.
            smoothness (float): Indicates the smoothness of the unified distance matrix.

        Returns:
            A ndarray includes the heat values of the map.
        """
        assert self.x_dim > 1 and self.x_dim > 1, "The map dimensions have to be higher than one"
        # Fixme: The star_flag should be checked again
        direct_neighbours_map = self.get_direct_neighbours_matrix(self.x_dim, self.y_dim, star_flag=False)
        heat_values = np.reshape(np.sum(d * direct_neighbours_map, axis=1) / np.sum(direct_neighbours_map, axis=1),
                                 (self.x_dim, self.y_dim)).transpose()

        if smoothness is not None:
            if smoothness == 0:
                smooth_heat_values = self.kernel_smoother(heat_values)
                return smooth_heat_values
            elif smoothness > 0:
                smooth_heat_values = self.kernel_smoother(heat_values, bandwidth=smoothness)
                return smooth_heat_values
            else:
                raise ValueError("The kernel smoother function parameter have to be in [0,+inf)")
        else:
            return heat_values

    def plot_heatmap(self, heat_values, explicit=True, connect_components=True, merge=True,
                     merge_range=4, labels=False):
        """ Plots a heat map

        The plot also contains the connected components of the map based on the landscape of the heat map.

        Args:
            heat_values (ndarray): Includes the heat map values for any cell.
            explicit (boolean): If True, the centroid is connected to the neuron explicitly
            connect_components (boolean): If True, the centroid connection to the neurons will be plotted.
            merge (boolean): If True, the star_bursts are merged
            merge_range (float): Indicates a range that is used as a percentage of a certain distance in the code to
                           determine whether components are closer to their centroids or centroids closer to each other.
            labels (boolean): If True, the neurons labels are plotted at the center of any cell.

        Note:
            The heat_value array is transposed before plotting and calculating the centroids,
            in order to have the same orientation over all plot function in SOM package.
        """
        # TODO: add an option to choose the cell shape (Square or Hexagon).
        assert self.x_dim > 1 and self.y_dim > 1, "Topology map has too small dimensions, x={} and y={}.".format(
            self.x_dim, self.y_dim)

        def draw_connecting_lines(merge_clusters):
            """Draws the connected component lines over the neurones map.

            Args:
                merge_clusters (boolean): controls whether we merge the star_bursts together.

            Returns:
            """
            if not merge_clusters:
                # find the centroid for each neuron on the map
                centroids = self.get_centroids(heat_values, explicit)
            else:
                # find the unique centroids for the neurons on the map
                centroids = self.get_merged_clusters(heat_values, explicit, merge_range)
            self.centroids = centroids  # set the class parameter
            self.unique_centroids = self.get_unique_centroids(self.centroids)  # set the class parameter

            # connect each neuron to its centroid
            for ix in range(self.x_dim):
                for iy in range(self.y_dim):
                    cx = centroids['centroid_x'][ix, iy]
                    cy = centroids['centroid_y'][ix, iy]
                    plt.plot([ix + 0.5, cx + 0.5],
                             [iy + 0.5, cy + 0.5],
                             color='blue',
                             linestyle='-',
                             linewidth=1.0)

        # Todo: I need to check it all the function in the son that use the same orientation.
        heat_values = heat_values.T  # I use it in order to uniform both hexbin plots and heatmap plot (matrix)
        color_template, _ = pd.cut(heat_values.flatten(), bins=100, labels=False, retbins=True)
        color_template = np.array(np.ndarray.transpose(np.reshape(color_template, heat_values.shape)))

        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)
        mesh = ax.pcolor(color_template, cmap=plt.cm.hot.reversed())
        ax.set_xticks(np.arange(self.x_dim) + 0.5, minor=False)
        ax.set_yticks(np.arange(self.y_dim) + 0.5, minor=False)
        ax.set_xticklabels(np.arange(self.x_dim), minor=False)
        ax.set_yticklabels(np.arange(self.y_dim), minor=False)

        if connect_components:
            draw_connecting_lines(merge)
        if labels:
            self.__draw_labels()

        plt.title('Supper Cluster Starburst Matrix')
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/heatmap_plot.png', bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()

    @staticmethod
    def get_unique_centroids(centroids, label=False):
        """ Computes a list of unique centroids.

        Args:
            centroids (ndarray): Includes the centroids coordinates of any neuron.
            label (boolean): If true, a cluster label will be returned
        Returns:
            A dict includes unique centroids.
        """
        unique_centroids = np.stack((np.array(centroids['centroid_x']).flatten().tolist(),
                                     np.array(centroids['centroid_y']).flatten().tolist()), axis=-1)
        unique_centroid = np.unique(unique_centroids, axis=0)
        if not label:
            return {"position_x": unique_centroid[:, 0].tolist(), "position_y": unique_centroid[:, 1].tolist()}
        else:
            return dict(enumerate(unique_centroid.tolist(), 1))

    def get_intra_clusters_distances(self, centroids, unique_centroids, unified_distance_matrix):
        """ Computes the intra_clusters distances (for all clusters).

        It computes the distance of all individuals from the centroid cluster by cluster.

        Args:
            unique_centroids (dict): Indicates the centroids coordinates
            centroids (ndarray): Includes the x, y coordinates of all neurons' centroids
            unified_distance_matrix (ndarray): Indicates the distance matrix.

        Returns:
            A dict of dict includes the centroids coordinates as key and distances as values.
        """

        cluster_intra_distances = {}
        for centroid_x, centroid_y in zip(unique_centroids['position_x'], unique_centroids['position_y']):
            cluster_intra_distances.update({(centroid_x,
                                             centroid_y): self.get_intra_cluster_distances(centroid_x,
                                                                                           centroid_y,
                                                                                           centroids,
                                                                                           unified_distance_matrix)})
        return cluster_intra_distances

    def get_intra_cluster_distances(self, centroid_x, centroid_y, centroids, unified_distance_matrix):
        """Computes the intra_cluster distances.

        Computes the distance of all individuals from the centroid.

        Args:
            centroid_x (int): Indicates the centroid (neuron) x coordinate
            centroid_y (int): Indicates the centroid (neuron) y coordinate
            centroids (ndarray): Includes the x, y coordinates of all neurons' centroids
            unified_distance_matrix (ndarray): Indicates the distance matrix.

        Returns:
            A dict includes the neuron coordinates as key and distance as value.
        """
        intra_distances = {}
        for xi in range(self.x_dim):
            for yi in range(self.y_dim):
                if centroids['centroid_x'][xi, yi] == centroid_x and centroids['centroid_y'][xi, yi] == centroid_y:
                    intra_distances.update({(xi, yi): unified_distance_matrix[xi, yi]})

        return intra_distances

    def get_mean_intra_clusters_sparsity(self, centroids, unique_centroids, unified_distance_matrix):
        """ Computes the mean intra_cluster sparsity (for all clusters).

        Args:
            unique_centroids (dict): Indicates the centroids coordinates
            centroids (ndarray): Includes the x, y coordinates of all neurons' centroids
            unified_distance_matrix (ndarray): Indicates the distance matrix.

        Returns:
            A dict of dict includes the centroids coordinates as key and sparsity as value.
        """
        mean_distance = {}
        for centroid_x, centroid_y in zip(unique_centroids['position_x'], unique_centroids['position_y']):
            distance = self.get_mean_intra_cluster_sparsity(centroid_x, centroid_y, unified_distance_matrix, centroids)
            mean_distance.update({(centroid_x, centroid_y): distance})
        return mean_distance

    def get_mean_intra_cluster_sparsity(self, centroid_x, centroid_y, unified_distance_matrix, centroids):
        """ Computes the mean intra_cluster sparsity (for one cluster).

        Args:
            centroid_x (int): Indicates the centroid (neuron) x coordinate
            centroid_y (int): Indicates the centroid (neuron) y coordinate
            centroids (ndarray): Includes the x, y coordinates of all neurons' centroids
            unified_distance_matrix (ndarray): Indicates the distance matrix.

        Returns:
            A float indicates the intra cluster mean sparsity.
        """
        cumulative_wight = 0
        elements = 0
        centroid_weight = unified_distance_matrix[centroid_x, centroid_y]
        for xi in range(self.x_dim):
            for yi in range(self.y_dim):
                if centroids['centroid_x'][xi, yi] == centroid_x and centroids['centroid_y'][xi, yi] == centroid_y:
                    cumulative_wight = cumulative_wight + abs(unified_distance_matrix[xi, yi] - centroid_weight)
                    elements = elements + 1
        return cumulative_wight / elements

    def get_feature_index(self, feature_name):
        """ Returns the index number of a feature name (of the dataset)

        Args:
            feature_name (str): indicates the feature name

        Returns:
            An int indicates the feature ordinal index.
        """
        return self.features_indexes.get(feature_name)

    def get_feature_indexes(self, features):
        """ Computes list of feature indexes.

        Args:
            features (list): Includes the features (name/index)

        Returns:
            A list of int includes the features indexes.
        """
        if features == 'all':
            return list(range(0, self.data.shape[1]))
        else:
            for feature_index, feature_item in enumerate(features):
                if isinstance(feature_item, str):
                    features[feature_index] = self.features_indexes.get(feature_item)
        return features

    @staticmethod
    def get_boundary(size):
        """ Computes the boundary nodes indexes in the SOM mesh

        Args:
            size (int): The mesh is a square and "size" is the  length of one of its sides.

        Returns:
            A list of the boundary nodes indexes
        """
        boundary_indexes = []
        if size <= 1:
            raise ValueError(BColors.FAIL + "The mesh size has to be =>2" + BColors.ENDC)
        elif size == 2:
            boundary_indexes = [0, 1, 2, 3]
        else:
            boundary_indexes.extend(range(0, size - 1))
            for said_iterator in range(1, size):
                boundary_indexes.extend([size * said_iterator - 1, size * said_iterator])
            boundary_indexes.extend(range(size * (size - 1) + 1, size ** 2))
        return boundary_indexes

    @staticmethod
    def get_boundary_sides(size):
        """ Computes the boundary nodes indexes in the SOM mesh per each side (left,top,right,bottom, corners)

        Args:
            size (int): The mesh is a square and "size" is the  length of one of its sides.

        Returns:
            A dict of the boundary nodes indexes keys in [lef,top,right,bottom]
        """
        if size <= 1:
            raise ValueError(BColors.FAIL + "The mesh size has to be =>2" + BColors.ENDC)
        elif size == 2:
            boundary_lattices = {"corners": [0, 1, 2, 3]}
        else:
            _corners = [0, size - 1, size * (size - 1), (size ** 2) - 1]
            top_side = []
            bottom_side = []
            left_side = list(range(1, size - 1))
            for said_iterator in range(1, size - 1):
                top_side.append(size * (said_iterator + 1) - 1)
                bottom_side.append(size * said_iterator)
            right_side = list(range(size * (size - 1) + 1, size ** 2 - 1))
            boundary_lattices = {"left": left_side, "top": top_side, "right": right_side,
                                 "bottom": bottom_side, "corners": _corners}
        return boundary_lattices

    @staticmethod
    def get_unboundary(size):
        """ Computes the unboundary nodes indexes in the SOM mesh

        Args:
            size (int): The mesh is a square and "size" is the  length of one of its sides.

        Returns:
            A list of the boundary nodes indexes
        """
        return list(set(range(0, size ** 2)).difference(SOM.get_boundary(size)))

    def _scatter(self, _data, _x='X', _y='Y', _c=None, _s=40, _drop=False):
        """ Adds a scatter plot as layer to a plot

        Args:
            _data (ndarray): The dataset
            _x (str): The x-axis feature name
            _y (str): The y-axis feature name
            _c (str): A list of the labels or a feature name
            _drop (boolean): If True It drops out of cell limits data points
                             (Note: Only applicable for the backward transformation)
            _s (int): Indicates the data points size in scatter plot

        Returns:
        """
        _df = pd.DataFrame(data=_data, columns=[_x, _y])
        if isinstance(_c, str):
            _df['lab'] = _data[_c]
        elif _c is None:
            pass
        else:
            _df['lab'] = _c

        if _drop:  # drop out of the rang (limits)
            _df = self._drop_out_of_range(_df)
        _df.replace([np.inf, -np.inf], np.nan, inplace=True)
        _df.dropna(inplace=True)

        if _c is not None:
            # fixme: if the label includes the result of the DBSCAN which has noise data
            if -1 in _df['lab'].unique():
                _palette = PARAVER_STATES_COLOR_PALETTE[0:palette_size(_df['lab'])]
            else:
                _palette = PARAVER_STATES_COLOR_PALETTE[1:palette_size(_df['lab']) + 1]
            ax = sns.scatterplot(_x, _y, marker='o', palette=_palette, s=_s, data=_df, hue='lab', edgecolor='k')
        else:
            ax = sns.scatterplot(_x, _y, marker='o', color='lime', s=_s, data=_df, edgecolor='k')
        return ax

    def _draw_init_som(self, ax, _x, _y, _nodes=False):
        """ Draws the self organizing map topological graph as a layer over the plat

        Args:
            _x (str): The x-axis feature name
            _y (str): The y-axis feature name
            _nodes (boolean): If True, it draws the neurones

        Returns:
        """

        direct_neighbours = np.argwhere(self.get_direct_neighbours_matrix(self.x_dim,
                                                                          self.y_dim,
                                                                          star_flag=False) == 1)
        # Fixme: The star_flag should be checked again
        for neurons_connection in direct_neighbours:
            ax.plot(self.initial_neurons_wight[neurons_connection, self.get_feature_index(_x)],
                    self.initial_neurons_wight[neurons_connection, self.get_feature_index(_y)],
                    'r-', linewidth=0.4)
        return ax

    def _draw_init_neuron_number(self, ax, _x, _y):
        """ Draws the neuron number as a layer over the plat

        Args:
            _x (str): The x-axis feature name
            _y (str): The y-axis feature name

        Returns:
        """
        for neuron_item in range(self.x_dim * self.y_dim):
            ax.annotate(neuron_item, (self.initial_neurons_wight[neuron_item, self.get_feature_index(_x)],
                                      self.initial_neurons_wight[neuron_item, self.get_feature_index(_y)]), c='b')
        return ax

    def _init_boundary_scatter(self, _x, _y):
        """ Draws the initial position of boundary neurons scatter a layer over the pot

        Args:
            _x (str): The x-axis feature name
            _y (str): The y-axis feature name

        Returns:
        """
        boundary_indexes = self.get_boundary(self.x_dim)
        ax = sns.scatterplot(x=self.initial_neurons_wight[boundary_indexes, self.get_feature_index(_x)],
                             y=self.initial_neurons_wight[boundary_indexes, self.get_feature_index(_y)],
                             s=50, color="black", marker="H")
        return ax

    def _init_neurons_scatter(self, _x, _y, _s=40):
        """ Draws the initial neurons scatter layer over the pot

        Args:
            _x (str): The x-axis feature name
            _y (str): The y-axis feature name
            _s (int): Indicates the data points size in scatter plot

        Returns:

        """
        ax = sns.scatterplot(x=self.initial_neurons_wight[:, self.get_feature_index(_x)],
                             y=self.initial_neurons_wight[:, self.get_feature_index(_y)],
                             s=_s * 1.15, color="slateblue", edgecolor="slateblue")
        return ax

    def _neurons_scatter(self, _x, _y):
        """ Draws the neurons scatter layer over the pot

        Args:
            _x (str): The x-axis feature name
            _y (str): The y-axis feature name

        Returns:

        """
        ax = sns.scatterplot(x=self.neurons[:, self.get_feature_index(_x)],
                             y=self.neurons[:, self.get_feature_index(_y)],
                             s=50, color="slateblue", marker="H")
        return ax

    def _draw_neuron_number(self, ax, _x, _y):
        """ Draws the neuron number as a layer over the plat

        Args:
            _x (str): The x-axis feature name
            _y (str): The y-axis feature name

        Returns:
        """
        for neuron_item in range(self.x_dim * self.y_dim):
            ax.annotate(neuron_item, (self.neurons[neuron_item, self.get_feature_index(_x)],
                                      self.neurons[neuron_item, self.get_feature_index(_y)]), c='b')
        return ax

    def _boundary_scatter(self, _x, _y):
        """ Draws the boundary neurons scatter as a layer over the pot

        Args:
            _x (str): The x-axis feature name
            _y (str): The y-axis feature name

        Returns:
        """
        boundary_indexes = self.get_boundary(self.x_dim)
        ax = sns.scatterplot(x=self.neurons[boundary_indexes, self.get_feature_index(_x)],
                             y=self.neurons[boundary_indexes, self.get_feature_index(_y)],
                             s=50, color="black", marker="H")
        return ax

    def _draw_som(self, ax, _x, _y):
        """ Draws the self organizing map topological graph as a layer over the plot

        Args:
            _x (str): The x-axis feature name
            _y (str): The y-axis feature name

        Returns:
        """
        direct_neighbours = np.argwhere(self.get_direct_neighbours_matrix(self.x_dim,
                                                                          self.y_dim,
                                                                          star_flag=False) == 1)
        # Fixme: The star_flag should be checked again
        for neurons_connection in direct_neighbours:
            ax.plot(self.neurons[neurons_connection, self.get_feature_index(_x)],
                    self.neurons[neurons_connection, self.get_feature_index(_y)],
                    'r-', linewidth=0.4)
        return ax

    def plot_init_weight_positions(self, feature_x, feature_y, neuron_number=True,
                                   _scatter=True, boundary=True, _c=None, _s=40, neuron_scatter=False):
        """ Plots the neurons initial prototype.

        It plots a scatter and the neurons prototype as a top layer.

        Args:
            _scatter (boolean): If True, the (scatter) of the real data is plotted also.
            neuron_number (boolean): If True, the neuron number will be plotted.
            feature_x (str): Indicates the x-axis data series.
            feature_y (str): Indicates the y-axis data series.
            boundary (boolean): If True, the boundary nodes will be distinctive in plot.
            _c (str): Indicates a feature that will be used data points colors
            _s (int) : The size of the data points in the scatter plot
            neuron_scatter (boolean): If True it shows the neurones

        Returns:
        """
        # Fixme: The star_flag should be checked again

        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)

        if _scatter:
            ax = self._scatter(self.data, feature_x, feature_y, self.labels, _s)

        if neuron_scatter:
            ax = self._init_neurons_scatter(feature_x, feature_y, _s)

        if boundary:
            ax = self._init_boundary_scatter(feature_x, feature_y)

        ax = self._draw_init_som(ax, feature_x, feature_y, _s)

        if neuron_number:
            ax = self._draw_init_neuron_number(ax, feature_x, feature_y)

        plt.grid(False)
        ax.legend().set_visible(False)
        plt.title('SOM Initial Weight Positions and Scatter Plot')
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/weight_positions_plot.png')
        else:
            plt.show()
        plt.close()

    def plot_weight_positions(self, feature_x, feature_y, neuron_number=True, _scatter=True,
                              boundary=True, point_annotation=None, _c=None, _s=40, neuron_scatter=False):
        """ Plots the neurons prototype.

        It plots a scatter and the neurons prototype as a top layer.

        Args:
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
            _c (str): Indicates a feature that will be used data points colors
            _s (int) : The size of the data points in the scatter plot
            neuron_scatter (boolean): If True it shows the neurones
        Returns:
        """
        # Fixme: The star_flag should be checked again
        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)

        if _scatter:
            ax = self._scatter(self.data, feature_x, feature_y, self.labels, _s)

        if neuron_scatter:
            ax = self._neurons_scatter(feature_x, feature_y)

        if boundary:
            ax = self._boundary_scatter(feature_x, feature_y)

        ax = self._draw_som(ax, feature_x, feature_y)

        if neuron_number:
            ax = self._draw_neuron_number(ax, feature_x, feature_y)

        if point_annotation:
            ax = self._draw_point_annotation(ax, self.data, feature_x, feature_y, point_annotation)

        ax.legend().set_visible(False)
        plt.title('SOM Weight Positions and Scatter Plot ({}).'.format(point_annotation))
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/weight_positions_plot.png')
        else:
            plt.show()
        plt.close()

    def plot_neighbor_distance_hexbin(self, smoothness=2, aligned_vertical=False, bin_number=100, label=False):
        """ Plots the neurons neighbor distance.

        Args:
            smoothness (float): Indicates the kernel smoothness parameter
            aligned_vertical (boolean): If True, the hexagons are aligned vertically.
            label (boolean): If True, the distance values are plotted in any cell.
            bin_number (int): Indicates the number of the data bins in order to set the colors' spectrum.

        Returns:
        """
        # TODO: the horizontal alignment have to be revised.

        assert self.x_dim > 1 and self.y_dim > 1, "Topology map is too small dimensions: x={},y={}.".format(self.x_dim,
                                                                                                            self.y_dim)

        rgba_color = cm.hot.reversed()(np.linspace(0, 1, bin_number))
        self.unified_distance_matrix = self.get_unified_distance_matrix(smoothness=smoothness)
        colors_cast, _ = pd.cut(self.unified_distance_matrix.T.flatten(), bins=bin_number, labels=False, retbins=True)
        labels = []
        coordinates = []

        for y_coordinate in range(self.y_dim):
            for x_coordinate in range(self.x_dim):
                coordinates.append([x_coordinate, y_coordinate, -(y_coordinate + x_coordinate % 2)])
                labels.append(str('{}{}'.format(x_coordinate, y_coordinate)))

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

        for x, y, c, l in zip(horizontal_coordinates, vertical_coordinates, colors_cast, labels):
            hexbin = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                    orientation=np.radians(rotation_angel),
                                    facecolor=rgba_color[c], alpha=None, edgecolor='gray')
            ax.add_patch(hexbin)
            ax.scatter(horizontal_coordinates, vertical_coordinates, c='green', alpha=0.5)

            if label:
                ax.text(x, y, l[0], ha='center', va='center', size=20)
        plt.title('SOM Neighbor Distances')
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/neighbor_distance_hexbin_plot.png')
        else:
            plt.show()
        plt.close()

    def plot_sample_hit(self, aligned_vertical=False, label=True):
        """ Plots the number of associated instances pre neuron.

        Args:
            aligned_vertical (boolean): If True, the hexagons are aligned vertically.
            label (boolean): If True, the number of the instances are plotted in any cell.

        Returns:

        """
        # TODO: the horizontal alignment have to be revised.

        assert self.x_dim > 1 and self.y_dim > 1, "Topology map has too small dimensions: x={},y={}.".format(
            self.x_dim,
            self.y_dim)
        max_association = max(Counter(self.neurons_association_list).values())
        labels = []
        coordinates = []

        for y_coordinate in range(self.y_dim):
            for x_coordinate in range(self.x_dim):
                coordinates.append([x_coordinate, y_coordinate, -(y_coordinate + x_coordinate % 2)])
                labels.append(str(self.neurons_association_list.count(self.get_ordinal_index(x_coordinate,
                                                                                             y_coordinate))))

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
        plt.title('SOM Sample Hits')
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/sample_hits_plot.png')
        else:
            plt.show()
        plt.close()

    def plot_neurons_guid(self, aligned_vertical=False, label=True):
        """ Plots the neurons map guid

        Args:
            aligned_vertical (boolean): If True, the hexagons are aligned vertically.
            label (boolean): If True, the number of the instances are plotted in any cell.

        Returns:
            Plot the map
        """
        # TODO: the horizontal alignment have to be revised.

        assert self.x_dim > 1 and self.y_dim > 1, "Topology map has too small dimensions: x={},y={}.".format(self.x_dim,
                                                                                                             self.y_dim)

        max_association = max(Counter(self.neurons_association_list).values())
        labels = range(0, self.y_dim * self.x_dim)
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

            if label and int(l) > 0:
                ax.text(x, y, l, ha='center', va='center', size=10)
        plt.title('SOM Code map')
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/som_code_map.png')
        else:
            plt.show()
        plt.close()

    def plot_weight_planes(self, aligned_vertical=False, feature='all', bin_number=100, code_map=True):
        """ Plots the neurons weight planes.

        Args:
            code_map (boolean): If True, the hexbin codemap is plated
            bin_number (int): Indicates the number of the bins in order to generate the color spectrumm
            feature (str/int): Indicates the name of the feature set that we would like to have plot of.
            aligned_vertical (boolean): If True, the hexagonales are aligned vertically.

        Returns:

        """
        if feature == 'all':
            grid_dimension = math.ceil(math.sqrt(self.data.shape[1]))
            grid_size = (grid_dimension, grid_dimension)
            fig = plt.figure(figsize=self._figsize, dpi=self._dpi)
            for feature_item in range(0, self.data.shape[1]):
                rgba_color = cm.hot.reversed()(np.linspace(0, 1, bin_number))
                # Fixme: the duplicates should be checked before.
                color_template, _ = pd.cut(self.neurons[:, feature_item], bins=bin_number, labels=False,
                                           retbins=True)
                colors = color_template
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

                vars()['ax{}'.format(feature_item)] = plt.subplot2grid(grid_size,
                                                                       (feature_item % grid_dimension,
                                                                        math.floor(feature_item / grid_dimension)))
                vars()['ax{}'.format(feature_item)].set_aspect('equal')
                vars()['ax{}'.format(feature_item)].set_facecolor('gray')
                vars()['ax{}'.format(feature_item)].title.set_text(
                    'Weight Plane Of "{}"'.format(self.data.columns[feature_item]))

                for x, y, c in zip(horizontal_coordinates, vertical_coordinates, colors):
                    hexbin = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                            orientation=np.radians(rotation_angel),
                                            facecolor=rgba_color[c], alpha=None, edgecolor='gray')
                    vars()['ax{}'.format(feature_item)].add_patch(hexbin)
                    vars()['ax{}'.format(feature_item)].scatter(horizontal_coordinates, vertical_coordinates, alpha=0.5)

        elif type(feature) == str and feature in list(self.data):
            feature_index = list(self.data).index(feature)
            rgba_color = cm.hot.reversed()(np.linspace(0, 1, bin_number))
            color_template, _ = pd.cut(self.neurons[:, feature_index],
                                       bins=bin_number,
                                       labels=False,
                                       retbins=True)
            colors = color_template
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

            fig, ax = plt.subplots(1)
            ax.set_aspect('equal')
            ax.set_facecolor('gray')
            ax.title.set_text('Weight Plane Of "{}"'.format(feature))

            for x, y, c in zip(horizontal_coordinates, vertical_coordinates, colors):
                hexbin = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                        orientation=np.radians(rotation_angel),
                                        facecolor=rgba_color[c], alpha=None, edgecolor='gray')
                ax.add_patch(hexbin)
                ax.scatter(horizontal_coordinates, vertical_coordinates, alpha=0.5)

        elif type(feature) == int and len(list(self.data)) > feature >= 0:
            feature_index = feature
            rgba_color = cm.hot.reversed()(np.linspace(0, 1, bin_number))
            color_template, _ = pd.cut(self.neurons[:, feature_index], bins=bin_number, labels=False,
                                       retbins=True)
            colors = color_template
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

            fig, ax = plt.subplots(1)
            ax.set_aspect('equal')
            ax.set_facecolor('gray')
            ax.title.set_text('Weight Plane Of "{}"'.format(self.data.columns[feature]))

            for x, y, c in zip(horizontal_coordinates, vertical_coordinates, colors):
                hexbin = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                        orientation=np.radians(rotation_angel),
                                        facecolor=rgba_color[c], alpha=None, edgecolor='gray')
                ax.add_patch(hexbin)
                ax.scatter(horizontal_coordinates, vertical_coordinates, alpha=0.5)

        elif type(feature) == list:
            grid_dimension = math.ceil(math.sqrt(len(feature)))
            grid_size = (grid_dimension, grid_dimension)
            fig = plt.figure(figsize=(16, 10))
            fig_counter = 0
            for feature_item in range(0, len(feature)):
                rgba_color = cm.hot.reversed()(np.linspace(0, 1, bin_number))
                color_template, _ = pd.cut(self.neurons[:, feature_item],
                                           bins=bin_number,
                                           labels=False,
                                           retbins=True)
                colors = color_template
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

                vars()['ax{}'.format(feature_item)] = plt.subplot2grid(grid_size,
                                                                       (feature_item % grid_dimension,
                                                                        math.floor(feature_item / grid_dimension)))
                vars()['ax{}'.format(feature_item)].set_aspect('equal')
                vars()['ax{}'.format(feature_item)].set_facecolor('gray')
                vars()['ax{}'.format(feature_item)].title.set_text(
                    'Weight Plane Of "{}"'.format(self.data.columns[feature_item]))

                for x, y, c in zip(horizontal_coordinates, vertical_coordinates, colors):
                    hexbin = RegularPolygon((x, y), numVertices=6, radius=2. / 3.,
                                            orientation=np.radians(rotation_angel),
                                            facecolor=rgba_color[c], alpha=None, edgecolor='gray')
                    vars()['ax{}'.format(feature_item)].add_patch(hexbin)
                    vars()['ax{}'.format(feature_item)].scatter(horizontal_coordinates, vertical_coordinates, alpha=0.5)
        else:
            sys.exit("The feature name/index is not the name of a training data frame dimension or index")

        plt.tight_layout()

        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/weight_planes_plot.png')
            plt.close()
        else:
            plt.show()
            plt.close()

    def get_feature_name(self, feature_index=None):
        """ Finds a feature name.

        Args:
            feature_index (int): Indicate the feature index in data set.

        Returns:
            A string indicates teh feature name.
        """
        return self.data.columns[feature_index]

    def plot_training_progress(self, log=False):
        """ Plots SOM Training Progress

        This shows the training progress over the iterations.
        Ideally you want the training to reach a minimum plateau.
        The measurement is Mean of distance from the closest Neuron

        Args:
            log (boolean): If True, the log of the Mean of distance from the closest Neuron takes to account.

        Returns:
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self._figsize, dpi=self._dpi)

        if log:
            plt.plot(range(0, self.epoch + 1), list(map(math.log, self.neurons_energy)), 'r-', linewidth=0.8)
        else:
            plt.plot(range(0, self.epoch + 1), self.neurons_energy, 'r-', linewidth=0.8)
        plt.grid(True, which='both', axis='both', linestyle='--')
        plt.xlabel('epoch')
        plt.ylabel('Mean distance of closest unit (log:{})'.format(log))
        plt.title('Training Progress')

        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/training_progress_plot.png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def som2networkx(self, _weight_features='all', _plot=False):
        """ Converts a fitted SOM to Networkx graph object.

        Args:
            _weight_features (str/list): Includes a features that we would like to compute the edges weight over
            _plot (boolean): If True, It plots the graph
        Returns:
            A networkx graph object
        """
        self.graph = nx.Graph()
        self.graph_weighting_features = self.get_feature_indexes(_weight_features)

        # Fixme: The star_flag should be checked again
        for neurons_connection in np.argwhere(self.get_direct_neighbours_matrix(self.x_dim,
                                                                                self.y_dim,
                                                                                star_flag=False) == 1):
            self.graph.add_edge(neurons_connection[0],
                                neurons_connection[1],
                                weight=euclidean(self.neurons[neurons_connection[0], self.graph_weighting_features],
                                                 self.neurons[neurons_connection[1], self.graph_weighting_features]))
        if _plot:
            nx.draw(self.graph, with_labels=True)
            plt.show()
        return self.graph

    def get_shortest_geodesic_path(self, source_node, target_node, _weight_features='all'):
        """Converts a fitted SON to Networkx graph object.

        Args:
            source_node (int): Start node for path. If not specified, compute the shortest paths for each starting node.
            target_node (int): End node for path. If not specified, compute the shortest paths to all possible nodes.
            _weight_features (str/list): Includes a features that we would like to compute the edges weight over

        Returns:
            A dict includes a sequential list of the path nodes and a float that indicates the path length.
        """
        # Todo: Double-check the situation of the SOM graph

        if not self.graph:
            self.som2networkx(_weight_features)
        elif set(self.graph_weighting_features).symmetric_difference(set(self.get_feature_indexes(_weight_features))):
            self.som2networkx(_weight_features)
        else:
            print(BColors.WARNING +
                  'Features {} are used to compute the edges weight of SOM graph.'.format(self.graph_weighting_features)
                  + BColors.ENDC)

        # Compute the shortest path
        shortest_geodesic_path = nx.shortest_path(self.graph,
                                                  source=int(source_node),
                                                  target=int(target_node),
                                                  weight='weight')
        shortest_geodesic_path_length = nx.shortest_path_length(self.graph,
                                                                source=int(source_node),
                                                                target=int(target_node),
                                                                weight='weight')
        return {'shortest_path': shortest_geodesic_path, 'shortest_path_length': shortest_geodesic_path_length}

    @staticmethod
    def mspf2fps(_mspf):
        """ Convert milliseconds per frame to frame per second.

        Args:
            _mspf (int): Milliseconds per frame value

        Returns:
            An int that is frame per second value
        """
        return int(1000 / _mspf)

    def get_animation(self, feature_x, feature_y, neuron_number=False, _scatter=True, boundary=True, mspf=250,
                      _format='mp4', num_core=cpu_count(), _output_path='outputs/', _size=(16, 14), _s=40):
        """ Generate a animation file of the SOM training steps in 2D

        Args:
            feature_x (int): Feature that is representing the x-axis in plot
            feature_y (int): Feature that is representing the y-axis in plot
            neuron_number (boolean): If True, the neuron number will be plotted.
            _scatter (boolean): If True, the (scatter) of the real data is plotted also.
            boundary (boolean): If True, the boundary nodes will be distinctive in plot.
            mspf (int): milliseconds per frame
            _format (str): mp4, gif or smp4 (It is generated mp4 sequentially)
            num_core (int): the number of cpu cores to parallelize the tasks
            _output_path (str): The path to save the output file.
            _size (tuple): The frame size of the animation
            _s (int) : The size of the data points in the scatter plot

        Returns:
            A str include the output path
        """
        # Fixme: The star_flag should be checked again
        assert _format in ['gif', 'mp4', 'smp4'], "The animation format '{}' is not supported".format(_format)

        def __take_frame(epoch_indexes, __figsize=_size):
            """ Create all training 2D wight position plot for any epoch

            Args:
                epoch_indexes (int): The number of training epochs
                __figsize (tuple): The frame size of the animation

            Returns:
                A set of png file for each plot is saved in temp path.
            """
            fig = plt.figure(figsize=__figsize)

            for _epoch_item in epoch_indexes:
                if _scatter:
                    _ax = self._scatter(self.data, feature_x, feature_y, None, _s)

                _ax = sns.scatterplot(x=self.animation[_epoch_item][:, self.get_feature_index(feature_x)],
                                      y=self.animation[_epoch_item][:, self.get_feature_index(feature_y)],
                                      s=_s * 1.15, color="slateblue", marker="H")
                if boundary:
                    _boundary_indexes = self.get_boundary(self.x_dim)
                    _ax = sns.scatterplot(
                        x=self.animation[_epoch_item][_boundary_indexes, self.get_feature_index(feature_x)],
                        y=self.animation[_epoch_item][_boundary_indexes, self.get_feature_index(feature_y)],
                        s=50, color="black", marker="H")

                for _neurons_connection in np.argwhere(self.get_direct_neighbours_matrix(self.x_dim,
                                                                                         self.y_dim,
                                                                                         star_flag=False) == 1):
                    _ax.plot(self.animation[_epoch_item][_neurons_connection, self.get_feature_index(feature_x)],
                             self.animation[_epoch_item][_neurons_connection, self.get_feature_index(feature_y)],
                             'r-', linewidth=0.4)
                if neuron_number:
                    for _neuron_item in range(self.x_dim * self.y_dim):
                        _ax.annotate(_neuron_item,
                                     (self.animation[_epoch_item][_neuron_item, self.get_feature_index(feature_x)],
                                      self.animation[_epoch_item][_neuron_item, self.get_feature_index(feature_y)]))
                plt.title('Epoch={}'.format(_epoch_item))
                plt.savefig('outputs/animation_temp/fig{}.png'.format(_epoch_item))
                fig.clf()

                progressbar(_epoch_item + 1, self.epoch, bar_len=20, status=str(_epoch_item) + "/" + str(self.epoch),
                            functionality='Video Frames generated')

        if _format in ['gif', 'mp4']:
            # Clean previous results and recreate the directory
            if os.path.isdir(self.animation_temp_path):
                warnings.warn(
                    'The temporal directory "{}" already exist, Have to delete'.format(self.animation_temp_path))
                shutil.rmtree(self.animation_temp_path)
            os.makedirs(self.animation_temp_path)

            epoch_chunks = np.array_split(range(self.epoch + 1), 4)
            processes = [Process(target=__take_frame,
                                 args=(epoch_chunks[chunk_start_index],)) for chunk_start_index in range(num_core)]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            file_names = [fn for fn in os.listdir(self.animation_temp_path) if fn.endswith('.png')]
            file_names = Tcl().call('lsort', '-dict', file_names)

            if _format == 'gif':
                kargs = {'mode': 'I'}
            else:
                # Mp4 output
                kargs = {'fps': self.mspf2fps(mspf), 'macro_block_size': None}
            with imageio.get_writer('{}/training_progress.{}'.format(_output_path, _format), **kargs) as writer:
                for filename in file_names:
                    if filename.startswith('fig'):
                        image = imageio.imread(self.animation_temp_path + filename)
                        writer.append_data(image)
        elif _format == 'smp4':
            camera = Camera(plt.figure(figsize=_size))
            for epoch_item in range(0, self.epoch):
                if _scatter:
                    ax = self._scatter(self.data, feature_x, feature_y, None, _s)

                ax = sns.scatterplot(x=self.animation[epoch_item][:, self.get_feature_index(feature_x)],
                                     y=self.animation[epoch_item][:, self.get_feature_index(feature_y)],
                                     s=_s * 1.15, color="slateblue", marker="H")

                if boundary:
                    boundary_indexes = self.get_boundary(self.x_dim)
                    ax = sns.scatterplot(
                        x=self.animation[epoch_item][boundary_indexes, self.get_feature_index(feature_x)],
                        y=self.animation[epoch_item][boundary_indexes, self.get_feature_index(feature_y)],
                        s=50, color="black", marker="H")

                # Fixme: The star_flag should be checked again
                for neurons_connection in np.argwhere(self.get_direct_neighbours_matrix(self.x_dim,
                                                                                        self.y_dim,
                                                                                        star_flag=False) == 1):
                    ax.plot(self.animation[epoch_item][neurons_connection, self.get_feature_index(feature_x)],
                            self.animation[epoch_item][neurons_connection, self.get_feature_index(feature_y)],
                            'r-', linewidth=0.4)
                if neuron_number:
                    for neuron_item in range(self.x_dim * self.y_dim):
                        ax.annotate(neuron_item,
                                    (self.animation[epoch_item][neuron_item, self.get_feature_index(feature_x)],
                                     self.animation[epoch_item][neuron_item, self.get_feature_index(feature_y)]))
                plt.title('SOM Weight Positions and Scatter Plot (Epoch={})'.format(epoch_item))

                camera.snap()
                progressbar(epoch_item + 1,
                            self.epoch,
                            bar_len=20,
                            status=str(epoch_item) + "/" + str(self.epoch),
                            functionality='Video Frames generated')

            anim = camera.animate(blit=True, interval=mspf)
            _extension = 'mp4'
            anim.save('{}/training_progress.{}'.format(_output_path, _format))
        return '{}/training_progress.{}'.format(_output_path, _format)

    # TODO: the supper clustering
    def self_cluster(self, explicit=False, merge_range=4, _smoothness=2, _merg=False):
        """ Clusters the neurones that are close to each other or liked to each other

        Args:
            explicit (boolean): If True, the centroid is connected to the neuron explicitly
            merge_range (float): Indicates a range that is used as a percentage of a certain distance in the code to
                           determine whether components are closer to their centroids or centroids closer to each other.
            _smoothness (float): Indicate the color smoothness
            _merg (boolean): If True, It apply the cluster merging
        Returns:
            Plot out the heat_map.

        Note:
            The heat_value array is transposed before plotting and calculating the centroid,
            in order to have the same orientation over all plot function in SON package.
        """
        # TODO: The association neurons to each cluster
        assert self.x_dim > 1 and self.y_dim > 1, "Topology map has too small dimensions, x={} and y={}.".format(
            self.x_dim, self.y_dim)

        def _get_label_map(uc_dict):
            return {''.join([str(elem) for elem in value]): key for key, value in uc_dict.items()}

        _unified_distance_matrix = self.get_unified_distance_matrix(smoothness=_smoothness)
        _unified_distance_matrix = _unified_distance_matrix.T  # to uniform both hexbin plots and heatmap plot (matrix)

        # get_clusters
        if _merg:
            # Fixme: we need to apply the cluster merging later
            self.centroids = self.get_merged_clusters(_unified_distance_matrix, explicit, merge_range)
        else:
            self.centroids = self.get_centroids(_unified_distance_matrix, explicit)

        unique_centroids = _get_label_map(self.get_unique_centroids(self.centroids, label=True))

        # Compute the label matrix
        labels = np.zeros((self.x_dim, self.y_dim), dtype=int)
        for ix in range(self.x_dim):
            for iy in range(self.y_dim):
                cx = self.centroids['centroid_x'][ix, iy]
                cy = self.centroids['centroid_y'][ix, iy]
                labels[ix, iy] = unique_centroids.get(''.join([str(elem) for elem in [cx, cy]]))
        self.neurons_cluster_id = labels
        return self.neurons_cluster_id

    def __draw_labels(self):
        """ Plots the labels on the map (rrg mesh) if available

        Returns:

        """
        # Fixme: it has to be compatible with the transposed matrix
        if not (self.labels is None) and (len(self.labels) != 0):
            _count = np.array([[0] * self.y_dim] * self.x_dim)
            # count the labels in each map cell
            for i in range(self.data.shape[0]):
                nix = self.neurons_association_list[i]
                c = self.get_neuron_coordinate(nix)
                _count[c[0] - 1, c[1] - 1] = _count[c[0] - 1, c[1] - 1] + 1

            for i in range(self.data.shape[0]):
                c = self.get_neuron_coordinate(self.neurons_association_list[i])
                # we only print one label per cell
                if _count[c[0] - 1, c[1] - 1] > 0:
                    _count[c[0] - 1, c[1] - 1] = 0
                    l = self.labels[i]
                    plt.text(c[1] + 0.5, c[0] + 0.5, l)  # It is transposed to match into the heatmap and cluster plot
        else:
            raise ValueError("The labels has not defined.")

    def plot_cluster_map(self, _cluster_ids, _cmap=None, labels=False):
        """ Plots the neurons cluster map

        Args:
            labels (boolean): Annotate the original labels over the plot
            _cluster_ids (2d ndarray): In shape of x_dim * y_dim where the value of each cell indicates its cluster label
            _cmap (str): Color map

        Returns:
        """

        def get_legend(_unique_cluster, __cmap):
            """ Creates the plot legend

            Args:
                _unique_cluster (list): Of clusters' name
                __cmap (str): Color map

            Returns:
                A list of Patch

            Note:
                See matplotlib.patches

            """
            try:
                color_list1 = [mcolors.rgb2hex(_cmap(i)) for i in range(_cmap.N)]
                legend_elements = []
                for cluster_item in _unique_cluster:
                    legend_elements.extend([Patch(facecolor=color_list1[cluster_item - 1],
                                                  label='Cluster {}'.format(cluster_item),
                                                  edgecolor=color_list1[cluster_item - 1])])
            except IndexError:
                warnings.warn("Warning: The legand cna not be generated for the 'plot_cluster_map'.")
            return legend_elements

        color_template = _cluster_ids.T  # To uniform orientation with hexbin plots and heatmap plot
        unique_cluster = sorted(list(set(color_template.flatten())))
        num_clusters = len(unique_cluster)
        _cmap = cm.get_cmap(_cmap, num_clusters)

        # Plotting
        _cmap = get_listed_cmap(color_template.flatten(), PARAVER_STATES_COLOR)
        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)
        mesh = ax.pcolor(color_template, cmap=_cmap, edgecolors='gray')
        ax.set_xticks(np.arange(self.x_dim) + 0.5, minor=False)
        ax.set_yticks(np.arange(self.y_dim) + 0.5, minor=False)
        ax.set_xticklabels(np.arange(self.x_dim), minor=False)
        ax.set_yticklabels(np.arange(self.y_dim), minor=False)

        if labels:
            self.__draw_labels()

        plt.title('Supper Cluster Boundary')
        fontP = FontProperties()
        fontP.set_size(self._figsize[0] * 0.7)
        ax.legend(handles=get_legend(unique_cluster, _cmap), bbox_to_anchor=(1.0, 1), loc='upper left', prop=fontP)
        if self.make_jupiter_report:
            plt.savefig('OCA_Report/pic/heatmap_plot.png', bbox_inches='tight')
        else:
            plt.show()
            plt.close()

    @staticmethod
    def get_oriented_direct_neighbours(size_x, size_y, star_flag=False, orientation='v'):
        # Fixme: The star_flag should be checked again
        """ Compute the direct neighbour's matrix(map) oriented vertically or horizontally

        We have matrix nxn and a neighboring matrix (n*n)x(n*n), it computes a (n*n)x(n*n) binary matrix.
        If the jth cell is a direct neighbor of ith cell it will set to one else zero.

        Args:
            size_y (int): Indicates the matrix columns number
            size_x (int): Indicates the matrix rows number.
            star_flag (boolean): If True, The star connection to neighbours will be assumed
            orientation (char): V means vertical and H means horizontal

        Returns:
            A binary ndarray includes the direct neighbors map.
        """

        def __vertical_mask(vertex):
            """ Mask the vertical vertexes"""
            if np.abs(vertex[0] - vertex[1]) == 1:
                return True
            else:
                return False

        def __horizontal_mask(vertex):
            """ Mask the horizontal vertexes"""
            if np.abs(vertex[0] - vertex[1]) != 1:
                return True
            else:
                return False

        direct_neighbours = np.argwhere(SOM.get_direct_neighbours_matrix(size_x, size_y, star_flag=False) == 1)
        if orientation.lower() == 'v':
            mask = np.apply_along_axis(__vertical_mask, 1, direct_neighbours)
        elif orientation.lower() == 'h':
            mask = np.apply_along_axis(__horizontal_mask, 1, direct_neighbours)
        else:
            raise ValueError('ERROR: The orientation has to be V: vertically or H: horizontally')
        return direct_neighbours[mask, :]

    def plot_cluster_association(self, feature_x, feature_y, label_matrix, _s=40):
        """ Scatter plots of the computed cluster_ids vs data labels

        Args:
            feature_x (str): Indicates the x-axis data series.
            feature_y (str): Indicates the y-axis data series.
            label_matrix (2d array): Indicates the cells cluster labels
            _s (int) : The size of the data points in the scatter plot

        Returns:

        """
        flat_cluster_map = label_matrix.T.flatten('F')
        __labels = list(map(lambda data_item: flat_cluster_map[data_item], self.neurons_association_list))
        _cmap = get_listed_cmap(__labels, PARAVER_STATES_COLOR)

        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)

        scatter = plt.scatter(self.data.loc[:, feature_x], self.data.loc[:, feature_y], marker='o',
                              c=__labels, cmap=_cmap, s=_s, edgecolor='k')

        # Make the legend
        handles, legend_labels = scatter.legend_elements()
        for cluster_item in range(len(legend_labels)):
            label_text = 'Cluster{}'.format(cluster_item + 1)
            legend_labels[cluster_item] = str('$\\mathdefault{' + label_text + '}$')
        legend = ax.legend(handles, legend_labels, loc="auto")
        ax.add_artist(legend)

        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(1))
        plt.title("Self Clusters scatter plot")
        plt.xticks(rotation=90)
        ax.margins(0.15)
        plt.grid()
        plt.show()

    @staticmethod
    def angle_between_vectors(a, b, deg=False):
        """ Computes the angle between two vector

        The value is between [0,1]

        Args:
            a (ndarray): Indicates the first vector
            b (ndarray): Indicates the second vector
            deg (boolean): If True, it will return the angel as degree else radian

        Returns:
            A float indicates the angel (degree/radian) between two vectors.
        """
        _arccos = dot(a, b) / norm(a) / norm(b)
        _arccos = 1.0 if _arccos > 1.0 else _arccos
        _arccos = -1.0 if _arccos < -1.0 else _arccos
        if not deg:
            return abs(_arccos)
        else:
            return math.degrees(np.arccos(_arccos))

    def cat2int(self):
        """ Maps the categorical labels to the int

        Returns:
            A dict of the mapping function
        """
        label_map = dict(enumerate(list(set(self.labels))))
        label_map = {value: key for key, value in label_map.items()}
        self.labels = self.labels.map(label_map)
        print(BColors.WARNING + 'The labels are mapped through: {}'.format(label_map) + BColors.ENDC)
        return label_map

    # ------------------------------------------------------------------------------------------------
    def get_inter_clusters_distance(self, centroids, unique_centroids, unified_distance_matrix):
        """ Computes the mean inter_clusters distances (pairwise).

        Computes the distance of all individuals from the centroid cluster by cluster.

        Args:
            unique_centroids (dict): Indicates the centroids coordinates
            centroids (ndarray): Includes the x, y coordinates of all neurons' centroids
            unified_distance_matrix (ndarray): Indicates the distance matrix.

        Returns:
            A dict of dict includes the centroids coordinates as key and distances as values.
        """
        clusters = self.list_clusters(centroids, unique_centroids, unified_distance_matrix)

        _heat_matrix = np.zeros(shape=(max([len(clusters[i]) for i in range(len(clusters))]), len(clusters)))

        for i in range(len(clusters)):
            for j in range(len(clusters[i])):
                _heat_matrix[j, i] = clusters[i][j]

        columns = _heat_matrix.shape[1]
        _temp = np.matrix.transpose(np.array(list(combinations([i for i in range(columns)], 2))))

        __temp = np.zeros(shape=(_heat_matrix.shape[0], _temp.shape[1]))

        for i in range(_temp.shape[1]):
            __temp[:, i] = np.where(_heat_matrix[:, _temp[1, i]] * _heat_matrix[:, _temp[0, i]] != 0,
                                    abs(_heat_matrix[:, _temp[0, i]] - _heat_matrix[:, _temp[1, i]]), 0)
        # both are not equals 0

        index = 0
        _distance_matrix = np.zeros((columns, columns))
        _mean = np.true_divide(__temp.sum(0), (__temp != 0).sum(0))

        for x_i in range(columns - 1):
            for y_i in range(x_i, columns - 1):
                _distance_matrix[x_i, y_i + 1] = _mean[index]
                _distance_matrix[y_i + 1, x_i] = _mean[index]
                index = index + 1

        return _distance_matrix

    def combine_decision(self, within_cluster_dist, distance_between_clusters, merge_range):
        """ Produces a boolean matrix representing which clusters should be combined.

        Args:
            within_cluster_dist (list): The distances from centroid to cluster elements for all centroids
            distance_between_clusters (list): The average pairwise distance between clusters
            merge_range (float): The distance where the clusters are merged together.

        Returns:

        """
        inter_cluster = distance_between_clusters
        centroid_dist = self.get_inter_clusters_distance_list(within_cluster_dist)
        dim = inter_cluster.shape[1]
        to_combine = np.matrix([[False] * dim] * dim)

        for xi in range(dim):
            for yi in range(dim):
                cdist = inter_cluster[xi, yi]
                if cdist != 0:
                    rx = centroid_dist[xi] * merge_range
                    ry = centroid_dist[yi] * merge_range
                    if (cdist < centroid_dist[xi] + rx or cdist < centroid_dist[yi] + ry):
                        to_combine[xi, yi] = True
        return to_combine

    def get_cluster_neurones(self, centroids):
        """ Computes the list of the cells associated to each cluster

        Args:
            centroids (ndarray): Includes the x, y coordinates of all neurons' centroids

        Returns:
            A dict where key is a list indicate the centroid and value is a list of the neuorne's index
        """

        def _get_label_map(uc_dict):
            return {''.join([str(elem) for elem in value]): key for key, value in uc_dict.items()}

        unique_centroids = _get_label_map(self.get_unique_centroids(centroids, label=True))

        labels = np.zeros((self.x_dim, self.y_dim), dtype=int)
        for ix in range(self.x_dim):
            for iy in range(self.y_dim):
                cx = centroids['centroid_x'][ix, iy]
                cy = centroids['centroid_y'][ix, iy]
                labels[ix, iy] = unique_centroids.get(''.join([str(elem) for elem in [cx, cy]]))
        return labels

    def get_sample_hit(self):
        """ Compute the sample hit of each neurone

        Returns:
            A list of the hits
        """
        __hit = []
        for y_coordinate in range(self.y_dim):
            for x_coordinate in range(self.x_dim):
                __hit.append(self.neurons_association_list.count(self.get_ordinal_index(x_coordinate, y_coordinate)))
        return __hit

    def __get_new_centroids(self, heat, centroids):
        """ Computes the new centric cell for each cluster

        Args:
            heat (ndarray): Includes the heat map values.
            centroids (ndarray): Includes the x, y coordinates of all neurons' centroids

        Returns:
            A dict of the new centroids of each cluster:
                - keys are the index of the new centroid
                - values the old coordinates of the centroid
        """

        def _get_label_map(uc_dict):
            return {''.join([str(elem) for elem in value]): key for key, value in uc_dict.items()}

        unique_centroids_coordinates = self.get_unique_centroids(centroids, label=True)
        unique_centroids = _get_label_map(unique_centroids_coordinates)

        labels = []
        for ix in range(self.x_dim):
            for iy in range(self.y_dim):
                cx = centroids['centroid_x'][ix, iy]
                cy = centroids['centroid_y'][ix, iy]
                _cluster_id = unique_centroids.get(''.join([str(elem) for elem in [cx, cy]]))
                labels.append(_cluster_id)

        _new_centroids_map = {}
        _sample_hits = heat.flatten()
        for _cluster in unique_centroids.values():
            _cluster_items = np.where(np.asarray(labels) == _cluster)
            min_index = np.argmin(_sample_hits[_cluster_items])
            _new_centroids_map.update({_cluster_items[0][min_index]: unique_centroids_coordinates.get(_cluster)})

        return _new_centroids_map

    def new_centroid(self, heat, marge_matrix, centroids, unique_centroids):
        """ Combines centroids based on matrix of booleans.

        Args:
            heat (ndarray): Includes the heat map values.
            marge_matrix (boolean 2d array): A boolean matrix containing the centroids to merge
            centroids (dict of array): Includes two matrix of the centroid locations in the map (centroid_x,centroid_y)
            unique_centroids (dict of array): Includes two lists of the unique centroid (position_x, position_y)

        Returns:
             A dict includes the x, y coordinates of the centroids for any neuron.
        """
        _new_centroid = centroids
        for x_i in range(marge_matrix.shape[0]):
            for y_i in range(marge_matrix.shape[1]):
                if marge_matrix[x_i, y_i]:
                    x1, y1 = unique_centroids['position_x'][x_i], unique_centroids['position_y'][x_i]
                    x2, y2 = unique_centroids['position_x'][y_i], unique_centroids['position_y'][y_i]
                    _new_centroid = self.swap_centroids(x1, y1, x2, y2, _new_centroid)

        __new_centroids_map = self.__get_new_centroids(heat, _new_centroid)
        for __new_index, __old_coord in __new_centroids_map.items():
            __new_coord = self.get_neuron_coordinate(__new_index)
            _new_centroid = self.swap_centroids(__old_coord[0], __old_coord[1],
                                                __new_coord[1], __new_coord[0], _new_centroid)

        return _new_centroid

    def swap_centroids(self, x1, y1, x2, y2, centroids):
        """ Changes every instance of a centroid to one that it should be combined with.
        Args:
            A matrix of the centroid locations in the map
        """
        for xi in range(self.x_dim):
            for yi in range(self.y_dim):
                if centroids['centroid_x'][xi, yi] == x1 and centroids['centroid_y'][xi, yi] == y1:
                    centroids['centroid_x'][xi, yi] = x2
                    centroids['centroid_y'][xi, yi] = y2
        return {"centroid_x": centroids['centroid_x'], "centroid_y": centroids['centroid_y']}

    def get_centroids(self, heat_values, explicit=False):
        """ Computes the centroid of any neuron.

        Args:
            heat_values (ndarray): Includes the heat map values.
            explicit (boolean): If True, it computes the centroids explicitly.

        Returns:
            A dict includes the x, y coordinates of the centroids for any neuron.
        """
        # Fixme: It needs to be revised completely from the scratch.

        centroid_x = np.matrix([[-1] * self.y_dim for _ in range(self.x_dim)])
        centroid_y = np.matrix([[-1] * self.y_dim for _ in range(self.y_dim)])

        def __get_centroid(ix, iy):
            # recursive function to find the centroid of a point on the map

            if (centroid_x[ix, iy] > -1) and (centroid_y[ix, iy] > -1):
                return {"bestx": centroid_x[ix, iy], "besty": centroid_y[ix, iy]}

            min_val = heat_values[ix, iy]
            min_x = ix
            min_y = iy

            # (ix, iy) is an inner map element
            if ix > 0 and ix < self.x_dim - 1 and iy > 0 and iy < self.y_dim - 1:

                if heat_values[ix - 1, iy - 1] < min_val:
                    min_val = heat_values[ix - 1, iy - 1]
                    min_x = ix - 1
                    min_y = iy - 1

                if heat_values[ix, iy - 1] < min_val:
                    min_val = heat_values[ix, iy - 1]
                    min_x = ix
                    min_y = iy - 1

                if heat_values[ix + 1, iy - 1] < min_val:
                    min_val = heat_values[ix + 1, iy - 1]
                    min_x = ix + 1
                    min_y = iy - 1

                if heat_values[ix + 1, iy] < min_val:
                    min_val = heat_values[ix + 1, iy]
                    min_x = ix + 1
                    min_y = iy

                if heat_values[ix + 1, iy + 1] < min_val:
                    min_val = heat_values[ix + 1, iy + 1]
                    min_x = ix + 1
                    min_y = iy + 1

                if heat_values[ix, iy + 1] < min_val:
                    min_val = heat_values[ix, iy + 1]
                    min_x = ix
                    min_y = iy + 1

                if heat_values[ix - 1, iy + 1] < min_val:
                    min_val = heat_values[ix - 1, iy + 1]
                    min_x = ix - 1
                    min_y = iy + 1

                if heat_values[ix - 1, iy] < min_val:
                    min_val = heat_values[ix - 1, iy]
                    min_x = ix - 1
                    min_y = iy

            # (ix, iy) is bottom left corner
            elif ix == 0 and iy == 0:

                if heat_values[ix + 1, iy] < min_val:
                    min_val = heat_values[ix + 1, iy]
                    min_x = ix + 1
                    min_y = iy

                if heat_values[ix + 1, iy + 1] < min_val:
                    min_val = heat_values[ix + 1, iy + 1]
                    min_x = ix + 1
                    min_y = iy + 1

                if heat_values[ix, iy + 1] < min_val:
                    min_val = heat_values[ix, iy + 1]
                    min_x = ix
                    min_y = iy + 1

            # (ix, iy) is bottom right corner
            elif ix == self.x_dim - 1 and iy == 0:

                if heat_values[ix, iy + 1] < min_val:
                    min_val = heat_values[ix, iy + 1]
                    min_x = ix
                    min_y = iy + 1

                if heat_values[ix - 1, iy + 1] < min_val:
                    min_val = heat_values[ix - 1, iy + 1]
                    min_x = ix - 1
                    min_y = iy + 1

                if heat_values[ix - 1, iy] < min_val:
                    min_val = heat_values[ix - 1, iy]
                    min_x = ix - 1
                    min_y = iy

            # (ix, iy) is top right corner
            elif ix == self.x_dim - 1 and iy == self.y_dim - 1:

                if heat_values[ix - 1, iy - 1] < min_val:
                    min_val = heat_values[ix - 1, iy - 1]
                    min_x = ix - 1
                    min_y = iy - 1

                if heat_values[ix, iy - 1] < min_val:
                    min_val = heat_values[ix, iy - 1]
                    min_x = ix
                    min_y = iy - 1

                if heat_values[ix - 1, iy] < min_val:
                    min_val = heat_values[ix - 1, iy]
                    min_x = ix - 1
                    min_y = iy

            # (ix, iy) is top left corner
            elif ix == 0 and iy == self.y_dim - 1:

                if heat_values[ix, iy - 1] < min_val:
                    min_val = heat_values[ix, iy - 1]
                    min_x = ix
                    min_y = iy - 1

                if heat_values[ix + 1, iy - 1] < min_val:
                    min_val = heat_values[ix + 1, iy - 1]
                    min_x = ix + 1
                    min_y = iy - 1

                if heat_values[ix + 1, iy] < min_val:
                    min_val = heat_values[ix + 1, iy]
                    min_x = ix + 1
                    min_y = iy

            # (ix, iy) is a left side element
            elif ix == 0 and iy > 0 and iy < self.y_dim - 1:

                if heat_values[ix, iy - 1] < min_val:
                    min_val = heat_values[ix, iy - 1]
                    min_x = ix
                    min_y = iy - 1

                if heat_values[ix + 1, iy - 1] < min_val:
                    min_val = heat_values[ix + 1, iy - 1]
                    min_x = ix + 1
                    min_y = iy - 1

                if heat_values[ix + 1, iy] < min_val:
                    min_val = heat_values[ix + 1, iy]
                    min_x = ix + 1
                    min_y = iy

                if heat_values[ix + 1, iy + 1] < min_val:
                    min_val = heat_values[ix + 1, iy + 1]
                    min_x = ix + 1
                    min_y = iy + 1

                if heat_values[ix, iy + 1] < min_val:
                    min_val = heat_values[ix, iy + 1]
                    min_x = ix
                    min_y = iy + 1

            # (ix, iy) is a bottom side element
            elif ix > 0 and ix < self.x_dim - 1 and iy == 0:

                if heat_values[ix + 1, iy] < min_val:
                    min_val = heat_values[ix + 1, iy]
                    min_x = ix + 1
                    min_y = iy

                if heat_values[ix + 1, iy + 1] < min_val:
                    min_val = heat_values[ix + 1, iy + 1]
                    min_x = ix + 1
                    min_y = iy + 1

                if heat_values[ix, iy + 1] < min_val:
                    min_val = heat_values[ix, iy + 1]
                    min_x = ix
                    min_y = iy + 1

                if heat_values[ix - 1, iy + 1] < min_val:
                    min_val = heat_values[ix - 1, iy + 1]
                    min_x = ix - 1
                    min_y = iy + 1

                if heat_values[ix - 1, iy] < min_val:
                    min_val = heat_values[ix - 1, iy]
                    min_x = ix - 1
                    min_y = iy

            # (ix, iy) is a right side element
            elif ix == self.x_dim - 1 and iy > 0 and iy < self.y_dim - 1:

                if heat_values[ix - 1, iy - 1] < min_val:
                    min_val = heat_values[ix - 1, iy - 1]
                    min_x = ix - 1
                    min_y = iy - 1

                if heat_values[ix, iy - 1] < min_val:
                    min_val = heat_values[ix, iy - 1]
                    min_x = ix
                    min_y = iy - 1

                if heat_values[ix, iy + 1] < min_val:
                    min_val = heat_values[ix, iy + 1]
                    min_x = ix
                    min_y = iy + 1

                if heat_values[ix - 1, iy + 1] < min_val:
                    min_val = heat_values[ix - 1, iy + 1]
                    min_x = ix - 1
                    min_y = iy + 1

                if heat_values[ix - 1, iy] < min_val:
                    min_val = heat_values[ix - 1, iy]
                    min_x = ix - 1
                    min_y = iy

            # (ix, iy) is a top side element
            elif ix > 0 and ix < self.x_dim - 1 and iy == self.y_dim - 1:

                if heat_values[ix - 1, iy - 1] < min_val:
                    min_val = heat_values[ix - 1, iy - 1]
                    min_x = ix - 1
                    min_y = iy - 1

                if heat_values[ix, iy - 1] < min_val:
                    min_val = heat_values[ix, iy - 1]
                    min_x = ix
                    min_y = iy - 1

                if heat_values[ix + 1, iy - 1] < min_val:
                    min_val = heat_values[ix + 1, iy - 1]
                    min_x = ix + 1
                    min_y = iy - 1

                if heat_values[ix + 1, iy] < min_val:
                    min_val = heat_values[ix + 1, iy]
                    min_x = ix + 1
                    min_y = iy

                if heat_values[ix - 1, iy] < min_val:
                    min_val = heat_values[ix - 1, iy]
                    min_x = ix - 1
                    min_y = iy

            """
            if successful move to the square with the smaller value, i_e_, call __get_centroid on this new square 
            note the RETURNED x-y coords in the centroid_x and centroid_y matrix at the current location return the
            RETURNED x-y coordinates
            """

            if min_x != ix or min_y != iy:
                r_val = __get_centroid(min_x, min_y)

                # if explicit is set show the exact connected component
                # otherwise construct a connected componenent where all
                # nodes are connected to a centrol node
                if explicit:
                    centroid_x[ix, iy] = min_x
                    centroid_y[ix, iy] = min_y
                    return {"bestx": min_x, "besty": min_y}

                else:
                    centroid_x[ix, iy] = r_val['bestx']
                    centroid_y[ix, iy] = r_val['besty']
                    return r_val

            else:
                centroid_x[ix, iy] = ix
                centroid_y[ix, iy] = iy
                return {"bestx": ix, "besty": iy}

        for i in range(self.x_dim):
            for j in range(self.y_dim):
                __get_centroid(i, j)

        return {"centroid_x": centroid_x, "centroid_y": centroid_y}

    def get_merged_clusters(self, heat, explicit, merge_range):
        """ Merges the clusters to increase the accuracy

        Args:
            heat (ndarray): Includes the heat map values.
            explicit (boolean): If True, the centroid is connected to the neuron explicitly
            merge_range (float): Indicates a range that is used as a percentage of a certain distance in the code to
                           determine whether components are closer to their centroids or centroids closer to each other.

        Returns:
            A dict includes the x, y coordinates of the centroids for any neuron.
        """

        # Fixme: It needs to be revised completely from the scratch.
        # compute the connected components
        centroids = self.get_centroids(heat, explicit)

        # Get unique centroids
        unique_centroids = self.get_unique_centroids(centroids)

        # Get distance from centroid to cluster elements for all centroids
        within_cluster_dist = self.get_mean_intra_clusters_sparsity(centroids, unique_centroids, heat)

        # Get average pairwise distance between clusters
        between_cluster_dist = self.get_inter_clusters_distance(centroids, unique_centroids, heat)

        # Get a boolean matrix of whether two components should be combined
        combine_cluster_bools = self.combine_decision(within_cluster_dist, between_cluster_dist, merge_range)

        return self.new_centroid(heat, combine_cluster_bools, centroids, unique_centroids)

    def list_clusters(self, centroids, unique_centroids, umat):
        """ Computes the clusters as a list of lists.

        Args:
            centroids (dict of array): Includes two matrix of the centroid locations in the map (centroid_x,centroid_y)
            unique_centroids (dict of array): Includes two lists of the unique centroid (position_x, position_y)
            umat (ndarray 2d): A unified distance matrix
        """
        cluster_list = []

        for i in range(len(unique_centroids['position_x'])):
            cx = unique_centroids['position_x'][i]
            cy = unique_centroids['position_y'][i]

            # get the clusters associated with a unique centroid and store it in a list
            cluster_list.append(self.list_from_centroid(cx, cy, centroids, umat))
        return cluster_list

    def list_from_centroid(self, x, y, centroids, umat):
        """ Computes all cluster members associated to a centroid.

        Args:
            x (int): the x position of a centroid
            y (int): the y position of a centroid
            centroids (dict of array): Includes two matrix of the centroid locations in the map (centroid_x,centroid_y)
            umat (ndarray 2d): A unified distance matrix
        """
        cluster_list = []
        for x_i in range(self.x_dim):
            for y_i in range(self.y_dim):
                centroid_x = centroids['centroid_x'][x_i, y_i]
                centroid_y = centroids['centroid_y'][x_i, y_i]
                if centroid_x == x and centroid_y == y:
                    cluster_list.append(np.matrix(umat)[x_i, y_i])
        return cluster_list

    @staticmethod
    def get_inter_clusters_distance_list(distance_dict=None):
        """ Converts inter clusters distance dict to a list.

        The key is a range(0,#cluster-1) and the values are the inter cluster mean distance.

        Args:
            distance_dict

        Returns:
            A list of inter cluster mean distance.
        """
        return list(distance_dict.values())

    @staticmethod
    def get_cell_lattices(size, bl_index):
        """ Computes the four lattice points' index of a cell

        Args:
            size (int): Indicates a size of a square mesh (size x size)
            bl_index (int): The bottom left lattice index of the cell

        Returns:
            A np.array of lattices indexes (4)
        """
        return np.array([bl_index, bl_index + 1, bl_index + size + 1, bl_index + size], dtype=np.int)

    @staticmethod
    def get_ruler_nodes(size, _from='bl'):
        """ Computes the nodes by excluding of top and right boundary nodes

        It helps to identifying the identical node of each cell.

        Args:
            _from (str): The diagonal start lattice point
                - bl: Bottom Left (/)
                - tl: Top left    (\)
            size (int): The mesh is a square and "size" is the  length of one of its sides.

        Returns:
            A list of grid constructive nodes
        """
        node_list = range(0, size ** 2)
        boundary_dict = SOM.get_boundary_sides(size)
        if _from == 'bl':
            exclude_list = boundary_dict.get('top') + boundary_dict.get('right') + boundary_dict.get('corners')
            include_list = [0]
        elif _from == 'tl':
            exclude_list = boundary_dict.get('bottom') + boundary_dict.get('right') + boundary_dict.get('corners')
            include_list = [size - 1]
        return include_list + list(set(node_list) - set(exclude_list))

    @staticmethod
    def get_diagonals(size, _from='bl'):
        """ Computes the diagonals indexes of all cells in a square grid

        \/
        /\

        Args:
            size (int): The mesh is a square and "size" is the  length of one of its sides.
            _from (str): The diagonal start lattice point {bl: Bottom Left , tl: Top left}

        Returns:
            A ndarray in size of (n_grid_cell, 2)
        """
        if _from == 'bl':
            start_nodes = SOM.get_ruler_nodes(size, _from=_from)
            end_nodes = np.add(start_nodes, size + 1).tolist()
        elif _from == 'tl':
            start_nodes = SOM.get_ruler_nodes(size, _from=_from)
            end_nodes = np.add(start_nodes, size - 1).tolist()
        return np.column_stack((start_nodes, end_nodes))

    @staticmethod
    def get_cells_lattices(size):
        """ Computes the lattice points of all cells in a square grid

        Args:
            size (int): The mesh is a square and "size" is the  length of one of its sides.

        Returns:
            A ndarray in size of (n_grid_cell, 4)  where n_grid_cell = (size -1)^2
        """
        ruler_nodes = SOM.get_ruler_nodes(size)
        v_get_cell_lattices = np.vectorize(SOM.get_cell_lattices, excluded=['size'], signature='()->(n)')
        return v_get_cell_lattices(size=size, bl_index=ruler_nodes)

    def get_cell_area(self, corners, feature_x=0, feature_y=1):
        """ Computes the cell (polygon) area

        The cell is indicated with the left bottom lattice

        Args:
            feature_x (str): Indicates the first feature
            feature_y (str): Indicates the second feature
            corners (array): A list of the corners points' indexes

        Returns:
            A float indicates the cell area
        """
        x = self.neurons[corners, self.get_feature_index(feature_x)]
        y = self.neurons[corners, self.get_feature_index(feature_y)]
        return Polygon(zip(x, y)).area

    def get_cell_init_area(self, corners, feature_x=0, feature_y=1):
        """ Computes the cell (polygon) initial area

        The cell is indicated with the left bottom lattice

        Args:
            feature_x (str): Indicates the first feature
            feature_y (str): Indicates the second feature
            corners (array): A list of the corners points' indexes

        Returns:
            A float indicates the cell area
        """
        x = self.initial_neurons_wight[corners, self.get_feature_index(feature_x)]
        y = self.initial_neurons_wight[corners, self.get_feature_index(feature_y)]
        return Polygon(zip(x, y)).area

    def get_neuron_dict(self, index_list):
        """ Make a subset of neurones wight dict

        Args:
            index_list (list(int)): List of neurones indexes

        Returns:
            Return a dict where the key is neuron index and value is the wight vector
        """
        return dict(zip(index_list, self.neurons[index_list]))

    def get_diagonal_neighbours(self, _index):
        """ Computes the diagonal neighbour lattices

        Args:
            _index (int): indicate a cell index

        Returns:
            A list include the ordinal index of the lattices
        """
        assert _index in range(0, self.x_dim ** 2), "ERROR: the neuron index out of the range"
        boundary_dict = self.get_boundary_sides(self.x_dim)
        if _index in boundary_dict.get('left'):
            diagonal_list = [_index + self.x_dim - 1, _index + self.x_dim + 1]
        elif _index in boundary_dict.get('top'):
            diagonal_list = [_index - self.x_dim - 1, _index + self.x_dim - 1]
        elif _index in boundary_dict.get('right'):
            diagonal_list = [_index - self.x_dim - 1, _index - self.x_dim + 1]
        elif _index in boundary_dict.get('bottom'):
            diagonal_list = [_index - self.x_dim + 1, _index + self.x_dim + 1]
        elif _index in boundary_dict.get('corners'):
            if _index == 0:
                diagonal_list = [self.x_dim + 1]
            elif _index == self.x_dim - 1:
                diagonal_list = [_index + self.x_dim - 1]
            elif _index == self.x_dim ** 2 - 1:
                diagonal_list = [_index - self.x_dim - 1]
            elif _index == self.x_dim ** 2 - self.x_dim:
                diagonal_list = [_index - self.x_dim + 1]
        else:
            diagonal_list = [_index - self.x_dim - 1, _index + self.x_dim - 1,
                             _index - self.x_dim + 1, _index + self.x_dim + 1]
        return diagonal_list

    # Todo: Breaking down hierarchical data with
    #  Treemap
    #  Sunburst charts
    #  Sankey Charts
    #  Waffle chart
    #  Bagplot


def __test_me():
    animal = ['dove', 'hen', 'duck', 'owl',
              'eagle', 'fox', 'dog', 'wolf',
              'cat', 'tiger', 'lion', 'horse', 'cow']
    attribute = [[1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                 [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                 [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
                 [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0],
                 [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                 [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
                 [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
                 [0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
                 [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
                 [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0],
                 [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]]
    attr = pd.DataFrame(attribute)
    attr.columns = ['small', 'medium', 'big', '2 legs',
                    '4 legs', 'hair', 'hooves', 'mane',
                    'feathers', 'hunt', 'run', 'fly', 'swim']
    m = SOM(map_shape=(8, 8))
    m.fit(attr, animal)


if __name__ == '__main__':
    __test_me()
