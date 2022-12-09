#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral feature selection methods that include Laplacian Score, MCFS and SPEC

This module includes the Unsupervised Spectral Feature Selection methods:
    - Laplacian score
    - Multi Cluster Feature Selection
    - Spec Feature Selection

We describe the main concept of Spectral feature selection in flowing. We have given a set of pairwise instance
similarity S, a graph G can be constructed to represent it. And the target concept specified in S is usually reflected
by the structure of G. A feature that is consistent with the graph structure assigns similar values to instances that
are near each other on the graph.

We modified the methods due to reduce the heavy computational cost of the original approaches by leverage of
Bootstrapping, ensemble learning, Resampling, multiprocess and etc.

Example:
    $ python3 spectral_methods.py config/config_FS_gromacs_64p.json ../../data/FS_test_data/imputed.csv -k 3 -m spec -bsize 2000 -r 5

See alos:
    - He, X., Cai, D., & Niyogi, P. (2006). Laplacian score for feature selection. In NIPS. MIT Press.
    - Deng Cai, Chiyuan Zhang, and Xiaofei He. Unsupervised feature selection for multi-cluster data.KDD, 2010
    - Zheng Zhao and Huan Liu. Spectral feature selection for supervised and unsupervised learning. In ICML, 2007

# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# Last update: 04/10/2022
"""

# TODO: the output of the eigen is needed to check.(the columns of the output are the eigen vectors)
# Fixme: remove the index from the feature list

import argparse
import heapq
import json
import math
import operator
import sys
import time
import warnings
from abc import ABCMeta, ABC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale
from terminaltables import DoubleTable
from multiprocessing import Process
from multiprocessing.managers import BaseManager
from kavica.mutual_knn import KNN
from kavica.distance_measure import rbf_kernel
from kavica.imputation.base import compatible_data_structure
from kavica.resampling import WightedBootstrapping
from kavica.feature_selection.base import FeatureSelection
from kavica.feature_selection.feature_analysis import pair_sort
from kavica.utils import BColors

__all__ = ['list_splitter',
           'gamma',
           'pair_sort',
           '__config',
           'arguments_parser',
           'mean_squared_displacement_plot',
           'plot_feature_important',
           'GraphUpdateAccelerator',
           'update_accelerator',
           '_BaseSpectralSelector',
           'LaplacianScore',
           'MultiClusterScore',
           'SPEC',
           'progress_bar'
           ]


def list_splitter(a_list, chunks=2):
    """ Split a list

    Args:
        a_list (list): Defined the input list.
        chunks (int): Indicates the numbers of the splits.

    Returns:
        A list of list includes the split lists.
    """
    # TODO: it has to be moved to the utility module
    length = len(a_list)
    if length < chunks:
        raise ValueError("It can't shrink the list into {} chunks, bigger than list size {}.".format(chunks, length))
    return [a_list[i * length // chunks: (i + 1) * length // chunks] for i in range(chunks)]


def gamma(X):
    """Compute the Gamma parameter

    The gamma parameter is the inverse of the standard deviation of the RBF kernel (Gaussian function),
    which is used as similarity measure between two points.
    𝛾=1/2𝜎^2

    Intuitively, the gamma parameter defines how far the influence of a single training example reaches, with low
    values meaning ‘far’ and high values meaning ‘close’. The gamma parameters can be seen as the inverse of the
    radius of influence of samples selected by the model as support vectors.

    Args:
        X (pandas DataFrame): Includes the input dataset.

    Returns:
        A float that represent the optimal Gamma parameter.

    Note:
        See Also: https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html
    """
    return 1 / (2 * X.std(axis=0).mean())


def __config(config, data):
    """Configure the initial data set.

    Args:
        config (str): Indicates the path of configuration file
        data (pandas): Includes the data set

    Returns:
        A pandas data frame that contains the prepared data.

    """
    with open(config, 'r') as config:
        config_dict = json.load(config)

    df = pd.read_csv(data)

    # config the data set based on configuration information
    df = df[list(config_dict['hardware_counters'].values())]  # sub set of features
    df = df.replace([np.inf, -np.inf], np.nan)
    lastShape = df.shape

    # Remove the all zero rows
    df = df[(df.T != 0).any()]
    print("The {} row are full null that are eliminated.".format(lastShape[0] - df.shape[0]))
    lastShape = df.shape

    # Remove all NaN columns.
    df = df.ix[:, (pd.notnull(df)).any()]
    print("The {} columns are full null that are eliminated.".format(lastShape[1] - df.shape[1]))

    if config_dict['missing_values'] == 'mean':
        df.fillna(df.mean(), inplace=True)

    if config_dict['scale']:
        df = pd.DataFrame(scale(df), index=df.index, columns=df.columns)
        print('The statistic is a characteristic of the SCALED sample')
        print(df.describe())

    df = df.reset_index()
    return df


def arguments_parser():
    """Parse the input arguments

    Returns:
        A dict includes the parsed arguments and their values.
    """
    # DevelopTime:It is used for testing and developing time.
    if len(sys.argv) == 1:  # set the arguments
        arguments = ['config/config_lulesh_27p.json',
                     '../parser/source1.csv',
                     '-k',
                     '2',
                     '-m',
                     'LS',
                     '-bsize',
                     '5000',
                     '-r',
                     '10'
                     ]
        sys.argv.extend(arguments)

    parser = argparse.ArgumentParser(description='The files that are needed for selecting features most important.')
    parser.add_argument('config', help='A .json configuration file that included the'
                                       'thread numbers,hardware counters and etc.')
    parser.add_argument('csvfile', help='A .csv dataset file')
    parser.add_argument('-k',
                        dest='k',
                        default=2,
                        action='store',
                        type=int,
                        help="It significances the number of the most important features.")
    parser.add_argument('-m',
                        dest='m',
                        default='LS',
                        choices=['LS', 'MCFS', 'SPEC'],
                        action='store',
                        type=str.upper,
                        help="The feature selection method that is either LS, MCFS or SPEC.")

    parser.add_argument('-bsize',
                        dest='bsize',
                        default=2500,
                        action='store',
                        type=int,
                        help="It indicates the 'Bag size' or 'ensemble size'.")

    parser.add_argument('-r',
                        dest='r',
                        default=10,
                        action='store',
                        type=int,
                        help="It indicates the Bootstrap Repetition.")

    args = parser.parse_args()

    if args.k < 2:
        raise ValueError("Selected features have to be (=> 2). It is set {}".format(args.k))

    return ({"configPath": args.config,
             "csvPath": args.csvfile,
             "k_features": args.k,
             "featureSelectionMethod": args.m,
             "bag_size": args.bsize,
             "repetition": args.r})


def mean_squared_displacement_plot(iteration_log_df, root=True):
    """Plot Mean Squared Displacement.

    We use this function in order to test_cp or visualising the progressive Mean Squared Displacement.

    Args:
        iteration_log_df (pandas): includes the position of the data points in the featrue space
        root (boolean): If True, the root mean squared displacement will be computed
    """
    msd = iteration_log_df.diff().apply(np.square).mean(axis=1)
    msd.index = msd.index + 1
    y_lab = "Mean Squared Displacement (MSD)"
    if root:
        msd = iteration_log_df.diff().apply(np.square).mean(axis=1).apply(np.sqrt)
        y_lab = "Root Mean Squared Displacement (RMSD)"

    plt.plot(msd.values, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel(y_lab)
    plt.title('Stabilization Curve of Bootstrap Feature Selection'.format(y_lab))
    plt.legend([y_lab])
    plt.grid()
    plt.show()


def plot_feature_important(score, reverse=True, ascending=True, k=2, output=None):
    """Bar plot of inverse/ inverse log

    Args:
        score (pandas series): Includes the feature scores.
        reverse (boolean): If True, we replace the score with 1/score.
        ascending (boolean): If True, it sorts the score ascending.
        k (int): Indicates the number of the features that we want to highlight.
        output (str): Indicates the output file name
    """
    score = score.sort_values(ascending=ascending)
    color_list = ['steelblue'] * (score.size - int(k))
    color_list.extend(['red'] * int(k))
    x_lab = "Score"
    if reverse:
        score = 1 / score
        x_lab = "Reverse Score"

    plt.figure(1, figsize=(8, 6))
    score.plot(kind='barh', color=color_list)
    plt.ylabel("Feature")
    plt.xlabel(x_lab)
    plt.title('Feature Importance Plot')
    plt.grid()
    plt.subplots_adjust(left=0.3)
    plt.savefig('feature_importance_plot_{}.jpg'.format(output))
    plt.show()


def progress_bar(counter, total, process_id=1, status='', bar_len=40, functionality=None):
    """ Shows and updates progressbar.

    Args:
        functionality:
        bar_len (int): Indicates the progressbar bar length.
        counter (int): Indicates the actual value
        total (int): Indicates the overall value
        status (str): Shows the more related information about the progressbar bar
        process_id (int): Indicates the process ID.

    Returns:
        Zero
    """
    filled_len = int(round(bar_len * counter / float(total)))
    percents = round(100.0 * counter / float(total), 1)
    bar = '|' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write(
        '\r\033[1;36;m[%s] <%s> chunk_id <%s> %s%s ...%s' % (bar, functionality, process_id, percents, '%', status))


class GraphUpdateAccelerator(object):
    """ Accelerates the graph edge updating with leverage of multiprocessing tech.

    Input: An adjacencylist (graph).
    Output: An adjacencylist (graph).

    Updating the graph edges with different proses
    """

    def __init__(self, adjacency_list, _gamma=None):
        """

        Args:
            adjacency_list (object adjacency list): Includes the KNN adjacency list.
            _gamma (float): Indicates the gamma (hyperparameter) for the rbf kernel.

        """
        self.adjacency_list = adjacency_list
        self.gamma = _gamma

    def set(self, data, process_id):
        """ Sets up the processes

        Args:
            data (list): Includes the indexes of the data frame rows that assigned to a process_id
            process_id (int): Indicates the process ID

        Returns:
            self
        """
        progress_line_len = len(data)
        actual_progress = 0
        for v in data:
            actual_progress += 1
            progress_bar(actual_progress, progress_line_len, process_id,
                         status=str(actual_progress) + "/" + str(progress_line_len) + "    ",
                         functionality='Update by RBf_kernel')
            v = self.adjacency_list.get_vertex(str(v))
            vid = v.get_id()
            for w in v.get_connections():
                wid = w.get_id()
                euclidean_weight = v.get_weight(w)
                rbf_weight = rbf_kernel(pre_distance=euclidean_weight, gamma=self.gamma)
                self.adjacency_list.update_edge(str(vid), str(wid), rbf_weight, smallest=False)
        return self

    def get(self):
        """ Return the adjacency list

        Returns:
            A adjacency list.
        """
        return self.adjacency_list


def update_accelerator(obj, items, process_id):
    """Updates an accelerator object.

    Args:
        obj (object accelerator): Includes the adjacency list.
        items (list): Includes the rows index that is assigned to a proses.
        process_id (int): Is ID of the proses.

    Returns:
        An updated accelerator object.
    """
    obj.set(items, process_id)
    return obj


class _BaseSpectralSelector(FeatureSelection, metaclass=ABCMeta):
    """Base spectral feature selection class
            - Generate the KNN graph and matrix.
            - Calculate the RBF kernel values and update the KNN graph
    """

    def __init__(self, X=None, method=None, k=2, bag_size=3000):
        """
        Args:
            X (pandas): Includes the data set.
            method (str): Indicates the feature selection method
            k (int): Indicates the number of selected features.
            bag_size (int): Indicates the bootstrap bag size.
        """

        self.has_fitted = False
        self.adjacency_list = KNN()
        self.origin_data = X
        self.adjacency_matrix = None
        self.original_index = None
        self.default_boostrap_bag_size = bag_size
        self.feature_score = {'method': method, 'features': None, 'scores': np.array([])}
        self.k = k

    # Fixme: Too big function
    def fit(self, X, adjacency_matrix=True, parallel=True,
            recursion_limit=None, multi_process=None, bag_size=None):
        """Run KNN on the X and obtain the Adjacency List.

        Args:
            X (pandas): Is the input data set.
            adjacency_matrix (boolean): If True, the adjacency Matrix is generated.
            parallel (boolean): If True, the data distributed among multi_process.
            recursion_limit (int): Is used in order to set recursion limit.
            multi_process (int): indicates the number of the multi process
            bag_size (int): indicates the bootstrap bag size.

        Returns:
            self
        """
        if bag_size:
            self.default_boostrap_bag_size = bag_size
        # fixme: it is a duplicated action
        self.origin_data = compatible_data_structure(data=X,
                                                     header=True,
                                                     index=True)
        # make copy of the old index (for distributed matching)
        if 'index' in self.origin_data.columns:
            # After pandas 0.21.0 : columns=['index']
            self.origin_data = self.origin_data.drop(['index'], axis=1)

        # Re_sampling from the data, the high duration is more probable for selecting.
        # TODO: Implement an stochastic method
        """
        Implement an stochastic method for selecting from different bags.
        (the one that is more close to the original data)
        Know, we just generate one bag.
        """
        if self.origin_data.shape[0] < self.default_boostrap_bag_size:
            self.default_boostrap_bag_size = self.origin_data.shape[0]
        else:
            pass
        try:
            self.origin_data = WightedBootstrapping.weighted_resampling(x=self.origin_data,
                                                                        bag_size=self.default_boostrap_bag_size,
                                                                        replace=True,
                                                                        bags=1,
                                                                        weight='Duration').get('0')
        except ValueError:
            # In case of:Invalid weights: weights sum to zero
            self.origin_data = WightedBootstrapping.weighted_resampling(x=self.origin_data,
                                                                        bag_size=self.default_boostrap_bag_size,
                                                                        replace=True,
                                                                        bags=1).get('0')
        except KeyError:
            # In case of:Invalid weights: weights sum to zero
            self.origin_data = WightedBootstrapping.weighted_resampling(x=self.origin_data,
                                                                        bag_size=self.default_boostrap_bag_size,
                                                                        replace=True,
                                                                        bags=1).get('0')
        finally:
            self.origin_data.reset_index(inplace=True)  # reset index is needed, some indexes are missed
            self.original_index = self.origin_data['index'].copy()
            if 'Duration' in self.origin_data.columns:
                self.origin_data = self.origin_data.drop(axis=1, labels=['Duration'])
            # After pandas 0.21.0 : columns=['index']
            self.origin_data = self.origin_data.drop(axis=1, labels=['index'])

        # fixme: it is obligatory to make the data standardize, it should move to data pre-processing
        self.origin_data = pd.DataFrame(scale(self.origin_data,
                                              with_mean=True,
                                              with_std=True,
                                              copy=False),
                                        index=self.origin_data.index,
                                        columns=self.origin_data.columns)

        # Initiate the feature rank list that will updated by the Specific methods
        self.feature_score['features'] = np.array(self.origin_data.columns.tolist())
        self._check_params(self.origin_data)

        self.adjacency_list.fit(self.origin_data,
                                adjacency_matrix=False,
                                header=True,
                                index=True)

        gammaValue = gamma(self.origin_data)

        # TODO: Combine with filter and ensemble
        # TODO: Matrix product version of rbf_kernel ????? It will be faster in Euclidean one.

        '''
        Alternative of rbf_kernel:
        rbf_kernel_matrix=f = lambda x: np.exp(x**2*(-gammaValue))
        rbf_kernel_matrix(self.adjacencyList.graph_to_matrix())
        '''

        if parallel:
            # TODO: Use multiprocess + HDF5 here
            if recursion_limit is None:
                # TODO: It is needed to find an optimal values for. it.
                recursion_limit = self.origin_data.shape[0] ** 2
                warnings.warn(
                    "\nThe recursion_limit is set to {} automatically.".format(recursion_limit, UserWarning))
            else:
                warnings.warn("\nThe recursion_limit is set to {} manually.".format(recursion_limit, UserWarning))
            sys.setrecursionlimit(recursion_limit)

            if multi_process is None:
                # TODO: It is needed to calculate the optimal chunk number.
                chunk_number = 10
                warnings.warn("The multi_process is set to {} by default.".format(chunk_number, UserWarning))
            else:
                chunk_number = multi_process

            BaseManager.register('FastUpdate', GraphUpdateAccelerator)
            manager = BaseManager()
            manager.start()
            temporal_knnGraph = manager.FastUpdate(self.adjacency_list.knnGraph, _gamma=gammaValue)

            # TODO: rewrite it as a module.

            chunks = list_splitter(list(self.adjacency_list.knnGraph.vert_dict.keys()), chunks=chunk_number)
            processes = [Process(target=update_accelerator, args=[temporal_knnGraph, chunks[chunk_id], chunk_id]) for
                         chunk_id in range(0, chunk_number)]

            # Run processes
            for p in processes:
                p.start()

            # Exit the completed processes
            for p in processes:
                p.join()
                if p.is_alive():
                    print("Job {} is not finished!".format(p))
            # Gather the data from the different presses
            self.adjacency_list.knnGraph = temporal_knnGraph.get()

        else:
            progress_line_len = len(self.adjacency_list.knnGraph.vert_dict) * self.adjacency_list.neighbors
            actual_progress = 0
            print('\n')
            for v in self.adjacency_list.knnGraph:
                vid = v.get_id()
                for w in v.get_connections():
                    actual_progress += 1
                    progress_bar(actual_progress,
                                 progress_line_len,
                                 1,
                                 status=str(actual_progress) + "/" + str(progress_line_len),
                                 functionality='Update by RBf_kernel')
                    wid = w.get_id()
                    '''
                    Old rbf: rbf_kernel(X=self.originData.loc[int(vid)], 
                                        Y=self.originData.loc[int(wid)], 
                                        gamma=gammaValue)
                    '''
                    euclidean_weight = v.get_weight(w)
                    rbf_weight = rbf_kernel(pre_distance=euclidean_weight, gamma=gammaValue)
                    self.adjacency_list.knnGraph.update_edge(str(vid),
                                                             str(wid),
                                                             rbf_weight,
                                                             smallest=False)

        if adjacency_matrix:
            self.adjacency_matrix = self.adjacency_list.graph_to_matrix(binary=False)
        self.has_fitted = True
        return self

    def _sorted_features(self, order=-1):
        """Sort features (By default Descending).

        Args:
            order (int [-1,1]): If it is -1 the feature list is sorted Sort Descending.

        Returns:
            A dict includes the sorted feature list and scores.
        """
        index = np.array(self.feature_score['scores']).argsort(kind='quicksort')
        return {'sorted_features': self.feature_score['features'][index][::order],
                'sorted_scores': self.feature_score['scores'][index][::order],
                'ordered': order}

    def feature_score_table(self):
        """Print feature score as a table

        Returns:
            A terminaltables object includes the feature and the scores.
        """
        sorted_feature_score = self._sorted_features()
        if sorted_feature_score.get('ordered') == 1:
            sort_arrow = '\u2191'
        elif sorted_feature_score.get('ordered') == -1:
            sort_arrow = '\u2193'
        else:
            raise ValueError("The ordered direction has to be ascending or descending.")

        table_data = [['Rank', 'Feature', str('Score ' + sort_arrow)]]

        for rank, featureItem in enumerate(sorted_feature_score['sorted_features']):
            table_data.append([rank, featureItem,
                               sorted_feature_score['sorted_scores'][rank]])
        table = DoubleTable(table_data,
                            title='{}'.format(str.upper(self.feature_score['method'])))
        table.justify_columns[2] = 'center'
        return table

    def _check_params(self, X):
        # TODO: write down the parameter checker
        pass


class LaplacianScore(_BaseSpectralSelector, ABC):
    """ Class to rank features according to the smallest Laplacian scores.

        See also:
            https://papers.nips.cc/paper/laplacian-score-for-feature-selection.pdf
    """

    # TODO: It is needed to add progressbar bar to flow the process.
    def __init__(self, X=None, k=None):
        """
        Args:
            X (pandas): Includes the input data.
            k (int): Indicates the number of the top feature that is needed to be selected.
        """
        super(LaplacianScore, self).__init__(X, method='LaplacianScore', k=k)

    def _sorted_features(self, order=1):
        """Sort features (By default Descending).

        Args:
            order (int [-1,1]): If it is -1 the feature list is sorted Sort Descending.

        Returns:
            A dict includes the sorted feature list and scores.
        """
        return super(LaplacianScore, self)._sorted_features(order=order)

    def rank_features(self, X=None, absolute_score=True):
        """Ranks the features based on the Laplacian score

        Args:
            absolute_score (boolean): If True, the absolute score takes to account.
            X (pandas): Includes the data set.

        Returns:
            self
        """
        if X is not None:
            self.fit(X)
        elif self.has_fitted:
            pass
        else:
            raise ValueError('The model has not fitted and the X is None')

        degree_matrix = np.array(self.adjacency_matrix.sum(axis=1))
        graph_Laplacian = np.subtract(np.diag(degree_matrix), self.adjacency_matrix)
        for feature in self.origin_data.columns:
            feature_vector = np.array(self.origin_data[feature].tolist())
            feature_rhat = np.array(feature_vector
                                    - (np.dot(feature_vector, degree_matrix)
                                       / degree_matrix.sum()))
            # todo: check the functionality of transpose
            featureLaplacianScore = np.dot(np.dot(feature_rhat, graph_Laplacian),
                                           feature_rhat.transpose())
            if absolute_score:
                self.feature_score['scores'] = np.append(self.feature_score['scores'],
                                                         np.absolute(featureLaplacianScore))
            else:
                self.feature_score['scores'] = np.append(self.feature_score['scores'],
                                                         featureLaplacianScore)
        return self

    def _check_params(self, X):
        # TODO: write down the parameter checker
        pass


class MultiClusterScore(_BaseSpectralSelector, ABC):
    """ Class to rank the features according to the highest Multi-Cluster Score.

    See also:
        https://www.cad.zju.edu.cn/home/dengcai/Publication/Conference/Multi-transfer_learning-feature-selection.pdf
    """

    def __init__(self, X=None, k=None, d=None):
        """

        Args:
            X (pandas): Includes the input data.
            k (int): Indicates the number of the top feature that is needed to be selected.
            d (int): Indicates the number of the multi clusters.
        """
        # TODO: add the d as argument to the input list.
        super(MultiClusterScore, self).__init__(X, method='Multi-Cluster', k=k)
        self.k_clusters = k
        self.selected_features = d

    def __k_cluster_estimator(self):
        """Select the optimal value of the multi clusters.
        Returns:
            An int represented the number of the multi cluster

        """
        k = round(math.sqrt(self.origin_data.shape[0]))
        if math.fmod(k, 2) == 1:
            return k
        else:
            return k + 1

    def __update_scores(self, coefficient_vector, absolute_score=True):
        """

        Args:
            coefficient_vector (numpy array): Includes the feature scores
            absolute_score (Boolean): If True it computes the absolut value of the score
        Returns:
            self
        """
        # TODO: make the output colorized if the value is changed in any iteration
        if absolute_score:
            coefficient_vector = np.absolute(coefficient_vector)

        if not len(self.feature_score['scores']):
            self.feature_score['scores'] = coefficient_vector
        for index, featureScoreItem in enumerate(self.feature_score['scores']):
            self.feature_score['scores'][index] = max(featureScoreItem,
                                                      coefficient_vector[index])
        return self

    def rank_features(self, X=None):

        if X is not None:
            self.fit(X)
        elif self.has_fitted:
            pass
        else:
            raise ValueError('The model has not fitted and the X is None')

        degree_matrix = np.array(self.adjacency_matrix.sum(axis=1))

        graph_laplacian = np.subtract(np.diag(degree_matrix), self.adjacency_matrix)

        # Calculate spectral Decomposition of graph Laplacian
        graph_laplacian = np.dot(np.linalg.inv(graph_laplacian), graph_laplacian)
        eigen_values, eigen_vectors = np.linalg.eigh(graph_laplacian)

        # The eigen values have to be abs()
        eigen_values = np.abs(eigen_values)

        # TODO: it should move to _check_parameter
        with warnings.catch_warnings():
            warnings.simplefilter("default", UserWarning)

            # Initiate the k
            if self.k_clusters is None:
                # fixme: the k estimator have to be writen
                self.k_clusters = self.__k_cluster_estimator()
                warnings.warn('\n The k parameter has not indicated, It is set automatically to {}.'
                              .format(self.k_clusters), UserWarning, stacklevel=2)
            elif self.k_clusters > len(eigen_values):
                raise ValueError("k (multi-clusters) > {} flat embedding vectors.".format(len(eigen_values)))

            # Initiate the d
            if self.selected_features is None:
                self.selected_features = self.origin_data.shape[1]
                print("\n")
                warnings.warn('The d selected Features has not indicated, It is set automatically to {}.'
                              .format(self.selected_features), UserWarning, stacklevel=2)
            elif self.selected_features > self.origin_data.shape[1]:
                print("\n")
                raise ValueError(
                    'The d selected Features > {} flat embedding vectors.'.format(self.origin_data.shape[1]))

        eigens = dict(zip(eigen_values.real, eigen_vectors))
        eigens = dict(sorted(eigens.items(),
                             key=operator.itemgetter(0),
                             reverse=True))  # sort inplace

        # Solve the L1-regularized regressions K time
        reg = linear_model.Lars(n_nonzero_coefs=self.selected_features)
        for eigenItem in range(self.k_clusters):
            _, vector = eigens.popitem()
            reg.fit(self.origin_data, vector)
            self.__update_scores(np.array(reg.coef_))

    def _check_params(self, X):
        # TODO: write down the parameter checker
        pass


class SPEC(_BaseSpectralSelector, ABC):
    """ Base class to rank the features according to the highest  Laplacian scores.

        - Separability_scores and Normalized_Separability_scores are Ascending
        - and K_cluster_Separability_scores is Descending.

    See also
        https://papers.nips.cc/paper/laplacian-score-for-feature-selection.pdf
    """

    # TODO: rewrite the feature sorting function that can accept the sorting parameter

    def __init__(self, X=None, k=None, k_clusters=2):
        super(SPEC, self).__init__(X, method='SPEC', k=k)
        # TODO: I need an optimiser function to select the optimal value.
        self.k_clusters = k_clusters
        self.feature_score = {'method': 'SPEC',
                              'features': None,
                              'Separability_scores': np.array([]),
                              'Normalized_Separability_scores': np.array([]),
                              'K_cluster_Separability_scores': np.array([])}

    def _sorted_features(self, order=1, sort_by="Separability_scores"):
        """Sort features (By default Descending).

        Args:
            order (int [-1,1]): If it is -1 the feature list is sorted Sort Descending.
            sort_by (str): Indicates a score metric
                            - Separability_scores
                            - Normalized_Separability_scores
                            - K_cluster_Separability_scores
        Returns:
            A dict includes the sorted feature list and scores.
        """
        if sort_by == "Separability_scores":
            index = np.array(self.feature_score['Separability_scores']).argsort(kind='quicksort')
        elif sort_by == "Normalized_Separability_scores":
            index = np.array(self.feature_score['Normalized_Separability_scores']).argsort(kind='quicksort')
        elif sort_by == "K_cluster_Separability_scores":
            index = np.array(self.feature_score['K_cluster_Separability_scores']).argsort(kind='quicksort')
            order = -1
        else:
            raise ValueError('The score {} is not fined.(Separability_scores, '
                             'Normalized_Separability_scores, '
                             'K_cluster_Separability_scores)'.format(sort_by))

        return {'sorted_features': self.feature_score['features'][index][::order],
                'sorted_Separability': self.feature_score['Separability_scores'][index][::order],
                'sorted_Normalized_Separability': self.feature_score['Normalized_Separability_scores'][index][::order],
                'sorted_K_cluster_Separability': self.feature_score['K_cluster_Separability_scores'][index][::order],
                'sort_by': sort_by}

    def feature_score_table(self):
        """Print feature score as a table

         Returns:
             A terminaltables object includes the feature and the scores.
         """
        sorted_feature_score = self._sorted_features()
        table_data = [
            ['Rank',
             'Feature',
             'Separability_scores \u2191',
             'Normalized_Separability_scores',
             'K_cluster_Separability_scores']
        ]
        for rank, featureItem in enumerate(sorted_feature_score['sorted_features']):
            table_data.append([rank,
                               featureItem,
                               sorted_feature_score['sorted_Separability'][rank],
                               sorted_feature_score['sorted_Normalized_Separability'][rank],
                               sorted_feature_score['sorted_K_cluster_Separability'][rank]])
        table = DoubleTable(table_data, title='{}'.format(str.upper(self.feature_score['method'])))
        table.justify_columns[2] = 'center'
        return table

    def rank_features(self, X=None, absolute_score=True):
        """Ranks the features based on the SPEC score

        Args:
            absolute_score (boolean): If True, the absolute score takes to account.
            X (pandas): Includes the data set.

        Returns:
            self
        """
        # Fixme: it is too big function
        if X is not None:
            self.fit(X)
        elif self.has_fitted:
            pass
        else:
            raise ValueError('The model has not fitted and the X is None')

        degree_matrix = np.array(self.adjacency_matrix.sum(axis=1))

        graph_laplacian = np.subtract(np.diag(degree_matrix), self.adjacency_matrix)

        # normalized graph Laplacian (alias name is used for memory efficiency)
        normalized_graph_laplacian = graph_laplacian = np.power(degree_matrix, -0.5) \
                                                       * graph_laplacian \
                                                       * np.power(degree_matrix, -0.5)[:, np.newaxis]
        del graph_laplacian  # graph_laplacian is not valid anymore.

        # Calculate spectral Decomposition of normalized graph Laplacian
        eigen_values, eigen_vectors = np.linalg.eigh(np.dot(np.linalg.inv(normalized_graph_laplacian),
                                                            normalized_graph_laplacian))
        # TODO: the eigen values have to be abs()?
        eigen_values = np.abs(eigen_values)
        micro_density_indicator = eigen_vectors[np.argmax(eigen_values)]
        eigen_values, eigen_vectors = pair_sort(eigen_values, eigen_vectors, order=-1)

        for feature in self.origin_data.columns:
            feature_vector = np.array(self.origin_data[feature].tolist())
            feature_vector_tilda = np.sqrt(degree_matrix) * feature_vector
            feature_vector_hat = feature_vector_tilda / feature_vector_tilda.sum()

            # TODO: it needs to check the calculation with Cosine similarity
            # Ranking Function 1: the value of the normalized cut (Shi & Malik, 1997) - ascending
            graph_separability = np.dot(np.dot(feature_vector_hat, normalized_graph_laplacian),
                                        feature_vector_hat.transpose())

            # Ranking Function 2: use spectral eigen_values to normalize the Ranking Function 1. - ascending
            normalized_graph_separability = graph_separability / (
                    1 - np.dot(feature_vector_hat, micro_density_indicator))

            # Ranking Function 3: If the k (number of clusters) is indicated, it should use the reducing noise.
            if self.k_clusters is not None:
                k_graph_separability = 0
                for eigen_value_item, eigen_vector_item in heapq.nlargest(self.k_clusters,
                                                                          zip(eigen_values, eigen_vectors)):
                    k_graph_separability += eigen_value_item * np.power(
                        cosine_similarity([feature_vector], [eigen_vector_item]), 2)
                if absolute_score:
                    self.feature_score['K_cluster_Separability_scores'] = np.append(
                        self.feature_score['K_cluster_Separability_scores'],
                        np.absolute(k_graph_separability))
                else:
                    self.feature_score['K_cluster_Separability_scores'] = np.append(
                        self.feature_score['K_cluster_Separability_scores'],
                        k_graph_separability)

            # Update the score list
            if absolute_score:
                self.feature_score['Separability_scores'] = np.append(
                    self.feature_score['Separability_scores'],
                    np.absolute(graph_separability))
                self.feature_score['Normalized_Separability_scores'] = np.append(
                    self.feature_score['Normalized_Separability_scores'],
                    np.absolute(normalized_graph_separability))
            else:
                self.feature_score['Separability_scores'] = np.append(
                    self.feature_score['Separability_scores'],
                    graph_separability)
                self.feature_score['Normalized_Separability_scores'] = np.append(
                    self.feature_score['Normalized_Separability_scores'],
                    normalized_graph_separability)

    def _check_params(self, X):
        # TODO: write down the parameter checker
        pass


def __test_me():
    # sample data
    '''
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
                      (9, 12, 0, 9, 5, 20, 89),
                      (10, 6, 7, 8, 3, 8, 2)])

    headers = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    index = [1, 2, 3, 4]
    df = pd.DataFrame(data, columns=headers, index=index, dtype=np.float)
    '''

    df = __config('config/config_FS_lulesh_27p.json', '../parser/source.csv')

    # feature selection
    test2 = SPEC(k=2)
    test2.fit(df)
    test2.rank_features(df)
    print(test2.feature_score)


def __select_feature():
    """ Applies the feature selection through the command-line

    Returns:

    Note:
        AdaBoost (Adaptive Boosting) Bootstrap is our approach to control and check the stability of the results.
        Hence, combining feature selection models may not necessarily beat the performance of the best feature subset in
        the ensemble, but it certainly reduces the overall risk of making a particularly poor selection.

    """
    # Fixme: Such big code block
    start = time.time()
    args = arguments_parser()
    repetitions = int(args['repetition'])
    df = __config(args['configPath'], args['csvPath'])

    if args['featureSelectionMethod'] == 'LS':
        _fs_model = LaplacianScore(k=args['k_features'])
    elif args['featureSelectionMethod'] == 'MCFS':
        _fs_model = MultiClusterScore(k=args['k_features'])
    elif args['featureSelectionMethod'] == 'SPEC':
        _fs_model = SPEC(k=args['k_features'])
    else:
        raise ValueError(
            BColors.FAIL + " The method -m {} is not defined.".format(args['featureSelectionMethod']) + BColors.ENDC)

    value_error_threshold = 64
    if _fs_model.feature_score.get('method') == 'SPEC':
        _separability_log = pd.DataFrame()
        _normalized_separability_log = pd.DataFrame()
        _k_cluster_separability_log = pd.DataFrame()
        for repetition in range(1, repetitions + 1):
            value_error_counter = 0
            print("\n\033[32mThe Bootstrap repetition {}/{}".format(repetition, repetitions), "\033[m")
            while True:
                try:
                    _fs_model.__init__(k=args['k_features'])
                    _fs_model.fit(df, bag_size=args['bag_size'])
                    # fixme: when the data is fitted it dose not need to refit here
                    _fs_model.rank_features()
                    _separability_log = _separability_log.append(
                        dict(zip(_fs_model.feature_score.get('features'),
                                 _fs_model.feature_score.get('Separability_scores'))), ignore_index=True)
                    _normalized_separability_log = _normalized_separability_log.append(
                        dict(zip(_fs_model.feature_score.get('features'),
                                 _fs_model.feature_score.get('Separability_scores'))), ignore_index=True)
                    _k_cluster_separability_log = _k_cluster_separability_log.append(
                        dict(zip(_fs_model.feature_score.get('features'),
                                 _fs_model.feature_score.get('Separability_scores'))), ignore_index=True)
                    break
                except np.linalg.LinAlgError as _error:
                    if 'Singular matrix' in str(_error):
                        pass
                    else:
                        raise
                except ValueError as _error:
                    value_error_counter += 1
                    if value_error_counter <= value_error_threshold:
                        print("The Error has occurred {}/{}, the sampling is repeated".format(value_error_counter,
                                                                                              value_error_threshold))
                    else:
                        raise ValueError("The data set is not appropriate for the {} feature selection".format(
                            _fs_model.feature_score.get('method')))

        _mean_separability = _mean_feature_score = _separability_log.mean(axis=0).sort_index()
        _mean_normalized_separability = _normalized_separability_log.mean(axis=0).sort_index()
        _mean_k_cluster_separability = _k_cluster_separability_log.mean(axis=0).sort_index()

        _fs_model.feature_score['features'] = _mean_separability.index
        _fs_model.feature_score['Separability_scores'] = _mean_separability.values
        _fs_model.feature_score['Normalized_Separability_scores'] = _mean_normalized_separability.values
        _fs_model.feature_score['K_cluster_Separability_scores'] = _mean_k_cluster_separability.values
    else:
        _feature_score_log = pd.DataFrame()
        for repetition in range(1, repetitions + 1):
            value_error_counter = 0
            print("\n\033[32mThe Bootstrap repetition {}/{}".format(repetition, repetitions), "\033[m")
            while True:
                try:
                    _fs_model.__init__(k=args['k_features'])
                    _fs_model.fit(df, bag_size=args['bag_size'])
                    # fixme: when the data is fitted it doesn't need to refit here
                    _fs_model.rank_features()

                    _feature_score_log = _feature_score_log.append(
                        dict(zip(_fs_model.feature_score.get('features'),
                                 _fs_model.feature_score.get('scores'))), ignore_index=True)
                    break
                except np.linalg.LinAlgError as _error:
                    if 'Singular matrix' in str(_error):
                        pass
                    else:
                        raise
                except ValueError as _error:
                    value_error_counter += 1
                    if value_error_counter <= value_error_threshold:
                        print("ValueError occurred {}/{}: Resample!".format(value_error_counter, value_error_threshold))
                        pass
                    else:
                        raise ValueError("The data set is not appropriate for the {} feature selection".format(
                            _fs_model.feature_score.get('method')))

        _mean_feature_score = _feature_score_log.mean(axis=0)
        _fs_model.feature_score['features'] = _mean_feature_score.index
        _fs_model.feature_score['scores'] = _mean_feature_score.values

    # plot
    _model = args['featureSelectionMethod']
    _ascending = {'SPEC': False, 'MCFS': True, 'LS': False}
    _reverse = {'SPEC': True, 'MCFS': False, 'LS': True}
    plot_feature_important(score=_mean_feature_score, ascending=_ascending.get(_model),
                           reverse=_reverse.get(_model), k=_fs_model.k, output=_model)

    print("\n")
    print(_fs_model.feature_score_table().table)
    print("\033[32mSuccessfully completed \033[1m{}\033[0m.".format(_fs_model.feature_score.get("method")))
    print('\033[0mTotal duration is: %.3f' % (time.time() - start))


if __name__ == '__main__':
    # __test_me()
    __select_feature()
