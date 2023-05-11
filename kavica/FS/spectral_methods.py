#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral feature selection methods that include Laplacian score, MCFS and SPEC

--------------------------------------------------------------------------------------------------------------------
References:
    - He, X., Cai, D., & Niyogi, P. (2006). Laplacian score for feature selection. In NIPS. MIT Press.
    - Deng Cai, Chiyuan Zhang, and Xiaofei He. Unsupervised feature selection for multi-cluster data.KDD, 2010
    - Zheng Zhao and Huan Liu. Spectral feature selection for supervised and unsupervised learning. In ICML, 2007
--------------------------------------------------------------------------------------------------------------------
"""
# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause

# TODO: the output of teh eigen is needed to check. the columns of the output are the eigen vectors.
# Fixme: remove the index from teh feature list

import argparse
import heapq
import json
import math
import operator
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import scale
from terminaltables import DoubleTable

from kavica.Mutual_knn import KNN
from kavica.distance_measure import rbf_kernel
from kavica.imputation.base import data_structure_Compatibilization
from kavica.resampling import WightedBootstrapping

from multiprocessing import Process
from multiprocessing.managers import BaseManager


# TODO: it has to be moved to the utility module
def list_splitter(alist, chunks=2):
    length = len(alist)
    if length < chunks:
        raise ValueError("The list can not be splitter into {} chunks that is bigger than list size {}.".format(chunks,
                                                                                                                length))
    return [alist[i * length // chunks: (i + 1) * length // chunks]
            for i in range(chunks)]


def gamma(X):
    # default is 1.0/2std
    return 1 / (2 * X.std(axis=0).mean())


# TODO: it needed to write to eliminate the redundancy
def has_fitted(estimator, attributes, msg=None, all_or_any=any):
    pass


def sort_parja(x, y, order=-1):
    # TODO: parameter check (numpy array)
    index = np.array(x).argsort(kind='quicksort')
    return (np.array(x)[index][::order], np.array(y)[index][::order])


# read the configuration file for preparing the features
def __configoration(config, data):
    # read the configuration file
    with open(config, 'r') as config:
        config_dict = json.load(config)

    # Read the data file
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
    print(df.mean(axis=0), df.std(axis=0))

    # fixme: it is just reset the indexing for test
    df = df.reset_index()

    return df


def progress(count, total, status=''):
    bar_len = 100
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r\033[1;36;m[%s] %s%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()


def arguments_parser():
    # set/receive the arguments
    if len(sys.argv) == 1:
        # It is used for testing and developing time.
        arguments = ['config/config_FS_gromacs_64p_INS_CYC.json',
                     '../parser/source.csv',
                     '-k',
                     '2',
                     '-m',
                     'LS',
                     '-bsize',
                     '2500'
                     ]
        sys.argv.extend(arguments)
    else:
        pass

    # parse the arguments
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
                        help="It significances the 'Bag size' or 'ensemble size'.")

    args = parser.parse_args()

    if args.k < 2:
        raise ValueError("Selected features have to be (=> 2). It is set {}".format(args.k))

    return ({"configPath": args.config,
             "csvPath": args.csvfile,
             "k_features": args.k,
             "featureSelectionMethod": args.m,
             "bag_size": args.bsize})


######################################################################
# Graph weighting functions
######################################################################
def _identity():
    return 1


def dot_product(X, Y):
    return np.dot(X, Y)


######################################################################
# Accelerator
######################################################################
class GraphUpdateAccelerator(object):
    """
    It accelerates the graph edge updating with leverage of multiprocessing technology.
    Input: An adjacencylist (graph).
    Output: An adjacencylist (graph).
    Updating the graph edges with different proses
    """

    def __init__(self, adjacency_list, gama=None):
        self.adjacency_list = adjacency_list
        self.gama = gama

    @staticmethod
    def progress_bar(counter, total, process_id=1, status='', functionality=None):
        bar_len = 40
        filled_len = int(round(bar_len * counter / float(total)))
        percents = round(100.0 * counter / float(total), 1)
        bar = '|' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write(
            '\r\033[1;36;m[%s] <%s> chunk_id <%s> %s%s ...%s' % (bar,
                                                                 functionality,
                                                                 process_id,
                                                                 percents,
                                                                 '%',
                                                                 status))
        sys.stdout.flush()
        return 0

    def set(self, data, process_id):
        progress_line_len = len(data)
        actual_progress = 0
        for v in data:
            actual_progress += 1
            self.progress_bar(actual_progress,
                              progress_line_len,
                              process_id,
                              status=str(actual_progress) + "/" + str(progress_line_len) + "    ",
                              functionality='Update by RBf_kernel')
            v = self.adjacency_list.get_vertex(str(v))
            vid = v.get_id()
            for w in v.get_connections():
                wid = w.get_id()
                euclidean_weight = v.get_weight(w)
                rbf_weight = rbf_kernel(pre_distance=euclidean_weight, gamma=self.gama)
                self.adjacency_list.update_edge(str(vid),
                                                str(wid),
                                                rbf_weight,
                                                smallest=False)

    def get(self):
        return self.adjacency_list


def update_accelerator(obj, items, process_id):
    obj.set(items, process_id)


######################################################################
# Base class
######################################################################
class _BaseSpectralSelector(object):
    """Initialize the spectral feature selection.
            - Generate the KNN graph and matrix.
            - Calculate the RBF kernel values and update the KNN graph
    Parameters
    """

    def __init__(self, X=None, method=None, default_boostrap_bag_size=4500):
        self.hasFitted = False
        self.adjacencyList = KNN()
        self.originData = X
        self.adjacencyMatrix = None
        self.original_index = None
        self.default_boostrap_bag_size = default_boostrap_bag_size
        self.featureScoure = {'method': method, 'features': None, 'scores': np.array([])}

    @staticmethod
    def progress_bar(counter, total, process_id=1, status='', functionality=None):
        bar_len = 40
        filled_len = int(round(bar_len * counter / float(total)))
        percents = round(100.0 * counter / float(total), 1)
        bar = '|' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write(
            '\r\033[1;36;m[%s] <%s> chunk_id <%s> %s%s ...%s' % (bar,
                                                                 functionality,
                                                                 process_id,
                                                                 percents,
                                                                 '%',
                                                                 status))

    def fit(self, X, adjacencyMatrix=True, parallel=True,
            recursion_limit=None, multi_process=None, bag_size=None):
        """Run KNN on the X and obtain the adjacencyList.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        adjacencyMatrix:
        parallel:
        recursion_limit:
        multi_process:
        bag_size:

        Returns
        -------
        self : object
        """
        if bag_size:
            self.default_boostrap_bag_size = bag_size
        # fixme: it is a duplicated action
        self.originData = data_structure_Compatibilization(data=X,
                                                           header=True,
                                                           index=True)
        # make copy of the old index (for distributed matching)
        self.originData = self.originData.drop(axis=1, labels=['index'])  # After pandas 0.21.0 : columns=['index']

        # Resampling from the data, the high duration is more probable for selecting.
        # TODO:
        """
        Implement an stochastic method for selecting from different bags.
        (the one that is more close to the original data)
        Know, we just generate one bag.
        """
        if self.originData.shape[0] < self.default_boostrap_bag_size:
            self.default_boostrap_bag_size = self.originData.shape[0]
        else:
            pass
        try:
            self.originData = WightedBootstrapping.wighted_resampeling(X=self.originData,
                                                                       bag_size=self.default_boostrap_bag_size,
                                                                       replace=True,
                                                                       bags=1,
                                                                       weight='Duration').get('0')
        except ValueError:
            # In case of:Invalid weights: weights sum to zero
            self.originData = WightedBootstrapping.wighted_resampeling(X=self.originData,
                                                                       bag_size=self.default_boostrap_bag_size,
                                                                       replace=True,
                                                                       bags=1).get('0')
        finally:
            self.originData.reset_index(inplace=True)  # reset index is needed, some indexses are missed
            self.original_index = self.originData['index'].copy()
            self.originData = self.originData.drop(axis=1, labels=['Duration'])
            self.originData = self.originData.drop(axis=1, labels=['index'])  # After pandas 0.21.0 : columns=['index']

        # fixme: it is obligatory to make the data standardize, it should move to data pre-processing
        self.originData = pd.DataFrame(scale(self.originData,
                                             with_mean=True,
                                             with_std=True,
                                             copy=False),
                                       index=self.originData.index,
                                       columns=self.originData.columns)

        # Initiate the feature rank list that will updated by the Specific methods
        self.featureScoure['features'] = np.array(self.originData.columns.tolist())
        self._check_params(self.originData)

        self.adjacencyList.fit(self.originData,
                               adjacencyMatrix=False,
                               header=True,
                               index=True)

        gammaValue = gamma(self.originData)

        # TODO: Combine with filter and ensemble
        # TODO: Matrix product version of rbf_kernel ????? It will be faster in Euclidean one.

        '''
        Alternative of rbf_kernal:
        rbf_kernal_matrix=f = lambda x: np.exp(x**2*(-gammaValue))
        rbf_kernal_matrix(self.adjacencyList.graph_to_matrix())
        '''

        if parallel:
            # TODO: Use multiprocess + HDF5 here
            if recursion_limit is None:
                # TODO: It is needed to find an optimal values for. it.
                recursion_limit = self.originData.shape[0] ** 2
                warnings.warn(
                    "The recursion_limit is set to {} automatically.".format(recursion_limit, UserWarning))
            else:
                warnings.warn("v The recursion_limit is set to {} manually.".format(recursion_limit, UserWarning))
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
            temporal_knnGraph = manager.FastUpdate(self.adjacencyList.knnGraph, gama=gammaValue)

            # TODO: rewrite it as a module.

            chunks = list_splitter(list(self.adjacencyList.knnGraph.vert_dict.keys()), chunks=chunk_number)
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
            self.adjacencyList.knnGraph = temporal_knnGraph.get()

        else:
            progress_line_len = len(self.adjacencyList.knnGraph.vert_dict) * self.adjacencyList.k_nighbors
            actual_progress = 0
            print('\n')
            for v in self.adjacencyList.knnGraph:
                vid = v.get_id()
                for w in v.get_connections():
                    actual_progress += 1
                    self.progress_bar(actual_progress,
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
                    self.adjacencyList.knnGraph.update_edge(str(vid),
                                                            str(wid),
                                                            rbf_weight,
                                                            smallest=False)

        if adjacencyMatrix:
            self.adjacencyMatrix = self.adjacencyList.graph_to_matrix(binary=False)
        self.hasFitted = True

        return self

    # Sort Descending.
    def _sorted_features(self, order=-1):
        index = np.array(self.featureScoure['scores']).argsort(kind='quicksort')
        return {'sorted_features': self.featureScoure['features'][index][::order],
                'sorted_scores': self.featureScoure['scores'][index][::order],
                'ordered': order}

    def feature_score_table(self):
        sortedFeatureScore = self._sorted_features()
        if sortedFeatureScore.get('ordered') == 1:
            sort_arrow = '\u2191'
        elif sortedFeatureScore.get('ordered') == -1:
            sort_arrow = '\u2193'
        else:
            raise ValueError("The ordered direction has to be ascending or descending.")

        table_data = [
            ['Rank', 'Feature', str('Score ' + sort_arrow)]
        ]
        for rank, featureItem in enumerate(sortedFeatureScore['sorted_features']):
            table_data.append([rank,
                               featureItem,
                               sortedFeatureScore['sorted_scores'][rank]])
        table = DoubleTable(table_data,
                            title='{}'.format(str.upper(self.featureScoure['method'])))
        table.justify_columns[2] = 'center'
        return table

    def _check_params(self, X):
        pass


# TODO: It is needed to add progress bar to flow the process.
######################################################################
# Specific methods
######################################################################
# isdone: it is speeded up up.
class LaplacianScore(_BaseSpectralSelector):
    """ Ranking the features according to the smallest Laplacian scores.
            Objective function:
                Min
            Parameters:
                ----------
            Attributes:
                ----------
            Examples:
                --------
            See also:
                https://papers.nips.cc/paper/laplacian-score-for-feature-selection.pdf
    """

    def __init__(self, X=None, k=None):
        # It is just for methods uniforming and k is not useful for LS.
        self.k = k
        super(LaplacianScore, self).__init__(X, 'LaplacianScore')

    # Sort the list Ascending.
    def _sorted_features(self, order=1):
        return super(LaplacianScore, self)._sorted_features(order=order)

    def rank_features(self, X=None):
        if X is not None:
            self.fit(X)
        elif self.hasFitted:
            pass
        else:
            raise ValueError('The model has not fitted and the X is None')

        degreeMatrix = np.array(self.adjacencyMatrix.sum(axis=1))
        grapfLaplacian = np.subtract(np.diag(degreeMatrix), self.adjacencyMatrix)
        for feature in self.originData.columns:
            featureVector = np.array(self.originData[feature].tolist())
            featureRHat = np.array(featureVector
                                   - (np.dot(featureVector, degreeMatrix)
                                      / degreeMatrix.sum()))
            # todo: check the functionality of transpose
            featureLaplacianScore = np.dot(np.dot(featureRHat, grapfLaplacian),
                                           featureRHat.transpose())
            self.featureScoure['scores'] = np.append(self.featureScoure['scores'],
                                                     featureLaplacianScore)

    def _check_params(self, X):
        pass


class MultiClusterScore(_BaseSpectralSelector):
    """ Ranking the features according to the highest  Multi-Cluster Score.
    Objective function:  Max

    Parameters
    ----------
    Attributes
    ----------
    Examples
    --------
    See also
    --------
    http://www.cad.zju.edu.cn/home/dengcai/Publication/Conference/Multi-*cluster_analysis-feature-selection.pdf
    """

    def __init__(self, X=None, k=None, d=None):
        super(MultiClusterScore, self).__init__(X, 'Multi-Cluster')
        self.k_clusters = k
        self.d_selectedFeatures = d

    def __k_cluster_estimator(self):
        k = round(math.sqrt(self.originData.shape[0]))
        if math.fmod(k, 2) == 1:
            return k
        else:
            return k + 1

    def __update_scores(self, coefficientVector):
        coefficientVector = np.absolute(coefficientVector)
        if not len(self.featureScoure['scores']):
            self.featureScoure['scores'] = coefficientVector
        for index, featureScoreItem in enumerate(self.featureScoure['scores']):
            self.featureScoure['scores'][index] = max(featureScoreItem,
                                                      coefficientVector[index])
        # TODO: make the output colorized if the value is changed in any iteration

    def rank_features(self, X=None):

        if X is not None:
            self.fit(X)
        elif self.hasFitted:
            pass
        else:
            raise ValueError('The model has not fitted and the X is None')

        degreeMatrix = np.array(self.adjacencyMatrix.sum(axis=1))

        graphLaplacian = np.subtract(np.diag(degreeMatrix), self.adjacencyMatrix)

        # Calculate spectral Decomposition of graph Laplacian
        graphLaplacian = np.dot(np.linalg.inv(graphLaplacian), graphLaplacian)
        eigenValues, eigenVectors = np.linalg.eigh(graphLaplacian)

        # The eigen values have to be abs()
        eigenValues = np.abs(eigenValues)

        # TODO: it should move to _check_parameter
        with warnings.catch_warnings():
            warnings.simplefilter("default", UserWarning)

            # Initiate the k
            if self.k_clusters is None:
                # fixme: the k estimator have to be writen
                self.k_clusters = self.__k_cluster_estimator()
                warnings.warn('\n The k parameter has not indicated, It is set automatically to {}.'
                              .format(self.k_clusters), UserWarning, stacklevel=2)
            elif self.k_clusters > len(eigenValues):
                raise ValueError("k (multi-clusters) > {} flat embedding vectors.".format(len(eigenValues)))

            # Initiate the d
            if self.d_selectedFeatures is None:
                self.d_selectedFeatures = self.originData.shape[1]
                print("\n")
                warnings.warn('The d selected Features has not indicated, It is set automatically to {}.'
                              .format(self.d_selectedFeatures), UserWarning, stacklevel=2)
            elif self.d_selectedFeatures > self.originData.shape[1]:
                print("\n")
                raise ValueError(
                    'The d selected Features > {} flat embedding vectors.'.format(self.originData.shape[1]))

        eigens = dict(zip(eigenValues.real,
                          eigenVectors))
        eigens = dict(sorted(eigens.items(),
                             key=operator.itemgetter(0),
                             reverse=True))  # sort inplace

        # Solve the L1-regularized regressions K time
        reg = linear_model.Lars(n_nonzero_coefs=self.d_selectedFeatures)
        for eigenItem in range(self.k_clusters):
            _, vector = eigens.popitem()
            reg.fit(self.originData, vector)
            self.__update_scores(np.array(reg.coef_))

    def _check_params(self, X):
        pass


class SPEC(_BaseSpectralSelector):
    """ Ranking the features according to the highest  Laplacian scores.
    Parameters
    ----------
        K: it is the number of the clusters
    Attributes
    ----------
    Examples
    --------
    See also
    --------
    https://papers.nips.cc/paper/laplacian-score-for-feature-selection.pdf
    """

    # TODO: rewrite the feature sorting function that can accept the sorting parameter

    """
    Separability_scores and Normalized_Separability_scores are Ascending 
    and K_cluster_Separability_scores is Descending.
    """

    def _sorted_features(self, order=1, sort_by="Separability_scores"):
        if sort_by == "Separability_scores":
            index = np.array(self.featureScoure['Separability_scores']).argsort(kind='quicksort')
        elif sort_by == "Normalized_Separability_scores":
            index = np.array(self.featureScoure['Normalized_Separability_scores']).argsort(kind='quicksort')
        elif sort_by == "K_cluster_Separability_scores":
            index = np.array(self.featureScoure['K_cluster_Separability_scores']).argsort(kind='quicksort')
            order = -1
        else:
            raise ValueError('The score {} is not fined.(Separability_scores, '
                             'Normalized_Separability_scores, '
                             'K_cluster_Separability_scores)'.format(sort_by))

        return {'sorted_features': self.featureScoure['features'][index][::order],
                'sorted_Separability': self.featureScoure['Separability_scores'][index][::order],
                'sorted_Normalized_Separability': self.featureScoure['Normalized_Separability_scores'][index][::order],
                'sorted_K_cluster_Separability': self.featureScoure['K_cluster_Separability_scores'][index][::order],
                'sort_by': sort_by}

    def feature_score_table(self):
        sortedFeatureScore = self._sorted_features()
        table_data = [
            ['Rank',
             'Feature',
             'Separability_scores \u2191',
             'Normalized_Separability_scores',
             'K_cluster_Separability_scores']
        ]
        for rank, featureItem in enumerate(sortedFeatureScore['sorted_features']):
            table_data.append([rank,
                               featureItem,
                               sortedFeatureScore['sorted_Separability'][rank],
                               sortedFeatureScore['sorted_Normalized_Separability'][rank],
                               sortedFeatureScore['sorted_K_cluster_Separability'][rank]])
        table = DoubleTable(table_data,
                            title='{}'.format(str.upper(self.featureScoure['method'])))
        table.justify_columns[2] = 'center'

        return table

    def __init__(self, X=None, k=None):
        super(SPEC, self).__init__(X, 'SPEC')

        self.k = k
        self.featureScoure = {'method': 'SPEC',
                              'features': None,
                              'Separability_scores': np.array([]),
                              'Normalized_Separability_scores': np.array([]),
                              'K_cluster_Separability_scores': np.array([])}

    def rank_features(self, X=None):
        if X is not None:
            self.fit(X)
        elif self.hasFitted:
            pass
        else:
            raise ValueError('The model has not fitted and the X is None')

        degreeMatrix = np.array(self.adjacencyMatrix.sum(axis=1))

        graphLaplacian = np.subtract(np.diag(degreeMatrix), self.adjacencyMatrix)

        # normalized graph Laplacian (alias name is used for memory efficiency purposes)
        normalizedGraphLaplacian = graphLaplacian = np.power(degreeMatrix, -0.5) \
                                                    * graphLaplacian \
                                                    * np.power(degreeMatrix, -0.5)[:, np.newaxis]
        del (graphLaplacian)  # graphLaplacian is not valid any more.

        # Calculate spectral Decomposition of normalized graph Laplacian
        eigenValues, eigenVectors = np.linalg.eigh(np.dot(np.linalg.inv(normalizedGraphLaplacian),
                                                          normalizedGraphLaplacian))
        # TODO: the eigen values have to be abs()
        eigenValues = np.abs(eigenValues)
        microDensityIndicator = eigenVectors[np.argmax(eigenValues)]
        eigenValues, eigenVectors = sort_parja(eigenValues, eigenVectors, order=-1)

        for feature in self.originData.columns:
            featureVector = np.array(self.originData[feature].tolist())
            featureVectorTilda = np.sqrt(degreeMatrix) * featureVector
            featureVectorHat = featureVectorTilda / featureVectorTilda.sum()

            # TODO: it needs to check the calculation with Cosine similarity
            # Ranking Function 1: the value of the normalized cut (Shi & Malik, 1997) - ascending
            graphSeparability = np.dot(np.dot(featureVectorHat, normalizedGraphLaplacian),
                                       featureVectorHat.transpose())

            # Ranking Function 2: use spectral eigenValues to normalize the Ranking Function 1. - ascending
            normalizedGraphSeparability = graphSeparability / (1 - np.dot(featureVectorHat, microDensityIndicator))

            # Ranking Function 3: If the k ( number of clusters) is indicated, it should use the  reducing noise.
            if self.k is not None:
                kGraphSeparability = 0
                for eigenValueItem, eigenVectorItem in heapq.nlargest(self.k, zip(eigenValues, eigenVectors)):
                    kGraphSeparability += eigenValueItem * np.power(
                        cosine_similarity([featureVector], [eigenVectorItem]), 2)
            # Update the score list
            self.featureScoure['Separability_scores'] = np.append(
                self.featureScoure['Separability_scores'],
                graphSeparability)
            self.featureScoure['Normalized_Separability_scores'] = np.append(
                self.featureScoure['Normalized_Separability_scores'],
                normalizedGraphSeparability)
            self.featureScoure['K_cluster_Separability_scores'] = np.append(
                self.featureScoure['K_cluster_Separability_scores'],
                kGraphSeparability)

    def _check_params(self, X):
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

    df = __configoration('config/config_lulesh_27p.json', '../parser/source.csv')

    # feature selection
    test2 = SPEC(k=2)
    test2.fit(df)
    test2.rank_features(df)
    print(test2.featureScoure)


def __select_feature():
    start = time.time()
    # try:
    args = arguments_parser()
    df = __configoration(args['configPath'], args['csvPath'])

    if args['featureSelectionMethod'] == 'LS':
        featureSelectionModel = LaplacianScore(k=args['k_features'])
    elif args['featureSelectionMethod'] == 'MCFS':
        featureSelectionModel = MultiClusterScore(k=args['k_features'])
    elif args['featureSelectionMethod'] == 'SPEC':
        featureSelectionModel = SPEC(k=args['k_features'])
    else:
        pass

    featureSelectionModel.fit(df, bag_size=args['bag_size'])
    featureSelectionModel.rank_features()  # fixme: when the data is fitted it dose not need to refit here
    print("\n", featureSelectionModel.featureScoure)
    print(featureSelectionModel.feature_score_table().table)
    print("\033[32mThe feature selection process is successfully completed by {} method.".format(
        featureSelectionModel.featureScoure.get("method")))
    # except:
    # print("\033[31mThe feature selection proses is failed.")
    # finally:
    duration = time.time() - start
    print('\033[0mTotal duration is: %.3f' % duration)


if __name__ == '__main__':
    # __test_me()
    __select_feature()
