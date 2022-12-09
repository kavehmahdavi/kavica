#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature selection by leverage of Feature Analysis that include PFA and IFA.

$ python3 feature_analysis.py config/config_FS_gromacs_64p.json ../../data/FS_test_data/imputed.csv -k 3 -m ifa

 References:
    -  Mahdavi, K., Labarta, J., & Gimenez, J. (2019, November). Unsupervised feature selection for noisy data.
       In International Conference on Advanced Data Mining and Applications (pp. 79-94). Springer, Cham.

    -  Y. Lu, I. Cohen, XS. Zhou, and Q. Tian, "Feature selection using principal feature analysis," in Proceedings of
       the 15th international conference on Multimedia. ACM, 2007, pp. 301-304.

# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# Last update: 04/10/2022
"""

import argparse
import json
import sys
import time
import warnings
from abc import ABC
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from kavica.distance_measure import euclidean_distance
from kavica.factor_analysis.factor_rotation import ObliqueRotation
from kavica.imputation.base import compatible_data_structure
from kavica.feature_selection.base import FeatureSelection
from sklearn import decomposition
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from terminaltables import DoubleTable
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA

__all__ = ['pair_sort',
           '_centroid',
           '__config',
           'PrincipalFeatureAnalysis',
           'IndependentFeatureAnalysis',
           '_BaseFeatureAnalysis']

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 365)
pd.set_option('display.width', 700)
np.set_printoptions(threshold=np.inf, linewidth=200)


def pair_sort(x, y, order=1):
    """ Sort an array base on the order of the other array

    Args:
        x (np.ndarray): include the first array
        y (np.ndarray): include the second array
        order (int): It could be -1 to do the detrimental and 1 incremental.

    Returns:
        Two np.ndarray where the second one sorted based on the first one
    """
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        index = np.array(x).argsort(kind='quicksort')
        return np.array(x)[index][::order], np.array(y)[index][::order]
    else:
        raise TypeError('ERROR: the parameters of pair_sort function most be np.ndarray.')


def _centroid(x, label):
    """ Computes the centroid of all groups in the dataset

    Args:
        x (np.ndarray): includes the dataset
        label (list): includes the labels

    Returns:
        A pandas includes all groups centroids
    """
    _x_df = pd.DataFrame(x)
    _x_df['label'] = label
    return _x_df.groupby('label').mean()


def __config(config_path, data_path):
    """ Read the configuration file to prepare the features

    Args:
        config_path (str): indicates the configuration path
        data_path (str): indicates the dataset path

    Returns:
        A pandas includes the cleaned dataset
    """
    with open(config_path, 'r') as config_path:
        config_dict = json.load(config_path)

    df = pd.read_csv(data_path)

    # config the data set based on configuration information
    df = df[list(config_dict['hardware_counters'].values())]  # sub set of features
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
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
    return df


def arguments_parser():
    """ Parses and sets the arguments

    Returns:
        A dict includes the parameters
    """
    # DevelopTime: It is used for testing and developing time.
    if len(sys.argv) == 1:
        arguments = ['config/config_lulesh_27p.json',
                     '../parser/source.csv',
                     '-k',
                     '2',
                     '-m',
                     'IFA'
                     ]
        sys.argv.extend(arguments)

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
                        default='IFA',
                        choices=['IFA', 'PFA'],
                        action='store',
                        type=str.upper,
                        help="The feature selection method that is either IFA or PFA.")

    args = parser.parse_args()

    if args.k < 2:
        raise ValueError("Selected features have to be (=> 2). It is set {}".format(args.k))

    return ({"configPath": args.config,
             "csvPath": args.csvfile,
             "k_features": args.k,
             "featureSelectionMethod": args.m})


def plot_feature_important(score, reverse=True, ascending=False, output=None):
    """Bar plot of score / inverse feature important

    Args:
        score (panda.series): Includes the feature rank.
        reverse (boolean): If True, we use the 1/score.
        ascending (boolean): If True, we sort the features ascending
        output (str): Indicates the output file name
    Returns:
        A str indicates the output file
    """
    subset_number = len(set(score['subset']))
    _cmap = get_cmap('tab10')
    colors = _cmap.colors
    x_lab = "Score"
    plt.figure(1, figsize=(8, 10))
    for category in score.groupby(['subset']):
        subset_index = category[0]
        plot_index = int('{}1{}'.format(str(subset_number), str(subset_index + 1)))
        plt.subplot(plot_index)
        category = category[1].drop(['subset'], axis=1).set_index('features')

        if category.size > 1:
            category = category.T.squeeze().sort_values(ascending=ascending)
            patterns = [""] * (category.size - 1)
            patterns.extend(['+'])
        elif category.size == 1:
            category = pd.Series(category.T.iat[0, 0], index=category.T.columns)
            patterns = ["+"]

        if reverse:
            category = 1 / category
            x_lab = "Reverse Score"

        category.plot(kind='barh', color=colors[subset_index])
        bars = plt.subplot(plot_index).patches
        for bar, hatch in zip(bars, patterns):
            bar.set_hatch(hatch)
        plt.ylabel("Feature")
        plt.xlabel(x_lab)

    st = plt.suptitle("Feature Importance Plot", fontsize="x-large")
    st.set_y(0.95)  # shift subplots down:
    plt.subplots_adjust(top=0.85, left=0.3, hspace=0.2)
    plt.savefig('feature_importance_plot_{}.jpg'.format(output))
    plt.show()
    return 'feature_importance_plot_{}.jpg'.format(output)


class _BaseFeatureAnalysis(FeatureSelection, ABC):
    """ Base class of feature analysis.

    """

    def __init__(self, X=None, method=None, k_features=None):
        self.has_fitted = False
        self.origin_data = X
        self.k_features = k_features
        self.selected_feature_list = []
        self.feature_score = {'method': method,
                              'scores': pd.DataFrame(columns=['features', 'subset', 'internal_score'])}

    def fit(self, X):
        """ Check the input data and fit to the model.

        Args:
            X (array-like): shape = [n_samples, n_features] The training input samples.

        Returns
            self
        """
        # fixme: it is a duplicated action
        self.origin_data = compatible_data_structure(data=X, header=True, index=True)

        # fixme: it is obligatory to make the data standardize, it should move t o data pre-processing
        self.origin_data = pd.DataFrame(scale(self.origin_data, with_mean=True, with_std=True, copy=False),
                                        index=self.origin_data.index,
                                        columns=self.origin_data.columns)

        # Initiate the feature rank list that will update during analysis
        self.feature_score['scores']['features'] = np.array(self.origin_data.columns.tolist())
        self._check_params(X)
        self.has_fitted = True
        return self

    def _sorted_features(self):
        """ Sorts the feature list based on their scors

        Returns:
            A pandas includes the sorted feature list
        """
        return self.feature_score['scores'].sort_values(['subset', 'internal_score'], ascending=[True, True])

    # TODO: revise and maybe rewrite
    def feature_score_table(self):
        """ Creates a feature score table

        Returns:
            A DoubleTable object includes the feature scours
        """
        sorted_feature_score = np.array(self._sorted_features())
        table_data = [['Feature', 'Subset', 'Internal_rank']]
        for _feature_item in sorted_feature_score:
            table_data.append(_feature_item.tolist())
        table = DoubleTable(table_data, title='{}'.format(str.upper(self.feature_score['method'])))
        table.justify_columns[2] = 'center'
        return table

    # Todo: Future work
    def _check_params(self, X):
        pass


class PrincipalFeatureAnalysis(_BaseFeatureAnalysis, ABC):
    """ Split the features to a k subset and applies the feature ranking inside any subset.

    """

    def __init__(self, X=None, k=None):
        super(PrincipalFeatureAnalysis, self).__init__(X, 'PFA', k)

    def __init_centroid(self, x, dendrogram=False):
        """ Defines the centroid for stabilizing the kmeans.

        Args:
            x (pandas): indicates the data
            dendrogram (boolean): If True iw could draw a dendrogram.

        Returns:
            A pandas includes all groups centroids
        """
        if dendrogram:
            sch.dendrogram(sch.linkage(x, method='ward'))  # create dendrogram
        hc = AgglomerativeClustering(n_clusters=self.k_features, affinity='euclidean', linkage='ward')
        labels = hc.fit_predict(x)
        return _centroid(x, labels)

    # TODO: Wighted ranking the feature should be implemented
    def rank_features(self, X=None, dendrogram=False):
        """ Ranks the features

        Args:
            X (pandas): Indicates the dataset
            dendrogram (boolean): If True it draws the dendrogram when initialize the centroids
        Returns:
            A list of the features
        """
        if X is not None:
            self.fit(X)
        elif self.has_fitted:
            pass
        else:
            raise ValueError('ERROR: The model has not fitted and the X is None')

        eigen_values, eigen_vectors = np.linalg.eigh(self.origin_data.cov())

        predefined_centroids = self.__init_centroid(eigen_vectors, dendrogram)

        # Do the clustering on rows that are the features.
        feature_clusters = KMeans(n_clusters=self.k_features,
                                  max_iter=100,
                                  algorithm='auto',
                                  precompute_distances='auto',
                                  init=predefined_centroids).fit(eigen_vectors)

        feature_subsets = feature_clusters.predict(eigen_vectors)
        feature_subsets_centroid = feature_clusters.cluster_centers_
        self.feature_score['scores']['subset'] = feature_subsets
        for index, label in enumerate(feature_subsets):
            self.feature_score['scores']['internal_score'][index] = euclidean_distance(
                eigen_vectors[index, :],
                feature_subsets_centroid[label, :])
        for name, group in self.feature_score['scores'].groupby('subset'):
            print('From subset:{} the feature --{}-- is selected.'.format(name,
                                                                          self.feature_score['scores'].iloc[
                                                                              group['internal_score'].astype(
                                                                                  float).idxmin()]['features']))

            self.selected_feature_list.append(
                self.feature_score['scores'].iloc[group['internal_score'].astype(float).idxmin()]['features'])
        return self.selected_feature_list

    def _check_params(self, X):
        pass


class IndependentFeatureAnalysis(_BaseFeatureAnalysis, ABC):
    """ Split the features to a k subset and applies the feature ranking inside any subset.

    """

    def __init__(self, X=None, k=None):
        super(IndependentFeatureAnalysis, self).__init__(X, 'IFA', k)

    def __init_centroid(self, x, dendrogram=False):
        """ Defines the centroid for stabilizing the kmeans.

        Args:
            x (pandas): indicates the data
            dendrogram (boolean): If True iw could draw a dendrogram.

        Returns:
            A pandas includes all groups centroids
        """
        if dendrogram:
            sch.dendrogram(sch.linkage(x, method='ward'))  # create dendrogram
        hc = AgglomerativeClustering(n_clusters=self.k_features, affinity='euclidean', linkage='ward')
        labels = hc.fit_predict(x)
        return _centroid(x, labels)

    # TODO: Insert the deprogram conditional
    # TODO: import the Promax in to the method
    def rank_features(self, X=None, rotation='promax'):
        if X is not None:
            self.fit(X)
        elif self.has_fitted:
            pass
        else:
            raise ValueError('The model has not fitted and the X is None')
        try:
            # TODO: the columns with all zero value have to be eliminated.
            # TODO: it is the problem of whiten=True.
            icaModel = decomposition.FastICA(whiten=True, random_state=1).fit(self.origin_data)
        except:
            warnings.warn("ICA is forced to run without whitening.", UserWarning)
            icaModel = decomposition.FastICA(whiten=False, random_state=1).fit(self.origin_data)
            icaModel = decomposition.FastICA(whiten=False, random_state=1).fit(self.origin_data)
        finally:
            # The transpose of ICA components are used because the output of ICA is(n_component,n_features)
            independent_components = icaModel.components_

            # the rotation that amplified the load of important component in any feature
            # The rows are the components and the columns are the features
            if rotation == 'promax':
                promax_rotation = ObliqueRotation('promax')
                promax_rotation.fit(independent_components)
                rotated_independent_components = promax_rotation.oblique_rotate()
                independent_components = rotated_independent_components

            # The rotated ICA components (n_component,n_features) transpose to the (n_features, n_component)
            independent_components = independent_components.T
            predefined_centroids = self.__init_centroid(independent_components)

            # Do the clustering on rows that are the features.
            feature_clustering = KMeans(n_clusters=self.k_features,
                                        max_iter=500,
                                        algorithm='auto',
                                        precompute_distances='auto',
                                        init=predefined_centroids).fit(independent_components)
            feature_subsets = feature_clustering.predict(independent_components)
            feature_subsets_centroid = feature_clustering.cluster_centers_
            self.feature_score['scores']['subset'] = feature_subsets

            # TODO: it is too slow. It needs some bust
            for index, label in enumerate(feature_subsets):
                self.feature_score['scores']['internal_score'][index] = euclidean_distance(
                    independent_components[index, :],
                    feature_subsets_centroid[label, :])
            for name, group in self.feature_score['scores'].groupby('subset'):
                print('Form subset:{} the feature --{}-- is selected.'.format(name,
                                                                              self.feature_score['scores'].iloc[
                                                                                  group['internal_score'].astype(
                                                                                      float).idxmin()]['features']))
                self.selected_feature_list.append(
                    self.feature_score['scores'].iloc[group['internal_score'].astype(float).idxmin()]['features'])

    def _check_params(self, X):
        pass


def __test_me():
    # sample dataset:

    data0 = np.array([(1, 1, 1, 1, 1, 1, 1),
                      (2, 2, 2, 2, 2, 2, 2),
                      (3, 4, 45, 23, 24, 19, 16),
                      (4, 2, 44, 23, 22, 13, 11),
                      (5, 2, 4, 3, 2, 1, 1),
                      (6, 1, 1, 1, 1, 1, 1),
                      (7, 2, 2, 2, 2, 2, 2),
                      (8, 2, 45, 23, 24, 13, 16),
                      (9, 12, 0, 9, 5, 20, 89),
                      (10, 6, 7, 8, 3, 8, 2),
                      (11, 8, 7, 43, 12, 56, 1),
                      (12, 13, 4, 5, 6, 33, 4),
                      (13, 94, 5, 16, 8, 52, 45)])
    data = np.array([(1, 1, 1, 1, 1, 1, 1),
                     (2, 2, 2, 2, 1, 2, 2),
                     (2, 2, 45, 23, 24, 13, 16),
                     (3, 12, 0, 9, 5, 20, 89)])
    data1 = np.array([("ind", "F1", "F2", "F3", "F4", "F5", "F6"),
                      (1, 1, 1, 1, 1, 1, 1),
                      (2, 2, 2, 4, 2, 7, 2),
                      (3, 4, 45, 23, 24, 19, 16),
                      (4, 2, 44, 23, 22, 13, 11),
                      (5, 2, 4, 3, 2, 1, 1),
                      (6, 1, 1, 1, 1, 78, 1),
                      (7, 2, 2, 8, 2, 2, 2),
                      (8, 2, 45, 23, 24, 13, 16),
                      (9, 12, 0, 9, 5, 20, 89),
                      (10, 6, 7, 8, 3, 8, 2),
                      (11, 8, 7, 43, 12, 56, 1),
                      (12, 13, 4, 5, 6, 33, 4),
                      (13, 94, 5, 16, 8, 52, 45),
                      (14, 2, 3, 4, 3, 5, 300)])

    data2 = np.array([("ind", "F1", "F2", "F3", "F4", "F5", "F6"),
                      (1, 1, 1, 1, 1, 1, 1),
                      (2, 2, 2, 2, 2, 2, 2),
                      (3, 2, 4, 3, 2, 1, 1),
                      (4, 1, 1, 1, 1, 1, 1),
                      (5, 2, 2, 2, 2, 2, 2)])

    data2 = np.array([(1, 1, 1, 1, 1, 1, 1),
                      (1, 2, 2, 2, 2, 2, 2),
                      (3, 2, 4, 3, 2, 1, 1),
                      (2, 1, 1, 1, 1, 1, 1),
                      (5, 2, 2, 2, 2, 2, 2),
                      (5, 2, 4, 3, 2, 1, 1),
                      (6, 1, 1, 1, 1, 78, 1),
                      (7, 2, 2, 8, 2, 2, 2),
                      (1, 1, 1, 1, 1, 1, 1),
                      (2, 2, 2, 2, 2, 2, 2),
                      (3, 2, 4, 3, 2, 1, 1),
                      (4, 1, 1, 1, 1, 1, 1), ])

    headers = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df = pd.DataFrame(data2, columns=headers, dtype=np.float)

    # df = __config('config/config_lulesh_27p.json', '../parser/source.csv')

    testICA = IndependentFeatureAnalysis(k=2)
    testICA.rank_features(df)
    print(testICA.feature_score_table().table)

    testPCA = PrincipalFeatureAnalysis(k=2)
    testPCA.rank_features(df, dendrogram=True)
    print(testPCA.feature_score_table().table)


# Todo: add dendrogram
def __select_feature():
    start = time.time()
    try:
        args = arguments_parser()
        df = __config(args['configPath'], args['csvPath'])
        if args['featureSelectionMethod'] == 'IFA':
            feature_selection_model = IndependentFeatureAnalysis(k=args['k_features'])
            feature_selection_model.rank_features(df, rotation='promax')
        elif args['featureSelectionMethod'] == 'PFA':
            feature_selection_model = PrincipalFeatureAnalysis(k=args['k_features'])
            feature_selection_model.rank_features(df, dendrogram=False)
        else:
            pass

        plot_feature_important(score=feature_selection_model.feature_score.get('scores'),
                               output=args['featureSelectionMethod'])
        print(feature_selection_model.feature_score_table().table)
        print("\033[32mThe feature selection process is successfully completed by {} method.".format(
            feature_selection_model.feature_score.get("method")))
    except AssertionError as error:
        print(error)
        print("\033[31mThe feature selection proses is failed.")
    finally:
        duration = time.time() - start
        print('\033[0mTotal duration is: %.3f' % duration)


if __name__ == '__main__':
    #__test_me()
    __select_feature()
