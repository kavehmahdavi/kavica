#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature selection by leverage of Feature Analysis that include PFA and IFA.

------------------------------------------------------------------------------------------------------------------------
 References:
    -  Y. Lu, I. Cohen, XS. Zhou, and Q. Tian, "Feature selection using principal feature analysis," in Proceedings of
       the 15th international conference on Multimedia. ACM, 2007, pp. 301-304.
------------------------------------------------------------------------------------------------------------------------
"""
# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause

# TODO: PFA
# TODO: IFA

import argparse
import sys
import warnings
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from terminaltables import DoubleTable
from kavica.imputation.base import data_structure_Compatibilization
from kavica.distance_measure import euclideanDistance
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn import decomposition
from kavica.factor_analysis.factor_rotation import ObliqueRotation
import json

__all__ = ['has_fitted',
           'sort_parja',
           '_centroid',
           '__configoration',
           '_BaseFeatureAnalysis',
           'PrincipalFeatureAnalysis',
           'IndependentFeatureAnalysis']


def has_fitted(estimator, attributes, msg=None, all_or_any=any):
    pass


def sort_parja(x, y, order=-1):
    # TODO: parameter check (numpy array)
    index = np.array(x).argsort(kind='quicksort')
    return (np.array(x)[index][::order], np.array(y)[index][::order])


# TODO: it is needed to rewrite it with method parameter
def _centroid(x, label):
    datafreamX = pd.DataFrame(x)
    datafreamX['label'] = label
    return datafreamX.groupby('label').mean()


# read the configuration file for preparing the features
def __configoration(config, data):
    # read the configuration file
    with open(config, 'r') as config:
        config_dict = json.load(config)

    # Read the data file
    df = pd.read_csv(data)

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
    # set/receive the arguments
    if len(sys.argv) == 1:
        # It is used for testing and developing time.
        arguments = ['config/config_FS_gromacs_64p_INS_CYC.json',
                     '../parser/source.csv',
                     '-k',
                     '2',
                     '-m',
                     'IFA'
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


######################################################################
# Base class
######################################################################
class _BaseFeatureAnalysis(object):
    """Initialize the feature analysis.

    Parameters
    """

    def __init__(self, X=None, method=None, k_features=None):
        self.hasFitted = False
        self.originData = X
        self.k_features = k_features
        self.featureScore = {'method': method,
                             'scores': pd.DataFrame(columns=['features', 'subset', 'internal_score'])}

    def fit(self, X):
        """ Check the input data and fit to the model.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        Returns
        -------
        self : object
        """
        # fixme: it is a duplicated action
        self.originData = data_structure_Compatibilization(data=X,
                                                           header=True,
                                                           index=True)
        # fixme: it is obligatory to make the data standardize, it should move t o data pre-processing
        self.originData = pd.DataFrame(scale(self.originData,
                                             with_mean=True,
                                             with_std=True,
                                             copy=False),
                                       index=self.originData.index,
                                       columns=self.originData.columns)

        # Initiate the feature rank list that will updated during analysis
        self.featureScore['scores']['features'] = np.array(self.originData.columns.tolist())
        self._check_params(X)
        self.hasFitted = True
        return self

    def _sorted_features(self):
        return self.featureScore['scores'].sort_values(['subset', 'internal_score'],
                                                       ascending=[True, True])

    # TODO: rewrite
    def _feature_score_table(self):
        sortedFeatureScore = np.array(self._sorted_features())
        table_data = [
            ['Feature', 'Subset', 'Internal_rank']
        ]
        for featurItem in sortedFeatureScore:
            table_data.append(featurItem.tolist())

        table = DoubleTable(table_data,
                            title='{}'.format(str.upper(self.featureScore['method'])))
        table.justify_columns[2] = 'center'
        return table

    def _check_params(self, X):
        pass


######################################################################
# Feature analysis methods
######################################################################
class PrincipalFeatureAnalysis(_BaseFeatureAnalysis):
    """ Split the features to a k subset and applies the feature ranking inside any subset.
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
        super(PrincipalFeatureAnalysis, self).__init__(X, 'PFA', k)

    def __centroid_predefining(self, x, dendrogram=False):
        "predefining  the centroid for stabilizing the kmean."
        if dendrogram:
            # create dendrogram
            sch.dendrogram(sch.linkage(x, method='ward'))

        hc = AgglomerativeClustering(n_clusters=self.k_features, affinity='euclidean', linkage='ward')
        labels = hc.fit_predict(x)
        return _centroid(x, labels)

    # TODO: Wighted ranking the feature should be implemented
    def _rank_features(self, X=None, dendrogram=False):
        if X is not None:
            self.fit(X)
        elif self.hasFitted:
            pass
        else:
            raise ValueError('The model has not fitted and the X is None')

        eigenValues, eigenVectors = np.linalg.eigh(self.originData.cov())
        predefinedCentroids = self.__centroid_predefining(eigenVectors)

        # Do the clustering on rows that are the features.
        featureClustering = KMeans(n_clusters=self.k_features,
                                   max_iter=300,
                                   algorithm='auto',
                                   precompute_distances='auto',
                                   init=predefinedCentroids).fit(eigenVectors)
        featureSubstes = featureClustering.predict(eigenVectors)
        featureSubstesCentroid = featureClustering.cluster_centers_
        self.featureScore['scores']['subset'] = featureSubstes
        for index, label in enumerate(featureSubstes):
            self.featureScore['scores']['internal_score'][index] = euclideanDistance(eigenVectors[index, :],
                                                                                     featureSubstesCentroid[label, :])

    def _check_params(self, X):
        pass


class IndependentFeatureAnalysis(_BaseFeatureAnalysis):
    """ Split the features to a k subset and applies the feature ranking inside any subset.
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
        super(IndependentFeatureAnalysis, self).__init__(X, 'IFA', k)

    def __centroid_predefining(self, x, dendrogram=False):
        "predefining  the centroid for stabilizing the kmean."
        if dendrogram:
            # create dendrogram
            sch.dendrogram(sch.linkage(x, method='ward'))

        hc = AgglomerativeClustering(n_clusters=self.k_features, affinity='euclidean', linkage='ward')
        labels = hc.fit_predict(x)
        return _centroid(x, labels)

    # TODO: Insert the deprogram conditional
    # TODO: import the promax in to the method
    def _rank_features(self, X=None, dendrogram=False, rotation='promax'):
        if X is not None:
            self.fit(X)
        elif self.hasFitted:
            pass
        else:
            raise ValueError('The model has not fitted and the X is None')
        try:
            # TODO: the columns with all zero value have to be eliminated.
            # TODO: it is the problem of whiten=True.
            icaModel = decomposition.FastICA(whiten=True, random_state=1).fit(self.originData)
        except:
            warnings.warn("ICA is forced to run without whitening.", UserWarning)
            icaModel = decomposition.FastICA(whiten=False, random_state=1).fit(self.originData)
            icaModel = decomposition.FastICA(whiten=False, random_state=1).fit(self.originData)
        finally:
            # The transpose of ICA components are used because the output of ICA is(n_component,n_features)
            independentComponents = icaModel.components_

            # the rotation that amplified the load of important component in any feature
            # The row is the components and the columns are the features
            if rotation == 'promax':
                promaxRotation = ObliqueRotation('promax')
                promaxRotation.fit(independentComponents)
                rotatedIndependentComponents = promaxRotation.oblique_rotate()
                independentComponents = rotatedIndependentComponents

            # The rotated ICA components (n_component,n_features) transpose to the (n_features, n_component)
            independentComponents = independentComponents.T
            predefinedCentroids = self.__centroid_predefining(independentComponents)

            # Do the clustering on rows that are the features.
            featureClustering = KMeans(n_clusters=self.k_features,
                                       max_iter=300,
                                       algorithm='auto',
                                       precompute_distances='auto',
                                       init=predefinedCentroids).fit(independentComponents)
            featureSubstes = featureClustering.predict(independentComponents)
            featureSubstesCentroid = featureClustering.cluster_centers_
            self.featureScore['scores']['subset'] = featureSubstes

            for index, label in enumerate(featureSubstes):
                self.featureScore['scores']['internal_score'][index] = euclideanDistance(
                    independentComponents[index, :],
                    featureSubstesCentroid[label, :])

    def _check_params(self, X):
        pass


def __test_me():
    # sample dataset:
    '''
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

    headers = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    index = [1, 2, 3, 4]
    df = pd.DataFrame(data0, columns=headers, index=index, dtype=np.float)
    '''

    df = __configoration('config/config_lulesh_27p.json', '../parser/source.csv')
    testICA = IndependentFeatureAnalysis(k=2)
    testICA._rank_features(df, dendrogram=True)
    print(testICA._feature_score_table().table)

    testPCA = PrincipalFeatureAnalysis(k=2)
    testPCA._rank_features(df, dendrogram=True)
    print(testPCA._feature_score_table().table)


# Todo: add dendogram ****
def __select_feature():
    start = time.time()
    try:
        args = arguments_parser()
        df = __configoration(args['configPath'], args['csvPath'])
        if args['featureSelectionMethod'] == 'IFA':
            featureSelectionModel = IndependentFeatureAnalysis(k=args['k_features'])
        elif args['featureSelectionMethod'] == 'PFA':
            featureSelectionModel = PrincipalFeatureAnalysis(k=args['k_features'])
        else:
            pass

        featureSelectionModel._rank_features(df, dendrogram=True)
        print(featureSelectionModel._feature_score_table().table)
        print("\033[32mThe feature selection process is successfully completed by {} method.".format(
            featureSelectionModel.featureScore.get("method")))
    except AssertionError as error:
        print(error)
        print("\033[31mThe feature selection proses is failed.")
    finally:
        duration = time.time() - start
        print('\033[0mTotal duration is: %.3f' % duration)


if __name__ == '__main__':
    # __test_me()
    __select_feature()
