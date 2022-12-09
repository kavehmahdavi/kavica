#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Organization Component Analysis

It is a Cluster Self Organization Map Inference Systems

This module is an Artificial Neural base Cluster Inference System. It infers the cluster structure by the leverages
of the Self Organization Map.

The main objectives:
    - Read the data from a csv file
    - Prepare data: Normalizing, scaling, selecting the columns, input the missing, adding dummy features
    - Apply the SOMIS over the data
    - Plot the results (the main inertia's)
    - Report the result (the driver feature and the driving direction)

This module includes both:
    - The Fuzzy Self Organization Map Inference Systems
    - The Self Organization Map Inference Systems

Examples:
    $ python3 cluster_organization.py clustering_data.csv --cw feature_x feature_y -c cluster_number
                                                        --s map_size_x map_size_y -i epoch_number
        cluster_data.csv: Includes the clustering data
        --cw (str): Indicates the two features that have been used in order to cluster data (feature_x, feature_y)
        -c (str): Indicates the cluster that we would like to apply the organization analysis on.
        --s (int): Indicates the Self Organization Map grid sizes (map_size_x, map_size_y)
        -i (int): Indicates the ANN's training iterations.

    $ python3 OCA.py config/OCA_configs/iris.json ../../data/oca_test_data/iris.csv --cw 'sepal_width' 'sepal_length' -c 1 --s 6 6 -i 700

Notes:
    It is applicable over both the original or the clustered data.


See:
    - Mahdavi, K., Mancho, J. L., & Lucas, J. G. (2021, July). Organization Component Analysis: The method for
      extracting insights from the shape of cluster. In 2021 International Joint Conference on Neural Networks
      (IJCNN) (pp. 1-10). IEEE.

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
last update: 31/10/2022
"""

import argparse
import gc
import json
import math
import os
import re
import shutil
import sys
import time
import warnings
import webbrowser
import matplotlib.colors as mcolors
from scipy.spatial.distance import euclidean
import matplotlib.font_manager
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.common import flatten
from pandas_profiling import ProfileReport
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from kavica.utils._bcolors import BColors
from kavica.utils._util import record_log
from kavica.cluster_inference_system.somis import SOMIS
from kavica._hpc.feature_engineering import burst_overlying_factor
from kavica.utils._plots import MARKER, PARAVER_STATES_COLOR, get_color_palette
import copy

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 365)
pd.set_option('display.width', 700)
np.set_printoptions(threshold=sys.maxsize)

matplotlib.use('Agg')  # 'module://backend_interagg' , 'WebAgg'
# matplotlib.pyplot.switch_backend('Agg')

__all__ = ['drop_outliers',
           'load_data',
           'organization_component_analysis']

# Global Configuration and result dict.
CONFIG = {}

"""
Distance measurements:
- angle:
- cosine_Similarity:
- distance:
- distance_angle

Ratio:
- x_PAPI_TOT_CYC
- d_PAPI_TOT_INS
"""


def drop_outliers(data, fraction=0.05, seed=42, algorithm="isolation_forest", n_samples='auto', plot=True):
    """ Eliminates the outliers.

    It computes the outliers by "Isolation_Forest" or "Robust_covariance". We need to let it know the fraction
    of the data that we suppose to be outliers.

    Args:
        plot (boolean): If True, it will plot out the contour plot.
        fraction (float): The amount of contamination of the data set, i.e. the proportion of outliers in the data set.
                          Used when fitting to define the threshold on the scores of the samples.
                          The contamination should be in the range [0, 0.5].
        seed (int): Controls the pseudo-randomness of the selection of the feature and split values for each branching
                    step and each tree in the forest. Pass an int for reproducible results across multiple function call
        n_samples (int): Just in case of Isolation_Forest, The number of samples to draw from X to train each base
                         estimator.
                            If int, then draw max_samples samples.
                            If float, then draw max_samples * X.shape[0] samples.
                            If “auto”, then max_samples=min(256, n_samples).
                            If max_samples is larger than the number of samples provided, all samples will be used
                            for all trees (no sampling).

        algorithm (str): It can be either "Isolation_Forest" or "Robust_covariance".
        data (pandas): Includes the input samples

    Returns:
        A pandas includes the inlines data points
    """
    _seed = np.random.RandomState(seed)
    before_data_shape = data.shape

    if algorithm.lower() == "robust_covariance":
        clf = EllipticEnvelope(contamination=fraction, random_state=_seed)
    elif algorithm.lower() == "isolation_forest":
        clf = IsolationForest(max_samples=n_samples, contamination=fraction, random_state=_seed)
    else:
        raise ValueError("The outliers detector algorithm is not pre_defined")

    x = data.loc[:, CONFIG.get('cluster_with')]

    clf.fit(x)
    _scores = clf.decision_function(x)
    threshold = stats.scoreatpercentile(_scores, 100 * fraction)
    y = clf.predict(x)
    y = np.where(y == -1, False, True)
    after_data_shape = data[y].shape

    update_config({'outliers_number': before_data_shape[0] - after_data_shape[0]})

    record_log(BColors.OKGREEN, "Outliers", "Input {} -> Output {}", before_data_shape, after_data_shape)
    record_log(BColors.OKGREEN, "Outliers", "The {}% = {} of data points are eliminated by {} successfully.",
               fraction * 100,
               CONFIG.get('outliers_number'),
               algorithm)

    if plot:
        plt.figure(figsize=(16, 14))

        # plot the levels lines
        xx, yy = np.meshgrid(np.linspace(x.iloc[:, 0].min() * 0.99, x.iloc[:, 0].max() * 1.01, 500),
                             np.linspace(x.iloc[:, 1].min() * 0.99, x.iloc[:, 1].max() * 1.01, 500))
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, levels=np.linspace(z.min(), threshold, 7), cmap=plt.cm.Blues_r)
        plt.contourf(xx, yy, z, levels=[threshold, z.max()], colors='orange')
        a = plt.contour(xx, yy, z, levels=[threshold], linewidths=2, colors='k')

        # plot the points
        y_color = np.where(y, 'green', 'red')
        plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y_color, marker='x')

        # Set the plot style
        plt.legend([a.collections[0], mpatches.Patch(color='red'), mpatches.Patch(color='green')],
                   ['learned decision function', 'Outliers', 'Inliers'],
                   prop=matplotlib.font_manager.FontProperties(size=11),
                   loc='upper left')

        plt.axis('tight')
        plt.title("%s. Outliers Contour plot" % algorithm)
        plt.xlim((x.iloc[:, 0].min() * 0.99, x.iloc[:, 0].max() * 1.01))
        plt.ylim((x.iloc[:, 1].min() * 0.99, x.iloc[:, 1].max() * 1.01))
        plt.subplots_adjust(0.04, 0.1, 0.96, 0.92, 0.1, 0.26)

        if CONFIG.get('make_jupiter_report'):
            plt.savefig('OCA_Report/pic/outliers.png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    return data[y]


def arguments_parser():
    """ Parse the arguments

    Return:
        A dict includes {"conf_path": str,
                         "csv_path": str,
                         "cluster_with": tuple,
                         "cluster_number": str,
                         "SOM_Size": tuple,
                         "epoch_number": int}
    """
    # Develop test_configs
    if len(sys.argv) == 1:
        # For testing and developing time.
        arguments = ['config/OCA_configs/config_stream_1.json',
                     '../../data/som_test_data/stream_transfer/stream_0.0001m_32n.csv',
                     '--cw', 'd_IPC', 'd_PAPI_TOT_INS',
                     '-c', '6',  # '6', '7', ...
                     '--s', '10', '10',
                     '-i', '6000']
        sys.argv.extend(arguments)

    parser = argparse.ArgumentParser(description='Cluster Self Organization Map Inference Systems')

    parser.add_argument('conf',
                        help='A .json configuration file.')
    parser.add_argument('csv',
                        help='A .csv dataset file, includes the independent features and dependent labels.')
    parser.add_argument('--cw',
                        dest='cw',
                        default=['n_IPC', 'd_PAPI_TOT_INS'],
                        nargs='+',
                        action='store',
                        type=str,
                        help="The features that have been used to cluster data.")
    parser.add_argument('-c',
                        dest='c',
                        default='1',
                        nargs='+',
                        action='store',
                        type=int,
                        help="The cluster that we would like to apply the organization analysis on.")
    parser.add_argument('--s',
                        dest='s',
                        nargs='+',
                        default=['10', '10'],
                        action='store',
                        type=int,
                        help="The Self Organization Map grid sizes (map_size_x, map_size_y)")
    parser.add_argument('-i',
                        dest='i',
                        default=200,
                        action='store',
                        type=int,
                        help="The SOM's training iterations.")

    args = parser.parse_args()

    args_dict = {"conf_path": args.conf,
                 "csv_path": args.csv,
                 "cluster_with": args.cw,
                 "cluster_number": args.c,
                 "SOM_Size": tuple(args.s),
                 "epoch_number": args.i}

    update_config(args_dict)
    record_log(BColors.OKGREEN, "Arguments", "The {} arguments are obtained successfully.", len(args_dict))

    return args_dict


def configure(config_path):
    """ Reads the configuration file

    Args:
        config_path (str): indicates the config file path.

    Returns:
        A dict includes the configuration data
    """
    with open(config_path, 'r') as conf:
        config_dict = json.load(conf)

    update_config(config_dict)
    record_log(BColors.OKGREEN, "Config", "The {} config items are read from file successfully.", len(config_dict))

    return config_dict


def update_config(item):
    """ Updates the config dict

    Args:
        item (dict): Includes item/s that we would like to be added to the configuration dict.

    Returns:
        True if it is updated correctly
    """
    global CONFIG
    CONFIG.update(item)
    record_log(BColors.OKGREEN, "Config", "The CONFIG dict is updated with {} new item/s successfully.", len(item))
    return True


def load_data(data, lab='ClusterID'):
    """ Loads data

    To impute a file, we need to indicate the main features, pass_through and the complementary in config file.

    Args:
        data (str): indicates the path of the data file (.csv)
        lab(str): Indicates the label feature

    Return:
        A pandas includes the data.

    Note:
        Regarding the Paraver performance data, it should be taken to account:
            d_PAPI_TOT_INS: original
            n_PAPI_TOT_INS: normalized
            x_PAPI_TOT_INS: extrapolated
    """

    _df = pd.read_csv(data)
    _df.columns = _df.columns.str.strip()  # remove white space at both ends:

    if 'ThreadId' in _df.columns or 'TaskId' in _df.columns:
        update_config({'thread_number': len(set(_df['ThreadId'])), 'task_number': len(set(_df['TaskId']))})

    update_config({'number_of_clusters': len(set(_df[lab]))})

    for feature_item in CONFIG.get('eliminating_columns'):
        if feature_item in _df.columns:
            _df.drop(feature_item, axis=1, inplace=True)
        else:
            record_log(BColors.WARNING, "Preprocess", "The feature {} does not exist in the features.", feature_item)

    record_log(BColors.OKGREEN, "Preprocess", "The data {} is loaded successfully.", _df.shape)

    return _df


def pmm_missing_imputation(data, k_neighbors=None):
    """ Imputes the missing values by Partial Mean Matching (pmm) algorithm

    For any missing value it looks for the k nearest neighbors and replace the value with the wighted mean.
    The wights equal to the distances.

    Args:
        data (panda): Includes the original data.
        k_neighbors (int): Indicates the number of the nearest neighbors.

    Returns:
        A pandas includes the clean data.
    """

    def get_optimal_k(_k, data_set_size=None):
        """ Computes the optimal number of the neighbors

        Args:
            _k (int): Indicates the number of the nearest neighbors
            data_set_size (int): Indicates the data set size

        Returns:
            An int indicates the optimal value of the k
        """
        if not _k:
            if CONFIG.get('task_number') or CONFIG.get('thread_number'):
                _k = max(CONFIG.get('task_number'), CONFIG.get('thread_number')) - 1
                if _k < 1:
                    _k = 1
        if _k >= data_set_size or _k <= 1 or not CONFIG.get('task_number'):
            _k = round(math.sqrt(data_set_size))
            if _k % 2 == 0:
                _k += 1
        return _k

    preserved_index = list(data.index)
    data.reset_index(inplace=True, drop=True)

    # Todo: It needed to check and remove the rows that have NaN in 'cluster_with' and 'KNN_complimentary'
    trainer_feature = CONFIG.get('cluster_with') + CONFIG.get('KNN_complimentary')
    predicted_features = list(set(data.columns) - set(trainer_feature))

    for feature_item in predicted_features:
        # Train the KNN with the not NaN rows of the feature_item
        train_data = data.dropna(subset=[feature_item]).loc[:, [*trainer_feature, feature_item]]
        train_data.reset_index(inplace=True, drop=True)

        k = get_optimal_k(k_neighbors, train_data.shape[0])

        knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(train_data.loc[:, trainer_feature])

        nan_item_indexes = np.argwhere(data.loc[:, feature_item].isna().values).tolist()

        for nan_item in list(flatten(nan_item_indexes)):
            distances, _index = knn_model.kneighbors([data.iloc[nan_item, :][trainer_feature]])

            # Update with the wighted mean values of the predicted KNN.
            distances_wight = np.power(distances, -1)
            if np.isinf(distances_wight).any():
                # Replace nan with zero and inf with finite numbers.
                print(BColors.WARNING + "Any -/+ inf in PMM algorithm replaced with -/+ large number" + BColors.ENDC)
                distances_wight = np.nan_to_num(distances_wight)
            norm_wight = normalize(distances_wight, axis=1, norm='l1')
            data.at[nan_item, feature_item] = np.dot(norm_wight, np.array(train_data.iloc[_index[0, :]][feature_item]))
    data.index = preserved_index
    return data


def input_missing_values(data, method='pmm', plot=True):
    """ Imputes the missing values

    Args:
        data (pandas): Includes the data
        method (str): Indicates the imputation method
        plot (boolean): If True, the missing value heatmap will be plotted.

    Returns:
        A pandas includes the imputed data.

    Notes:
        Also see https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
    """
    imputation_method_dict = {'mean': 'Extrapolation',
                              'pmm': 'Partial Mean Matching algorithm',
                              'inter': 'Interpolation'}
    assert method.lower() in imputation_method_dict.keys(), "The imputation method does not exist."

    if plot:  # missing map
        fig, ax = plt.subplots()
        heatmap_plot = sns.heatmap(data.isna(), cbar=False)
        heatmap_plot.set_xticklabels(heatmap_plot.get_xticklabels(), rotation=45, horizontalalignment='right', size=10)
        fig.tight_layout()
        if CONFIG.get('make_jupiter_report'):
            plt.savefig('OCA_Report/pic/missing_value_plot.png', bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    data = data.replace([np.inf, -np.inf], np.nan)
    total_number_of_none = data.isna().sum().sum()

    if total_number_of_none > 0:
        if method.lower() == 'mean':
            data = data.fillna(data.mean())
        elif method.lower() == 'pmm':
            pmm_missing_imputation(data)
        elif method.lower() == 'inter':
            # Todo: interpolate -> pandas.DataFrame.interpolate
            pass
        record_log(BColors.OKGREEN, "Preprocess", "The {} missing values are imputed successfully by {}.",
                   total_number_of_none, imputation_method_dict.get(method.lower()))
    else:
        record_log(BColors.OKGREEN, "Preprocess", "Missing value imputation is bypassed (No Missing).")

    return data


def __filter(df, filter_config=None):
    """ Filters data

    Args:
        df (pandas): Includes the original data
        filter_config (list of tuple): Indicates filters configuration.

    Returns:
        A pandas includes the filtered data.

    Note:
        Example: {needed_features:[],operation_chain:'needed_features[0]>10e7'}
    """
    init_shape = df.shape
    for deriving_item in filter_config:
        needed_features = deriving_item.get('needed_features')
        operation_chain = deriving_item.get('operation_chain')

        for ind, needed_feature_item in enumerate(needed_features):
            operation_chain = operation_chain.replace("needed_features[{}]".format(ind),
                                                      "{}".format(needed_features[ind]))
        exec_command = "{}".format(operation_chain)
        df = df.query(exec_command)
        record_log(BColors.OKGREEN, "Preprocess", "{}/{} data points are eliminated by filter '{}'",
                   init_shape[0] - df.shape[0], df.shape[0], exec_command)
        init_shape = df.shape
    return df


def prepare_data(data, cluster_number=1, _derive=None, _filter=None, _outliers=None, _normalize=None, _sampling=None,
                 _engineer_features=None, _log_log=None, _ratio_divisor=None, _lab='ClusterID'):
    """ Applies the pre-processing over the original data.

    It applies the preprocessing steps which are defined in config file beforehand.

    Args:
        data (pandas): Includes the clustered data
        cluster_number (str): Indicate a cluster (label) number.
        _derive (dict): some features will be derived from the original features and attached to the dataset
        _filter (dict): it applies the filters which ar defined in config file
        _outliers (dict): it computes and eliminate the outliers
        _normalize (str): it normalizes the data e.g. max, l1, or l2
        _sampling (dict): it does the sampling
        _engineer_features (dict): it creates derived features
        _log_log (list): computes the log of selected features
        _ratio_divisor (str): It computes the ratio of the listed features
        _lab(str): Indicates the label feature

    Returns:
        A pandas includes the cleaned data.
    """
    data = data.loc[data[_lab].isin(cluster_number)]

    if _engineer_features:
        for engineering_feature_item in CONFIG.get('engineer_feature'):
            if engineering_feature_item == 'burst_overlying_factor'.lower():
                data['burst_overlying_factor'] = burst_overlying_factor(data.loc[:, ['Begin_Time', 'End_Time']])

    if _sampling.get("frac") != 1.0:
        data = data.sample(random_state=1, **_sampling)

    if _filter:
        data = __filter(data, CONFIG.get('filter'))

    if _outliers:
        data = drop_outliers(data, **_outliers)

    if _ratio_divisor:
        if CONFIG.get('ratio_invert'):
            data.loc[:, CONFIG.get('convert_to_ratio')] = data.loc[:, CONFIG.get('convert_to_ratio')].rdiv(
                data[CONFIG.get('ratio_divisor')], axis=0)
        else:
            data.loc[:, CONFIG.get('convert_to_ratio')] = data.loc[:, CONFIG.get('convert_to_ratio')].div(
                data[CONFIG.get('ratio_divisor')], axis=0)

    if _log_log:
        data.loc[:, CONFIG.get('log_log')] = np.log(data.loc[:, CONFIG.get('log_log')])

    if data.columns[data.isna().all()].tolist():
        print(BColors.WARNING + "Warning: The feature {} are all null and they will dropped down from the data set"
                                "before the analysis.".format(data.columns[data.isna().all()].tolist()) + BColors.ENDC)
        data.dropna(axis=1, how='all', inplace=True)

    if _derive:
        # Fixme: It needs to check it is ok here after imputation or before.
        data = derive_feature(data, CONFIG.get('derived_features'))
        print(BColors.WARNING + "Warning: The feature {} are all null and they will dropped down from the data set"
                                "before the analysis.".format(data.columns[data.isna().all()].tolist()) + BColors.ENDC)
        data.dropna(axis=1, how='all', inplace=True)
        update_config({'derived_features': CONFIG.get('derived_features'),
                       'number_of_derived_features': len(CONFIG.get('derived_features'))})

    # Drop not needed data and pass_trough features
    data.drop([_lab], axis=1, inplace=True)
    pass_trough_features = data.loc[:, CONFIG.get('pass_trough')]
    data.drop(CONFIG.get('pass_trough'), axis=1, inplace=True)

    data = input_missing_values(data)

    if _normalize:
        data = pd.DataFrame(normalize(data, axis=0, norm=_normalize), columns=data.columns)

    update_config({'number_of_features': data.shape[1] - len(CONFIG.get('derived_features')),
                   'number_of_observations': data.shape[0]})

    record_log(BColors.OKGREEN, "Preprocess", "The prepossessing has been completed successfully {}.", data.shape)

    return data, pass_trough_features


def plot_overall_scatter(df, features=None, log_log=[False, True], _clusteringsuit=False, lab='ClusterID'):
    """ Plots scatter of all clusters

    Args:
        log_log (boolean list): For any axis in the scatter plot, if it is True, the log transformation will applied.
        features (list): Includes x, y feature in order to plot the data
        df (pandas):  Includes the data nad a cluster_id label
        _clusteringsuit (boolean): If True, the first cluster is assumed as noise.
        lab(str): Indicates the label feature

    """
    if features is None:
        raise ValueError("Error: The feature (x,y) most be defined.")

    if any(log_log):
        axis_labels = []
        for axis, log_transfer in enumerate(log_log):
            if log_transfer:
                df.loc[:, features[axis]] = np.log(df.loc[:, features[axis]])
                axis_labels.append("Log( {} )".format(features[axis]))
            else:
                axis_labels.append(features[axis])
        df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, inplace=True)
    else:
        axis_labels = features

    color_list = get_color_palette(PARAVER_STATES_COLOR, None, True)

    if _clusteringsuit:
        reduction_index = 5
    else:
        reduction_index = 0

    fig, ax = plt.subplots()
    for cluster_item in list(set(df.loc[:, lab])):
        df1 = df[df[lab] == cluster_item]

        if (cluster_item - reduction_index) == 0 and _clusteringsuit:
            _label = 'Noise'
        else:
            _label = 'Cluster_{}'.format(cluster_item - reduction_index)

        ax.scatter(df1.loc[:, features[0]],
                   df1.loc[:, features[1]],
                   c=color_list[cluster_item - reduction_index],
                   s=20,
                   marker=MARKER[cluster_item - reduction_index],
                   label=_label)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])

    if CONFIG.get('make_jupiter_report'):
        plt.savefig('OCA_Report/pic/over_all_scatter_plot.png', bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def derive_feature(df, deriving_config):
    """ Drives features

    It engineers new features by the defined formulation from the original features.

    Args:
        df (pandas): Includes the original data
        deriving_config (list of tuple): Indicates a derived feature configuration.

    Returns:
        A pandas includes the derived and original data.

    Notes:
        Example: {name:'derived_name', needed_features:[],operation_chain:'needed_features[0]/needed_features[1]'}
    """
    for deriving_item in deriving_config:
        name = deriving_item.get('name')
        needed_features = deriving_item.get('needed_features')
        operation_chain = deriving_item.get('operation_chain')
        for ind, needed_feature_item in enumerate(needed_features):
            operation_chain = operation_chain.replace("needed_features[{}]".format(ind),
                                                      'row.{}'.format(needed_features[ind]))
        exec_command = "df[name] = df.apply(lambda row: {}, axis=1)".format(operation_chain)
        exec(exec_command)

        record_log(BColors.OKGREEN, "Preprocess", "The feature {} is driven successfully by '{}'.",
                   name, operation_chain.replace('row.', ''))
    return df


def update_oca_report_content(_inertia='geodesic'):
    """ Updates the OCA Report contents.

    Args:
        _inertia (str): Indicates the type of the inertia either 'geodesic' or 'euclidean'

    Returns:
        A dict includes the content materials.
    """
    if _inertia == 'geodesic':
        primary_inertia = 'geodesic_Primary_Inertia'
        secondary_inertia = 'geodesic_Secondary_Inertia'
    else:
        primary_inertia = 'Primary_Inertia'
        secondary_inertia = 'Secondary_Inertia'

    # Build header
    with open('.oca_report_hypothesis/ocs_report_header_content.json', 'r') as conf:
        header_dict = json.load(conf)
        style_material = header_dict.get('style_material')
        menu_material = header_dict.get('menu_material')
        version_material = header_dict.get('version_material').format(header_dict.get('kavica_ver'),
                                                                      header_dict.get('kavica_git'))
        config_material = header_dict.get('config_material').format(os.path.abspath(CONFIG.get('conf_path')),
                                                                    os.path.basename(CONFIG.get('conf_path')))
        commandline_material = header_dict.get('commandline_material').format(CONFIG.get('conf_path'),
                                                                              CONFIG.get('csv_path'),
                                                                              CONFIG.get('cluster_with'),
                                                                              CONFIG.get('cluster_number'),
                                                                              CONFIG.get('SOM_Size'),
                                                                              CONFIG.get('epoch_number'))

    # Build body
    with open(".oca_report_hypothesis/oca_report_platform.html", "rt") as oca_report_platform:
        oca_material = oca_report_platform.read()
        inertia_content = ''
        for key_item in dict(CONFIG.get(primary_inertia)).keys():
            primary_i = CONFIG.get(primary_inertia, {}).get(key_item)
            primary_i = round(primary_i, 3) if isinstance(primary_i, float) else str(primary_i)
            secondary_i = CONFIG.get(secondary_inertia, {}).get(key_item)
            secondary_i = round(secondary_i, 3) if isinstance(secondary_i, float) else str(secondary_i)
            inertia_content = inertia_content + '\n<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(key_item,
                                                                                                      primary_i,
                                                                                                      secondary_i)
        oca_data_map_dict = {'#number_of_clusters': CONFIG.get('number_of_clusters'),
                             '#cluster_by': ', '.join(map(str, CONFIG.get('cluster_with'))),
                             '#selected_clusters': ', '.join(map(str, CONFIG.get('cluster_number'))),
                             '#number_of_original_variables': CONFIG.get('number_of_features'),
                             '#number_of_derived_variables': CONFIG.get('number_of_derived_features'),
                             '#number_of_observations': CONFIG.get('number_of_observations'),
                             '#ANN_size': ' x '.join(map(str, CONFIG.get('SOM_Size'))),
                             '#epoch': CONFIG.get('epoch_number'),
                             '#upper_bound_acc': "{:.2%}".format(CONFIG.get('topology_accuracy',
                                                                            {}).get('upper_bound')),
                             '#lower_bound_acc': "{:.2%}".format(CONFIG.get('topology_accuracy',
                                                                            {}).get('lower_bound')),
                             '#accuracy': "{:.2%}".format(CONFIG.get('topology_accuracy', {}).get('accuracy')),
                             '#kolgomorov_smirnov_test': "{:.2%}".format(CONFIG.get('likelihood_test')),
                             '#embedding_index_test': "{:.2%}".format(
                                 CONFIG.get('embedding_index_test')) if CONFIG.get('embedding_index_test') else 'None',
                             '#inertia_content': inertia_content,
                             '#outliers_algorithm': CONFIG.get('Outlier detection', {}).get('algorithm'),
                             '#outliers_contamination': "{:.2%}".format(CONFIG.get('Outlier detection',
                                                                                   {}).get('fraction')),
                             '#outliers_n_samples': CONFIG.get('Outlier detection', {}).get('n_samples'),
                             '#Number_Outliers_detected': CONFIG.get('outliers_number')}

        for map_key, map_value in oca_data_map_dict.items():
            oca_material = oca_material.replace(str(map_key), str(map_value))

    record_log(BColors.OKGREEN, "Report", 'OCA report content is ready.')

    return {'version_material': version_material,
            'config_material': config_material,
            'commandline_material': commandline_material,
            'style_material': style_material,
            'menu_material': menu_material,
            'oca_material': oca_material}


def get_oca_report(df, open_in_browser=True, output_file='OCA_report.html', title='Organization Component Analysis'):
    """ Writes the data report in .html format

    Create two files:
        1- The report information (json)
        2- The Organization Component Analysis Report (html)

    Args:
        df (pandas): Includes the dataset
        open_in_browser (boolean): If True, The final report opens in browser.
        output_file (str): Output file name and path
        title (str): The report title

    """

    profile = ProfileReport(df=df, title=title, html={'style': {'full_width': True}})
    profile.to_file(output_file="OCA_Report/{}".format(output_file))

    # Information
    with open('OCA_Report/result.json', 'w') as fp:
        json.dump(CONFIG, fp)

    # Update the report content
    oca_report_contents = update_oca_report_content()
    with open("OCA_Report/{}".format(output_file), "rt") as report_file:
        data = report_file.read()
        data = data.replace('</head>', oca_report_contents.get('style_material'))
        data = re.sub(r'Download configuration.*?yaml</a>', oca_report_contents.get('config_material'), data)
        data = data.replace('<li><a class=anchor href=#overview>Overview</a></li>',
                            oca_report_contents.get('menu_material'))
        data = data.replace('<div class=content><div class=container-fluid>',
                            '\n<div class=content><div class=container-fluid>{}'.format(
                                oca_report_contents.get('oca_material')))
        data = data.replace('pandas_profiling --config_file config.yaml [YOUR_FILE.csv]',
                            oca_report_contents.get('commandline_material'))
        data = re.sub(r'<a href=https://github.com/pandas-profiling/pandas-profiling>pandas-profiling.*?</a>',
                      oca_report_contents.get('version_material'), data)

    with open("OCA_Report/{}".format(output_file), "wt") as report_file:
        report_file.write(data)

    record_log(BColors.OKGREEN, "Report", "Final OCA report is generated, 'OCA_Report/{}'", output_file)

    if open_in_browser:
        gc.collect()
        try:
            webbrowser.get('firefox').open_new_tab('OCA_Report/{}'.format(output_file))
        except (RuntimeError, TypeError, NameError):
            webbrowser.get('firefox').open_new('OCA_Report/{}'.format(output_file))


def initialize_report_directory(parent_path='OCA_Report'):
    """ Deletes the old data and directory (plots, result and final report Html)

    Returns:

    """
    eliminate_items = ['result.json', 'pic', 'OCA_report.html']
    create_items = ['pic']

    def remove(_path):
        """ Removes the file

        Args:
            _path (str): Indicates a file/folder path.
        """
        try:
            if os.path.isfile(_path) or os.path.islink(_path):
                os.remove(_path)  # remove the file
                record_log(BColors.OKGREEN, "Initialize", "File {} has been removed successfully", _path)
            elif os.path.isdir(_path):
                shutil.rmtree(_path)  # remove dir and all contains
                record_log(BColors.OKGREEN, "Initialize", "Directory {} has been removed successfully", _path)
            else:
                record_log(BColors.WARNING, "Initialize", "{} dose not exist.", _path)
        except OSError:
            print("Directory {} can not be removed".format(_path))

    # Remove the old files
    for eliminate_item in eliminate_items:
        path = os.path.join(parent_path, eliminate_item)
        remove(path)

    # Create the new files
    for create_item in create_items:
        path = os.path.join(parent_path, create_item)
        if not os.path.exists(path):
            os.makedirs(path)
            record_log(BColors.OKGREEN, "Initialize", "Directory {} has been created successfully", path)

    record_log(BColors.OKGREEN, "Initialize", "Final report directory {} has been cleared successfully", parent_path)


def organization_component_analysis():
    """Handles the command line"""
    start = time.time()

    try:
        initialize_report_directory()  # initialize final report directory
        args = arguments_parser()  # read the arguments
        configure(args['conf_path'])  # load the configuration
        df = load_data(args['csv_path'])  # load the data

        # scatter plot of the overall data
        plot_overall_scatter(df.copy(),
                             features=CONFIG.get('cluster_with'),
                             _clusteringsuit=CONFIG.get('clusteringsuit'))

        # Prepare data
        df, _pass_troughs = prepare_data(df, args['cluster_number'],
                                         _engineer_features=CONFIG.get('engineer_feature'),
                                         _derive=CONFIG.get('derive_feature'),
                                         _filter=CONFIG.get('filter'),
                                         _outliers=CONFIG.get('Outlier detection'),
                                         _normalize=CONFIG.get('normalize'),
                                         _sampling=CONFIG.get('sampling'),
                                         _log_log=CONFIG.get('log_log'),
                                         _ratio_divisor=CONFIG.get('ratio_divisor'))

        df.reset_index(inplace=True, drop=True)
        _cluster_with = copy.deepcopy(CONFIG.get('cluster_with'))

        print(df.describe().loc[['mean', 'std'], _cluster_with])

        print(BColors.OKBLUE, pd.DataFrame(normalize(df, axis=0, norm='max'), columns=df.columns).describe(),
              BColors.ENDC)

        m = SOMIS(map_shape=args['SOM_Size'], epoch=args['epoch_number'], training_rate=0.1,
                  neighborhood_size=1, grid_features=list(reversed(_cluster_with)))
        m.fit(df, init_method='rrg')

        # Validate
        for item_key, item_value in m.get_goodness_of_fit(full_list=True).items():
            print(item_key, ":", item_value)
            update_config({item_key: item_value})

        # Get inertia
        inertia_data = m.get_inertia(_cluster_with, normalized=True, alpha=0.01, print_table=False,
                                     optimal_alpha_value=False, _norm='l2', by=CONFIG.get('similarity_by'))

        for inertia_item_key, inertia_item_value in inertia_data.to_dict('index').items():
            update_config({inertia_item_key: inertia_item_value})

        # geodesic test_configs
        geodesic_inertia_data = m.get_inertia(_cluster_with, normalized=True, alpha=0.01,
                                              optimal_alpha_value=False, _norm='l2', by=CONFIG.get('similarity_by'),
                                              print_table=False, divergence='geodesic')

        for geodesic_inertia_item_key, geodesic_inertia_item_value in geodesic_inertia_data.to_dict('dict').items():
            update_config({"geodesic_{}".format(geodesic_inertia_item_key): geodesic_inertia_item_value})

        # Plotting
        _cluster_with = copy.deepcopy(CONFIG.get('cluster_with'))

        m.plot_weight_positions(_cluster_with[0], _cluster_with[1])
        m.plot_weight_planes(feature='all')
        m.plot_weight_positions(_cluster_with[0], _cluster_with[1])
        m.plot_sample_hit()
        m.plot_neurons_guid()
        m.plot_neighbor_distance_hexbin()
        m.plot_inertia_drivers_slopes()
        m.plot_inertia_drivers_slopes(inertia='secondary')
        m.plot_inertia_drivers_asymmetric_bar(features=_cluster_with)
        m.plot_unified_distance_matrix(explicit=True, merge_clusters=False)

        # Generate the HTML report
        get_oca_report(df)

        print("\033[32mThe Cluster Self Organization Map Inference Systems process is successfully completed.")
        return 0
    except AssertionError as error:
        print("\033[31mThe Cluster Self Organization Map Inference Systems proses is failed.")
        print(error)
    finally:
        duration = time.time() - start
        print('\033[0mTotal duration is: %.3f' % duration)


if __name__ == '__main__':
    organization_component_analysis()

# __test_me
# python3 OCA.py config/iris.json ../../data/oca_test_data/iris.csv --cw 'sepal_width' 'sepal_length' -c 1 --s 6 6  -i 700
