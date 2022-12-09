#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Models

A set of models which are applied to test_configs package.

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
First update: 07/10/2022
"""
from kavica.utils import cat2int
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.cluster import OPTICS, cluster_optics_dbscan  # DBSCAN
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import matplotlib.gridspec as gridspec
from itertools import count
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kavica.utils._bcolors import BColors
from sklearn.metrics import silhouette_score
from matplotlib.font_manager import FontProperties
from kavica.utils._plots import (get_color_map, PARAVER_STATES_COLOR)
import sys
import random

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize, linewidth=200)
_seed = 10
random.seed(_seed)
np.random.seed(_seed)
plt.rcParams.update({'font.size': 16})

__all__ = [
    "SVM",
    "balance",
    "cluster_analysis",
    "class_analysis",
    "get_search_space",
    "dbscan",
    "optics_dbscan",
]


def balance(df, _lab='lab', _status='over', _strategy='not majority'):
    """ Balance the data set for the classification

    Args:
        df:
        _lab:
        _status (str): It can be:
            . Over : This would ensure that the minority class was oversampled to have proportional number of examples
                     as the majority class, for binary classification problems.
            . Under: For example, a dataset with 1,000 examples in the majority class and 100 examples in the minority
                     class will be under-sampled such that both classes would have 100 examples in the transformed
                     training dataset.
        _strategy (str): Indicates the sampling strategy
            - 'majority': resample only the majority class;
            - 'not minority': resample all classes but the minority class;
            - 'not majority': resample all classes but the majority class;
            - 'all': resample all classes;
            - 'auto': equivalent to 'not minority'.
    See:
        https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.RandomUnderSampler.html

    Returns:
        A pandas which is balanced
    """
    print(BColors.OKGREEN + 'Initial class balance: \n {}'.format(df[_lab].value_counts()) + BColors.ENDC, )
    if _status == 'under':
        rs = RandomUnderSampler(sampling_strategy=_strategy, random_state=4)
    elif _status == 'over':
        rs = RandomOverSampler(sampling_strategy=_strategy, random_state=4)
    df, _ = rs.fit_resample(df, df[_lab])
    print(BColors.OKGREEN + 'Updated class balance: \n {}'.format(df[_lab].value_counts()) + BColors.ENDC)
    return df


def SVM(data, target, _kernel='linear', _tune=False, _plot=True, _test_size=0.30, axis_lab=[]):
    """ Runs the SVM classification and plot the result

    Args:
        data:
        target:
        _kernel:
        _tune:
        _plot:
        _test_size:
        axis_lab:

    Returns:

    """
    target = cat2int(target)
    X_train, X_test, y_train, y_test = train_test_split(data, np.ravel(target),
                                                        test_size=_test_size, random_state=4)

    # train the model on train set
    model = SVC(kernel=_kernel, random_state=4)
    model.fit(X_train, y_train)

    # print prediction results
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    print(BColors.OKGREEN + 'The number of support vectors per class: {}'.format(model.n_support_) + BColors.ENDC)

    def __plot_decision_boundaries(clf, X, Y, h=1):
        """

        Args:
            X:
            Y:
            h:

        Returns:

        Note: https://scikit-learn.org/0.18/auto_examples/svm/plot_iris.html
        """

        # create a mesh to plot in
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        fig, ax = plt.subplots(figsize=(16, 16), dpi=100)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.cool, alpha=0.8, levels=len(list(set(Y))) - 1)  #
        # print(Z[list(reversed(range(Z.shape[0]))),:])

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.hot)
        plt.xlabel(axis_lab[0])
        plt.ylabel(axis_lab[1])
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.show()

    if _tune:
        # defining parameter range
        param_grid = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['linear', 'rbf']}  # 'poly' # 'sigmoid'

        grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

        # fitting the model for grid search
        grid.fit(X_train, y_train)

        # print best parameter after tuning
        print(grid.best_params_)

        # print how our model looks after hyper-parameter tuning
        print(grid.best_estimator_)

        grid_predictions = grid.predict(X_test)

        # print classification report
        print(classification_report(y_test, grid_predictions))

    if _plot:
        __plot_decision_boundaries(clf=model, X=X_train.to_numpy(), Y=y_train)


def cluster_analysis(pickle_path, _eps, _min_points, x_axis=0, y_axis=1, _grid=True,
                     _model='dbscan', comp=7, compulsory=[], _lab=None, _s=20,
                     _x_lab="", _y_lab="", duration=None):
    df = pd.read_pickle(pickle_path)
    dbscan(df, _eps, _min_points, x_axis=x_axis, y_axis=y_axis, show_grid=_grid, _x_lab=_x_lab, _y_lab=_y_lab,
           method=_model, comp=comp, _compulsory=compulsory, _lab=_lab, _s=_s, _lab_font=22,
           duration=duration)


def class_analysis(df=None, pickle_path=None, _lab='lab', features=[0, 1], _balance=True,
                   _tun=False, _kernel='linear', _test_size=0.30, axis_lab=[]):
    """

    Args:
        pickle_path:
        _lab:
        features:

    Returns:

    See:
        https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
    """
    # ETL and preparation
    if pickle_path:
        df = pd.read_pickle(pickle_path)
    else:
        df = df.copy()

    if _balance:
        df = balance(df, _lab)

    df_feat = df[features]
    df_target = df[_lab]
    SVM(df_feat, df_target, _tune=_tun, _kernel=_kernel, _test_size=_test_size, axis_lab=axis_lab)


def get_search_space(x=None, data=None, cov=None, _k=10):
    def get_ks(x=None, data=None, cov=None):
        return [4, 64]

    def get_eps(df, _ks, _lambda=10):
        """

        Args:
            df:
            _ks:
            _k:

        Returns:

        Note:
            https://kneed.readthedocs.io/en/stable/parameters.html
        """
        __eps = []
        plt.style.use("ggplot")
        plt.figure(figsize=(10, 10))
        for _k in _ks:
            neigh = NearestNeighbors(n_neighbors=_k)
            nbrs = neigh.fit(df[[0, 1]])
            distances, indices = nbrs.kneighbors(df[[0, 1]])
            distances = np.mean(distances[:, 1:], axis=1)
            distances = np.sort(distances, axis=0)
            distances = distances * _lambda
            _points = dict(enumerate(distances))
            kn = KneeLocator(x=list(_points.keys()),
                             y=list(_points.values()),
                             curve='convex',
                             direction='increasing')
            # Plotting K-distance Graph
            plt.plot(distances, label=f"MinPts = {_k}, eps={round(_points.get(kn.knee), 6)}")
            plt.hlines(_points.get(kn.knee), 0, kn.knee, linestyles="--", colors='k', linewidth=1)
            plt.vlines(kn.knee, 0, _points.get(kn.knee), linestyles="--", colors='k', linewidth=1)
            __eps.append(_points.get(kn.knee))
        plt.title('Knee Locator', fontsize=20)
        plt.xlabel('Data Points sorted by distance', fontsize=14)
        plt.ylabel('Epsilon', fontsize=14)
        plt.legend()
        plt.show()

        return __eps

    __ks = get_ks(x, data, cov)
    __epss = get_eps(x, __ks)

    # Plot hte search space
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a grid of x and y
    xg, yg = np.meshgrid(np.linspace(__ks[0], __ks[1], 10), np.linspace(__epss[0], __epss[1], 10))
    z = 0 * xg
    ax.plot_wireframe(xg, yg, z, rstride=1, cstride=1, color='blue')
    ax.set_xlabel('MinPts', labelpad=10)
    ax.set_ylabel('eps', labelpad=10)
    ax.set_zlabel('Clustering Evaluation Metric for Clustring',
                  labelpad=10)  # Clustering Evaluation Metric for Clustring
    plt.title('Grid Search Space')
    ax.set_zlim3d(0, 1.2)
    plt.show()
    print("The search space is: {}".format({'MinPts': __ks, 'eps': __epss}))
    return {'MinPts': __ks, 'eps': __epss}


def dbscan(df, _eps=0.5, _min_points=5, x_axis=0, y_axis=1, _lab=None, _plot=True, show_grid=True, method='dbscan',
           comp=7, _compulsory=[], _norm=False, _s=40, _lab_font=16, _x_lab="", _y_lab="", duration=None,
           _map_c=False, _tune=False, _experiment=True, _output='./outputs/cluster_ID.csv'):
    __original_df = df.copy()

    """ Dbscan clustering and plotting

    Args:
        _s:
        _norm:
        _compulsory:
        comp:
        show_grid:
        _lab:
        df (pandas): Data points
        y (1D array): The original label of the data points
        _eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of
                      the other. This is not a maximum bound on the distances of points within a cluster. This is
                      the most important DBSCAN parameter to choose appropriately for your data set and distance
                      function.
        _min_points (int): The number of samples (or total weight) in a neighborhood for a point to be considered
                           as a core point. This includes the point itself.
        x_axis (int): The (index of) feature is plotted in x axis
        y_axis (int): The (index of) feature is plotted in y axis
        _plot (boolean): Show the scatter plot
        method (str): Clustering method that can be Dbscan or OPTIC
        _output (str): Indicates the output path for the clustering result

    Returns:
        An 1d array includes the cluster_id of each data point
    """
    # data pre_processing
    nan_number = np.count_nonzero(df.isnull().values)
    if nan_number:
        print(BColors.WARNING +
              'Dbscan >> {} of {} rows with None are dropped.'.format(nan_number, df.shape[0]) + BColors.ENDC)
        df.dropna(inplace=True)

    if _norm:
        df = (df - df.mean()) / df.std()

    if method == 'dbscan':
        # Search space
        if _tune:
            __search_space = get_search_space(df[[x_axis, y_axis]], df[[x_axis, y_axis]])
            _eps, _min_points = __search_space['eps'][1], __search_space['MinPts'][1]

        # clustering
        db = DBSCAN(eps=_eps, min_samples=_min_points)
        db.fit(df[[x_axis, y_axis] + _compulsory])
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # silhouette analysis
        silhouette_avg = silhouette_score(df[[x_axis, y_axis]], labels)
        print(silhouette_avg)

        title_text = "Dbscan result (eps={} , min_points={})".format(_eps, _min_points)
    elif method == 'optics':
        db = OPTICS(min_samples=_min_points).fit(df)
        labels = db.labels_
        title_text = "Optic result (eps={} , min_points={})".format(_eps, _min_points)
    elif method == 'GMM':
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=comp)
        gmm.fit(df)
        labels = gmm.predict(df)
        title_text = "GMM result (Components={})".format(comp)

    if duration is not None:
        df['ClusterID'] = labels
        df['duration'] = duration
        _agr_duration = df.groupby('ClusterID').sum()
        if _agr_duration.index[0] == -1:
            _agr_duration = _agr_duration[1:]
        __sorted_color = np.flip(_agr_duration['duration'].argsort()).reset_index(drop=True)
        __sorted_color = __sorted_color.to_dict()
        __sorted_color = dict([(value, key) for key, value in __sorted_color.items()])
        df['ClusterID'].replace(__sorted_color, inplace=True)
    else:
        df['ClusterID'] = labels

    df.to_csv(_output)

    label_true = df[_lab].to_numpy()
    # Evaluation
    if _lab:
        print("-" * 20, "Evaluation", "-" * 20)
        print("Normalized Mutual Information: {} ".format(metrics.normalized_mutual_info_score(label_true, labels)))
        """
        Upper bound of 1: Values close to zero indicate two label assignments that are largely independent, while 
        values close to one indicate significant agreement. Further, an AMI of exactly 1 indicates that the two label 
        assignments are equal (with or without permutation)
        """
        print("Mutual Information: {} ".format(metrics.adjusted_mutual_info_score(label_true, labels)))
        print("Homogeneity: {} ".format(metrics.homogeneity_score(label_true, labels)))
        print("V-measure: {} ".format(metrics.v_measure_score(label_true, labels)))
        print("V-measure: {} ".format(metrics.v_measure_score(label_true, labels)))
        print("Fowlkes-Mallows scores: {} ".format(metrics.fowlkes_mallows_score(label_true, labels)))
        print("-" * 20, "Evaluation", "-" * 20)
    # Report: Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    unique_labels = set(labels)
    # get_color_map
    if _plot:
        color_list = get_color_map(PARAVER_STATES_COLOR, _hex=False)
        # Fixme:color map if needed, not needed now.
        """
        if (duration is not None) & _map_c:
            df['duration'] =duration
            _agr_duration=df.groupby('ClusterID').sum()
            __sorted_color=np.flip(_agr_duration['duration'].argsort()).reset_index(drop=True)
            __sorted_color= __sorted_color.to_dict()
            __sorted_color = dict([(value, key+1) for key, value in __sorted_color.items()])
            _color_temp=[1,2,3,4,5,6,9,7,14,12,8,10,11,15,13,20,30,30,30,30,30,30,30,30,30] FSCM L1
        """
        fig, ax = plt.subplots(figsize=(14, 14), dpi=100)
        for cluster_item in list(unique_labels):
            cluster_item = cluster_item
            cluster_item_df = df[df['ClusterID'] == cluster_item]
            if cluster_item in [-1]:
                # Noise
                ax.scatter(cluster_item_df[[x_axis]],
                           cluster_item_df[[y_axis]],
                           c=color_list[0],
                           s=_s * 0.7,
                           marker='x',
                           label='Noise')
            else:
                ax.scatter(cluster_item_df[[x_axis]],
                           cluster_item_df[[y_axis]],
                           c=color_list[cluster_item + 1],
                           s=_s,  # edgecolor='k',
                           marker='o',
                           label='Cluster_{}'.format(cluster_item + 1))

        ax.margins(0.15)
        plt.xticks(rotation=90)
        plt.title(title_text)
        fontP = FontProperties()
        fontP.set_size('small')
        ax.legend(loc='best', prop=fontP)  # bbox_to_anchor=(1, 0.5)
        if show_grid:
            ax.grid(linestyle='-', which='major')  # , color='k',
            # ax.set_aspect('equal', adjustable='box')

        ax.tick_params(axis='x', labelsize=_lab_font)
        ax.tick_params(axis='y', labelsize=_lab_font)
        plt.xlabel(_x_lab)
        plt.ylabel(_y_lab)
        plt.show()
    return labels


def optics_dbscan(_df, _min_points=30, x_axis=0, y_axis=1):
    X = _df.to_numpy()
    clust = OPTICS(min_samples=_min_points)
    clust.fit(X)

    space = np.arange(len(X))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]

    eps_list = np.arange(10.6, 15.2, 0.2)

    # Reachability plot
    color_list = ['Blue', 'green', 'orange',
                  'fuchsia', 'mediumpurple', 'y',
                  'aqua', 'yellowgreen', 'k', 'darkkhaki',
                  'seagreen', 'maroon', 'lime',
                  'pink', 'peru', 'teal', 'plum', 'navy',
                  'orange', 'gold', 'sienna', 'coral', 'olive']

    # Reachability graph
    fig, ax = plt.subplots(figsize=(12, 12), dpi=200)
    for klass in range(max(labels) + 1):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax.plot(Xk, Rk, color_list[klass], alpha=0.3, marker='.', linestyle='-')
    ax.plot(space[labels == -1], reachability[labels == -1], 'r+', alpha=0.3)
    ax.set_ylabel('Reachability (epsilon distance)')
    ax.set_title('Reachability Plot')

    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='#666666', linestyle='-')
    ax.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.show()

    # Overall scatter plot
    labels = clust.labels_
    _df['ClusterID'] = labels
    unique_labels = set(labels)
    fig, ax = plt.subplots(figsize=(16, 16))
    for cluster_item in list(unique_labels):
        cluster_item = cluster_item
        cluster_item_df = _df[_df['ClusterID'] == cluster_item]
        if cluster_item == -1:
            # Noise
            ax.scatter(cluster_item_df[[x_axis]],
                       cluster_item_df[[y_axis]],
                       c='red',
                       s=20,
                       marker='x',
                       label='Noise')
        else:
            ax.scatter(cluster_item_df[[x_axis]],
                       cluster_item_df[[y_axis]],
                       c=color_list[cluster_item],
                       s=30, edgecolor='k',
                       marker='o',
                       label='Cluster_{}'.format(cluster_item + 1))

    ax.margins(0.15)
    plt.xticks(rotation=90)
    plt.title("Automatic Clustering OPTICS (min_points={})".format(_min_points))
    ax.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(1))
    fontP = FontProperties()
    fontP.set_size('x-small')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=fontP)
    ax.margins(0.15)
    plt.grid()
    plt.show()

    plt.figure(figsize=(40, 28))
    G = gridspec.GridSpec(4, 6)
    for x_loc in range(0, 4):
        for y_loc in range(0, 6):
            locals()['ax{}'.format(x_loc * 6 + y_loc)] = plt.subplot(G[x_loc, y_loc])

    # OPTICS
    for klass in range(max(labels) + 1):
        Xk = X[clust.labels_ == klass]
        locals()['ax{}'.format(0)].plot(Xk[:, 0], Xk[:, 1], color_list[klass],
                                        alpha=0.6, marker='*', linestyle='None')
    locals()['ax{}'.format(0)].plot(X[clust.labels_ == -1, 0], X[clust.labels_ == -1, 1], 'r+', alpha=0.6)
    locals()['ax{}'.format(0)].set_title('Automatic Clustering OPTICS')

    for item_index, eps_item in zip(count(), eps_list):
        local_labels = cluster_optics_dbscan(reachability=clust.reachability_,
                                             core_distances=clust.core_distances_,
                                             ordering=clust.ordering_, eps=eps_item)
        for klass in range(max(local_labels) + 1):
            Xk = X[local_labels == klass]
            locals()['ax{}'.format(item_index + 1)].plot(Xk[:, 0], Xk[:, 1], color_list[klass],
                                                         alpha=0.6, marker='*', linestyle='None')
        locals()['ax{}'.format(item_index + 1)].plot(X[local_labels == -1, 0], X[local_labels == -1, 1],
                                                     'r+', alpha=0.6)
        locals()['ax{}'.format(item_index + 1)].set_title('Clustering at {:.2f} epsilon cut DBSCAN'.format(eps_item))
    plt.tight_layout()
    plt.show()
