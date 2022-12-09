#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Exploratory Analysis over the raw data

It is used with print statement.

Note:
    https://plotly.com/python/pca-visualization/
    https://towardsdatascience.com/how-exactly-umap-works-13e3040e1668 (UMAP + t-SNE)
    https://umap-learn.readthedocs.io/en/latest/basic_usage.html

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
first update: 29/11/2021
last update: 09/11/2022
"""

import pandas as pd
import json
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
from kavica.utils._bcolors import BColors
from sklearn.metrics import silhouette_score
from kavica.utils._util import cat_num_columns, num2cat
import plotly.express as px
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from kavica.utils._plots import scatter_plot, get_discrete_cmp, get_color_map, PARAVER_STATES_COLOR, MARKER
from pandas_profiling import ProfileReport
import webbrowser
import multiprocessing
from kavica.utils._util import cat2int
from kavica.utils._models import cluster_analysis
import matplotlib

matplotlib.use('module://backend_interagg')

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
np.set_printoptions(threshold=sys.maxsize, linewidth=200)
SEED = 10
random.seed(SEED)
np.random.seed(SEED)
plt.rcParams.update({'font.size': 16})

__all__ = [
    "factorize",
    "exploratory_analysis"
]

COLOR_LIST = get_color_map(PARAVER_STATES_COLOR, _hex=True)
CPUs_AVAIL = multiprocessing.cpu_count()
OUTPUT_PATH = "../cluster_inference_system/empirical_report/FSCM_paper_dataset/emperical_final_2d"


def balance_analysis(df, label, _plot=False, _seed=4, major=True):
    """ Analysis the  class balance in the dataset

    Args:
        df (pandas): Includes the dataset
        label (str): Indicates the class labels
        _plot (boolean): If True, It draw the plot.
        _seed (int) : Indicates the random state
        major (boolean): If True, it uses the big class size

    Returns:
        A pandas includes the balanced data
    """
    class_balance = df[label].value_counts()
    print("Initial class counts:\n", df[label].value_counts())

    if _plot:
        class_balance.plot(kind='pie', fontsize=12)
        plt.legend(labels=class_balance.index, loc="best")
        plt.axis('equal')
        plt.show()

    _size = class_balance.max() if major else class_balance.min()

    _groups = df.groupby(label)
    df_groups = [x.sample(_size, replace=True, random_state=_seed) for y, x in _groups]
    balance_df = pd.concat(df_groups)
    print("Initial class counts:\n", balance_df[label].value_counts())

    return balance_df.reset_index(drop=True)


def log_transfer(df, label, _columns=None):
    """ Computes the log of the features

    Args:
        df (pandas): Includes the dataset
        label (str): Indicates the class labels
        _columns (list or None): Indicates the list of the features to transfer. If None, it computes the log of all
                             numerical features


    Returns:
        A pandas
    """
    if _columns is None:
        _columns = cat_num_columns(df).get('num')

    df.style.set_precision(2)
    for column in _columns:
        try:
            if column != label:
                df[column] = np.log10(df[column])
        except (ValueError, AttributeError):
            pass
    return df


def sampling_k_elements(group, k=3):
    if len(group) < k:
        return group
    return group.sample(k, random_state=10)


def factorize(algorithm, X, y=None, _n_components=2, _norm=True, _cmap=px.colors.qualitative.Dark24, _output_name=None,
              _output_path=None, _typ='2d', **kwargs):
    def __plot(_embedding, _n_components, _typ, _cmap, __lab, method='pca', _labels=None):
        _embedding.sort_values(by=[__lab], inplace=True)
        _embedding = _embedding.astype({str(__lab): str})

        if method.lower() == 'pca':
            __labels = _labels
        else:
            __labels = {str(i): "COM {}".format(i) for i in range(_n_components)}

        if _typ == 'matrix':
            if __lab is None:
                fig = px.scatter_matrix(_embedding, dimensions=range(_n_components), labels=__labels,
                                        color_discrete_map=_cmap, title=method.upper())
            else:
                fig = px.scatter_matrix(_embedding, dimensions=range(_n_components), color=__lab, labels=__labels,
                                        color_discrete_map=_cmap, title=method.upper())
            fig.update_traces(diagonal_visible=False)
        elif _typ == '2d':
            if __lab is None:
                fig = px.scatter(_embedding, x=0, y=1, width=800, height=600, title=method.upper(), labels=__labels,
                                 color_discrete_map=_cmap)
            else:
                fig = px.scatter(_embedding, x=0, y=1, color=__lab, title=method.upper(), labels=__labels,
                                 width=800, height=600, color_discrete_map=_cmap)
        elif _typ == '3d':

            if __lab is None:
                fig = px.scatter_3d(_embedding, x=0, y=1, z=2, width=800, height=600,
                                    title=method.upper(),
                                    labels=__labels,
                                    color_discrete_map=_cmap)
            else:
                fig = px.scatter_3d(_embedding, x=0, y=1, z=2, color=__lab, width=800, height=600,
                                    color_discrete_map=_cmap, title=method.upper(), labels=__labels)
            fig.update_traces(marker=dict(size=1, line=dict(width=0)))
        fig.show()
        return 0

    __lab = y.name

    if _norm:
        X = X.copy()
        scaler = StandardScaler()
        X.iloc[:, :] = scaler.fit_transform(X.to_numpy())

    if algorithm.lower() == 'pca':
        reducer = PCA(n_components=_n_components, random_state=SEED, **kwargs)
    elif algorithm.lower() == 'umap':
        reducer = umap.UMAP(n_components=_n_components, random_state=SEED, n_jobs=CPUs_AVAIL, **kwargs)
    elif algorithm.lower() == 'tsne':
        reducer = TSNE(n_components=_n_components, random_state=SEED, n_jobs=CPUs_AVAIL, **kwargs)
    elif algorithm.lower() == 'isomap':
        reducer = manifold.Isomap(n_components=_n_components, n_jobs=CPUs_AVAIL, **kwargs)
    elif algorithm.lower() == 'spectral_embedding':
        reducer = manifold.SpectralEmbedding(n_components=_n_components, random_state=SEED, n_jobs=CPUs_AVAIL, **kwargs)
    elif algorithm.lower() == 'lle':
        reducer = manifold.LocallyLinearEmbedding(n_components=_n_components, random_state=SEED, n_jobs=CPUs_AVAIL,
                                                  **kwargs)
    elif algorithm.lower() == 'mds':
        reducer = manifold.MDS(n_components=_n_components, random_state=SEED, n_jobs=CPUs_AVAIL, **kwargs)

    embedding = reducer.fit_transform(X)
    embedding = pd.DataFrame(data=embedding)

    embedding = pd.concat([embedding, y], axis=1)
    # Old version: [embedding.reset_index().drop(['index'], axis=1), y.reset_index().drop(['index'], axis=1)]

    print("The {} output has {} shape.".format(algorithm, embedding.shape))
    pd.DataFrame(embedding).to_pickle('{}/{}_{}.pkl'.format(_output_path, _output_name, algorithm))

    if algorithm.lower() == 'pca':
        labels = {str(i): f"PC {i + 1} ({var:.1f}%)" for i, var in enumerate(reducer.explained_variance_ratio_ * 100)}
    else:
        labels = None

    __plot(embedding, _n_components, _typ, _cmap, __lab, method=algorithm, _labels=labels)
    return 0


def exploratory_analysis(ds=None, logs=False, normalize=True, balance=False, __n_components=2,
                         __typ='2d', corr=True, __cor_tresh=0.95, _pca=True, _umap=True, _tsne=True, _isomap=True,
                         _smacof=True, _spectral_embedding=True, _lle=True, _mds=True, feature_list=None,
                         profile=False):
    def __especial_pp(_df, _lab):
        _cmap = cm.get_cmap('tab20', 100).reversed()
        if ds == 'car':
            # _df = _df[_df['Cylinders'].isin([4, 8])]
            _df = _df

        if ds == 'segment':
            """
            _df = _df[_df['ClusterID'].isin([4, 5, 6, 7])]
            # df = df[['rawgreen-mean','rawred-mean','rawblue-mean','ClusterID','class']]
            plot3D(_df, 'exgreen-mean', 'exblue-mean', 'exblue-mean', 'ClusterID')
            plot3D_interactive(_df, 'exgreen-mean', 'exblue-mean', 'exblue-mean', 'ClusterID')

            corrMatrix = _df.corr()
            colormap = plt.cm.magma
            plt.figure(figsize=(16, 12))
            plt.title('Pearson correlation of continuous features', y=1.05, size=15)
            sns.heatmap(corrMatrix, linewidths=0.1, vmax=1.0, square=True,
                        cmap=colormap, linecolor='white', annot=True)
            plt.show()
            """
            pass

        if ds == 'earthquakes':
            """
            corrMatrix = _df.corr()
            colormap = plt.cm.magma
            plt.figure(figsize=(16, 12))
            plt.title('Pearson correlation of continuous features', y=1.05, size=15)
            sns.heatmap(corrMatrix, linewidths=0.1, vmax=1.0, square=True,
                        cmap=colormap, linecolor='white', annot=True)
            plt.show()
            """
            pass

        if ds == 'wine':
            # _df = _df[_df['ClusterID'].isin([1, 3])]
            _df = _df

        if ds == 'Hdbscan':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.scatter('x', 'y', data=df)
            plt.show()

        if ds == 'sequoia':
            label_name = '_class'
            _df['rig'] = _df._class.str.split().str.get(0)
            top_area = _df[label_name].value_counts().head(10).index.to_list()
            _df = _df[_df[label_name].isin(top_area)]
            _df['_class'], _ = _df['_class'].factorize()
            _df['rig'], _ = _df['rig'].factorize()

        if ds == 'rayan_ames':
            label_name = ['SaleCondition']
            num_to_cat = ['YearBuilt', 'YearRemodAdd', 'YearSold']
            _df = num2cat(_df, num_to_cat)
            _num_col = cat_num_columns(_df)['num']
            _cat_col = cat_num_columns(_df)['cat']
            _num_col = _num_col + label_name
            _df = _df.loc[:, _num_col]
            print(_num_col)
            print(_cat_col)

        if ds == 'rayan_auto':
            label_name = [_lab]
            num_to_cat = ['drive_wheels', 'num_of_doors', 'num_of_cylinders', 'peak_rpm']
            _df = num2cat(_df, num_to_cat)
            _num_col = cat_num_columns(_df)['num']
            _cat_col = cat_num_columns(_df)['cat']
            _num_col = _num_col + label_name

            _df = _df.loc[:, _num_col]
            print(_num_col)
            print(_cat_col)

        if ds == 'rayan_cust':
            label_name = [_lab]
            num_to_cat = ["ID", 'MonthsSinceLastClaim', 'MonthsSincePolicyInception', 'NumberofOpenComplaints',
                          'NumberofPolicies']
            _df = num2cat(_df, num_to_cat)
            _num_col = cat_num_columns(_df)['num']
            _cat_col = cat_num_columns(_df)['cat']
            _num_col = _num_col + label_name

            _df = _df.loc[:, _num_col]
            print(_num_col)
            print(_cat_col)

        if ds == 'wifi':
            num_to_cat = ['class']
            _df = num2cat(_df, num_to_cat)

        if ds == 'pan_cancer':
            n = 20530  # 20530
            label = _df['Class']
            _df = _df.iloc[:, :n]
            _df['Class'] = label

        if ds == 'breast-cancer':
            _label_map = {"M": "Malignant", "B": "Benign"}
            _df[_lab].replace(_label_map, inplace=True)

        if ds == 'Covid19':
            _df = _df.set_index('ID').T

        if ds == 'Gromacs':
            # Get silhouette_score
            feature_list = ['L1_Rat', 'n_PAPI_TOT_INS']
            lable = 'ClusterID'
            silhouette_avg = silhouette_score(df[feature_list], df[lable])
            print(silhouette_avg)

        return _df

    with open('../cluster_inference_system/empirical_report/FSCM_paper_dataset/datalist.json') as json_data:
        dataset_dict = json.load(json_data)

    data_dict = dataset_dict.get(ds)
    __lab = str(data_dict.get('labels'))
    df = pd.read_csv(data_dict.get('path'))

    if dataset_dict.get('cat2int'):
        df[__lab] = cat2int(df[__lab])

    print(BColors.OKGREEN + 'The existing original features: \n {}'.format(list(df.columns)) + BColors.ENDC)

    df = __especial_pp(df, __lab)
    if data_dict.get('iglist'):
        df.drop(data_dict.get('iglist'), inplace=True, axis=1)

    if logs:
        df = log_transfer(df, __lab)

    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    if balance:
        df = balance_analysis(df, __lab, major=False)

    if feature_list:  # overall scatter plot
        scatter_plot(df=df, _x=feature_list[0], _y=feature_list[1], _c=__lab, _s=40,
                     _show_grid=True, _font=16, _title='Scatter Plot of {}'.format(ds))

    if __lab:
        label = df[__lab]
        print(label.value_counts(dropna=False))
        df.drop(__lab, inplace=True, axis=1)
    else:
        label = None

    if normalize:
        scaler = StandardScaler()
        df.iloc[:, :] = scaler.fit_transform(df.to_numpy())

    print(BColors.OKGREEN + 'The final dataset shape is: {}'.format(df.shape) + BColors.ENDC)

    if __lab:
        df = pd.concat([df, label], axis=1)
        if data_dict.get('clusteringsuit'):
            # df = clusters_duration_sort(df, __lab, 'Duration')
            df[__lab] = df[__lab] - 5
            label = df[__lab]
        _color_discrete_map = get_discrete_cmp(df, __lab, color_set=COLOR_LIST)
        _markers = MARKER[0:len(_color_discrete_map)]
        df = df.astype({str(__lab): str})

    if profile:
        output_file = 'exploratory_analysis_report.html'
        profile = ProfileReport(df=df, title='KAVICA Exploratory Analysis', html={'style': {'full_width': True}})
        profile.to_file(output_file="{}/{}".format(OUTPUT_PATH, output_file))
        try:
            webbrowser.get('firefox').open_new_tab("{}/{}".format(OUTPUT_PATH, output_file))
        except (RuntimeError, TypeError, NameError):
            webbrowser.get('firefox').open_new("{}/{}".format(OUTPUT_PATH, output_file))

    if _pca:
        factorize('pca', df, label, _n_components=__n_components, _typ=__typ, _output_name=ds, _norm=True,
                  _output_path=OUTPUT_PATH, _cmap=_color_discrete_map)
    if _umap:
        factorize('umap', df, label, _output_name=ds, _norm=False, _output_path=OUTPUT_PATH,
                  _cmap=_color_discrete_map, _typ=__typ, _n_components=__n_components,
                  n_neighbors=20, min_dist=0.1)
    if _tsne:
        factorize('tsne', df, label, _n_components=__n_components, _output_name=ds, _norm=False,
                  _output_path=OUTPUT_PATH, _cmap=_color_discrete_map, _typ=__typ)
    if _isomap:
        factorize('isomap', df, label, _n_components=__n_components, _output_name=ds, _norm=False,
                  _output_path=OUTPUT_PATH, _cmap=_color_discrete_map, _typ=__typ, n_neighbors=30)
    if _spectral_embedding:
        factorize('spectral_embedding', df, label, _n_components=__n_components, _output_name=ds, _norm=False,
                  _output_path=OUTPUT_PATH, _cmap=_color_discrete_map, _typ=__typ, affinity='nearest_neighbors',
                  n_neighbors=30)
    if _lle:
        factorize('lle', df, label, _n_components=__n_components, _output_name=ds, _norm=False,
                  _output_path=OUTPUT_PATH, _cmap=_color_discrete_map, _typ=__typ, method="modified")
    if _mds:
        factorize('mds', df, label, _n_components=__n_components, _output_name=ds, _norm=False,
                  _output_path=OUTPUT_PATH, _cmap=_color_discrete_map, _typ=__typ)

    if corr:
        cor_matrix = df.corr().abs()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > __cor_tresh)]
        df.drop(to_drop, axis=1, inplace=True)
        cor_matrix = df.corr().abs()
        fig = px.imshow(cor_matrix, width=800, height=600)
        fig.show()

    if __lab:
        try:
            fig = px.scatter_matrix(df, color_discrete_map=_color_discrete_map,
                                    width=1600, height=1200,
                                    color=__lab)

        except ValueError:
            fig = px.scatter_matrix(df, color=__lab,
                                    color_discrete_map=_color_discrete_map,
                                    width=1600, height=1200)
    else:
        fig = px.scatter_matrix(df, width=1600, height=1200)
    fig.update_traces(marker=dict(size=2, line=dict(width=0)), diagonal_visible=False, showupperhalf=False)
    fig.show()
    plt.show()


if __name__ == '__main__':
    # cluster_analysis('{}/Gromacs_umap.pkl'.format(OUTPUT_PATH),2,16,_lab='ClusterID')

    # exploratory_analysis(ds='OFF_EVENTES', logs=False, normalize=True, balance=False, __n_components=2, __typ='2d')

    exploratory_analysis(ds='Gromacs', logs=False, normalize=True, balance=False, __n_components=2, __typ='2d',
                         feature_list=None,  # ['L1_Rat', 'n_PAPI_TOT_INS'],
                         _pca=True, _umap=True, _tsne=False, _isomap=False, _spectral_embedding=False,
                         _lle=False, _mds=False)
