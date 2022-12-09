#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for the PRV traces
"""
import numpy as np
from matplotlib import cm
from itertools import count
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
from kavica.utils import BColors
from sklearn.preprocessing import MinMaxScaler
from kavica.cluster_inference_system.som import (record_log, get_color_palette, palette_size,
                                                 PARAVER_STATES_COLOR, MARKER)

__all__ = [
    'plot_overall_scatter',
    'load_trace_csv'
]


def plot_overall_scatter(_df, label_id, features=None, log_log=[False, False]):
    """ Plot scatter of all clusters

    Args:
        log_log (boolean list): For any axis in the scatter plot, if it is True, the log transformation will applied.
        features (list): Includes x, y feature in order to plot the data
        _df (pandas):  Includes the data and a cluster_id label
        label_id (list): The data points labels
    Returns:
    """
    # Apply the log transfer
    if any(log_log):
        axis_labels = []
        for axis, log_transfer in enumerate(log_log):
            if log_transfer:
                _df.loc[:, features[axis]] = np.log(_df.loc[:, features[axis]])
                axis_labels.append("Log( {} )".format(features[axis]))
            else:
                axis_labels.append(features[axis])
        _df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, inplace=True)
    else:
        axis_labels = features

    _n_samples = _df.groupby(label_id).size().tolist()
    _stds = _df.groupby(label_id).std().mean(axis=1).tolist()

    label_text = []
    for label_item, cl_size, cl_std in zip(count(), _n_samples, _stds):
        if label_item != 0:
            label_text.append('C{}: {} / {:.3f}'.format(label_item, cl_size, cl_std))
        else:
            label_text.append('Noise: {} / {:.3f}'.format(cl_size, cl_std))

    _cmap = cm.get_cmap('tab20', 50).reversed()
    fig, ax = plt.subplots(figsize=(14, 12), dpi=200)

    for cluster_item in list(set(_df.loc[:, 'ClusterID'])):
        df_item = _df[_df['ClusterID'] == cluster_item]
        cluster_item = cluster_item - 6
        if cluster_item != 0:
            ax.scatter(df_item.loc[:, features[0]],
                       df_item.loc[:, features[1]],
                       s=20,
                       marker=MARKER[cluster_item],
                       label=label_text[cluster_item])
        else:
            ax.scatter(df_item.loc[:, features[0]],
                       df_item.loc[:, features[1]],
                       s=10,
                       marker='+',
                       c='maroon',
                       label=label_text[cluster_item])

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    fontP = FontProperties()
    fontP.set_size('x-small')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=fontP)
    plt.grid(True)
    ax.margins(0.15)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    plt.show()
    plt.close()


def load_trace_csv(_path, x_axis='n_IPC', y_axis='n_PAPI_TOT_INS', label_id='ClusterID', _plot=True,
                   _plot_labels=True, clusters=['6', '7', '8', '9'], featurs=None, _scale=True):
    """ Read data file.

    Args:
        _path (str): indicates the path of the data file (.csv)
        _plot (boolean): Show the scatter plot
        _plot_labels (boolean): Apply the labels color in scatter plot
        x_axis (str): The (index of) feature is plotted in x axis
        y_axis (str): The (index of) feature is plotted in y axis
        label_id (str): The (index of) feature is indicating the data labels
        clusters (list): Indicates the ist of clusters for filtering
    Return:
        A pandas includes the data.

    Note:
        d_PAPI_TOT_INS: original
        n_PAPI_TOT_INS: normalized
        x_PAPI_TOT_INS: extrapolated
    """
    trace_df = pd.read_csv(_path, usecols=featurs)  # Read the data file
    print(trace_df)
    if clusters:
        trace_df = trace_df[trace_df[label_id].isin(clusters)]
        trace_df.reset_index(drop=True, inplace=True)

    trace_df.columns = trace_df.columns.str.strip()  # remove white space at both ends:
    record_log(BColors.OKGREEN, "Preprocess", "The data {} is loaded successfully.", trace_df.shape)
    label = trace_df.loc[:, [label_id]].squeeze(axis=1)
    X = trace_df.drop([label_id], axis='columns')
    if _scale:
        scaler = MinMaxScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    plot_overall_scatter(pd.concat([X, label.reindex(X.index)], axis=1), label_id='ClusterID',
                         features=[x_axis, y_axis])

    return X, label.squeeze()
