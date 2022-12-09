#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate samples of synthetic data sets.
"""
import numpy as np
from matplotlib import cm
from itertools import count
import sklearn.datasets
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

__all__ = [
    'make_dataset'
]


def make_dataset(make="blobs", x_axis=0, y_axis=1, _plot=True, _plot_labels=True,
                 _annotated=False, **kwargs):
    """Generate synthetic sample data

    Args:
        make (str): The synthetic data structure
        _plot (boolean): Show the scatter plot
        _plot_labels (boolean): Apply the labels color in scatter plot
        x_axis (int): The (index of) feature is plotted in x axis
        y_axis (int): The (index of) feature is plotted in y axis
        _annotated (boolean): Show the cluster size and std inside the plot area
        **kwargs:

    Returns:

    Note:
        https://www.kite.com/python/docs/sklearn.datasets.make_classification

    """
    if make == 'blobs':
        X, label = sklearn.datasets.make_blobs(**kwargs)
        X = pd.DataFrame(X)
        label = pd.Series(label)
        _n_samples = kwargs.get('n_samples')
        _centroids = kwargs.get('centers')
        _stds = kwargs.get('cluster_std')
        _legend_header = "Size/Mean/std"
        label_text = []
        for label_item, cl_size, cl_std, cl_cent in zip(count(), _n_samples, _stds, _centroids):
            label_text.append('C{}:{}/{}/{}'.format(label_item, cl_size, cl_cent, cl_std))
    elif make == 'circles':
        X, label = sklearn.datasets.make_circles(**kwargs)
        X = pd.DataFrame(X)
        label = pd.Series(label)
        _n_samples = kwargs.get('n_samples')
        _legend_header = "Size"
        label_text = []
        for label_item, cl_size in zip(count(), _n_samples):
            label_text.append('C{}: {}'.format(label_item, cl_size))
    elif make == 'moons':
        X, label = sklearn.datasets.make_moons(**kwargs)
        X = pd.DataFrame(X)
        label = pd.Series(label)
        _legend_header = ""
        label_text = ['C0', 'C1']
    elif make == 'aniso':
        X, label = sklearn.datasets.make_blobs(**kwargs)
        # Changes how x1, x2 coordinates are shifted
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
        X = pd.DataFrame(X)
        label = pd.Series(label)
        _n_samples = kwargs.get('n_samples')
        _centroids = kwargs.get('centers')
        _stds = kwargs.get('cluster_std')
        _legend_header = "Size & std"
        label_text = []
        for label_item, cl_size, cl_std in zip(count(), _n_samples, _stds):
            label_text.append('C{}: {} / {}'.format(label_item, cl_size, cl_std))
    elif make == 'class':
        X, label = sklearn.datasets.make_classification(**kwargs)
        X = pd.DataFrame(X)
        label = pd.Series(label)
        _legend_header = ""
        label_text = []
        for label_item, _ in zip(count(), list(set(label))):
            label_text.append('C{}'.format(label_item))
    elif make == 'quan':
        X = None
        label_text = []
        _legend_header = ""
        for key, value in kwargs.get('conf').items():
            if X is None:
                X, label = sklearn.datasets.make_gaussian_quantiles(**value)
            else:
                X_temp, label_temp = sklearn.datasets.make_gaussian_quantiles(**value)
                X = np.append(X, X_temp, axis=0)
                label_temp[label_temp == 0] = key
                label = np.append(label, label_temp, axis=0)
            label_text.append('C{}'.format(key))
        X = pd.DataFrame(X)
        label = pd.Series(label)
    if _plot:
        _cmap = cm.get_cmap('tab20', 20).reversed()
        fig, ax = plt.subplots(figsize=(16, 16), dpi=100)
        if not _plot_labels:
            scatter = plt.scatter(X.loc[:, x_axis], X.loc[:, y_axis], marker='o', c='lime',
                                  s=30, edgecolor='k')
        else:
            scatter = plt.scatter(X.loc[:, x_axis], X.loc[:, y_axis], marker='o', c=label,
                                  cmap=_cmap, s=30, edgecolor='k')

        if _annotated:
            for cl_size, cl_centroid, cl_std in zip(_n_samples, _centroids, _stds):
                cl_centroid = (cl_centroid[0], cl_centroid[1] - 6)
                ax.annotate("S:{}\nstd{}".format(cl_size, cl_std), xy=cl_centroid,
                            ha="center", va="center", size=10,
                            bbox=dict(boxstyle="Square,pad=0.3", fc="yellow"))
        # make the legend
        handles, legend_labels = scatter.legend_elements()
        legend_labels = map(lambda label_item: str('$\\mathdefault{' + label_item + '}$'), label_text)
        legend = ax.legend(handles, legend_labels, loc="lower left", title=_legend_header)
        ax.add_artist(legend)
        ax.margins(0.15)
        plt.xticks(rotation=90)
        plt.title("Synthetic sample data (with original labeling)")
        # ax.xaxis.set_major_locator(MultipleLocator(1))
        # ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_minor_locator(AutoMinorLocator(1))
        plt.grid()
        plt.show()
    return X, label
