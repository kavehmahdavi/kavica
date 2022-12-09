#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Plots

It includes all most of the general plotting functions

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
First update: 08/10/2022
"""
import copy
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib import cm
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap, BoundaryNorm, to_hex, to_rgba
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
import os
import plotly.express as px
import matplotlib

__all__ = [
    'density_plot_3d',
    'density_plot_2d',
    'density_plotly_3d',
    'PARAVER_STATES_COLOR',
    'MARKER',
    'get_color_map',
    'get_color_palette',
    'palette_size',
    'get_listed_cmap',
    'scatter_plot'
]

PARAVER_STATES_COLOR = {5: [179, 0, 0],
                        6: [0, 255, 0],
                        7: [255, 255, 0],
                        8: [235, 0, 0],
                        9: [0, 162, 0],
                        10: [255, 0, 255],
                        11: [100, 100, 177],
                        12: [172, 174, 41],
                        13: [255, 144, 26],
                        14: [2, 255, 177],
                        15: [192, 224, 0],
                        16: [66, 66, 66],
                        17: [189, 168, 100],
                        18: [95, 200, 0],
                        19: [203, 60, 69],
                        20: [0, 109, 255],
                        21: [200, 61, 68],
                        22: [200, 66, 0],
                        23: [0, 41, 0],
                        24: [139, 121, 177],
                        25: [116, 116, 116],
                        26: [200, 50, 89],
                        27: [255, 171, 98],
                        28: [0, 68, 189],
                        29: [52, 43, 0],
                        30: [255, 46, 0],
                        31: [100, 216, 32],
                        32: [0, 0, 112],
                        33: [105, 105, 0],
                        34: [132, 75, 255],
                        35: [184, 232, 0],
                        36: [0, 109, 112],
                        37: [189, 168, 100],
                        38: [132, 75, 75],
                        39: [255, 75, 75],
                        40: [255, 20, 0],
                        41: [52, 0, 0],
                        42: [0, 66, 0],
                        43: [184, 132, 0],
                        44: [100, 16, 32],
                        45: [146, 255, 255],
                        46: [179, 0, 0],
                        47: [0, 255, 0],
                        48: [255, 255, 0],
                        49: [235, 0, 0],
                        50: [0, 162, 0],
                        51: [255, 0, 255],
                        52: [100, 100, 177],
                        53: [172, 174, 41],
                        54: [255, 144, 26],
                        55: [2, 255, 177],
                        56: [192, 224, 0],
                        57: [66, 66, 66],
                        58: [189, 168, 100],
                        59: [95, 200, 0],
                        60: [203, 60, 69],
                        61: [0, 109, 255],
                        62: [200, 61, 68],
                        63: [200, 66, 0],
                        64: [0, 41, 0],
                        65: [139, 121, 177],
                        66: [116, 116, 116],
                        67: [200, 50, 89],
                        68: [255, 171, 98],
                        69: [0, 68, 189],
                        70: [52, 43, 0],
                        71: [255, 46, 0],
                        72: [100, 216, 32],
                        73: [0, 0, 112],
                        74: [105, 105, 0],
                        75: [132, 75, 255],
                        76: [184, 232, 0],
                        77: [0, 109, 112],
                        78: [189, 168, 100],
                        79: [132, 75, 75],
                        80: [255, 75, 75],
                        81: [255, 20, 0],
                        82: [52, 0, 0],
                        83: [0, 66, 0],
                        84: [184, 132, 0],
                        85: [100, 16, 32],
                        86: [146, 255, 255],
                        87: [146, 255, 255]
                        }

MARKER = ['+', 'h', 'H', 'x', 'X', 'D',
          'd', '.', ',', 'o', 'v', '^',
          '<', '>', '1', '2', '3', '4',
          '8', 's', 'p', 'P', '*', '|',
          'd', '.', ',', 'o', 'v', '^',
          '<', '>', '1', '2', '3', '4',
          '+', 'h', 'H', 'x', 'X', 'D',
          'd', '.', ',', 'o', 'v', '^',
          '+', 'h', 'H', 'x', 'X', 'D',
          'd', '.', ',', 'o', 'v', '^',
          '<', '>', '1', '2', '3', '4',
          '8', 's', 'p', 'P', '*', '|',
          'd', '.', ',', 'o', 'v', '^',
          '<', '>', '1', '2', '3', '4',
          '+', 'h', 'H', 'x', 'X', 'D',
          'd', '.', ',', 'o', 'v', '^']

matplotlib.use('module://backend_interagg')


def get_discrete_cmp(df, lab, color_set=None):
    """ Generates a discrete color map

    Args:
        df (pandas): indicate the dataset
        lab (str): The cluster name
        color_set (list): a list of overall colores

    Returns:
        A dict includes the color map
    """
    _df = copy.deepcopy(df)
    __cluster_label_set = list(map(str, list(set(_df[lab]))))
    _df = _df.astype({str(lab): str})
    _color_discrete_map = dict(zip(__cluster_label_set, color_set[0:len(__cluster_label_set) + 1]))
    return _color_discrete_map


def get_listed_cmap(labels, color_states=PARAVER_STATES_COLOR):
    """ Generates a listed (discrete) color map

    Args:
        labels (numpy / pandas series): Includes the categorical data (labels)
        color_states (dict): Includes the main set of the colores

    Returns:
        A cmap
    """
    _palette = get_color_palette(color_states, None, True)
    if -1 in labels:
        _palette = _palette[0:palette_size(labels)]
    else:
        _palette = _palette[1:palette_size(labels) + 1]
    return mcolors.ListedColormap(_palette)


def get_color_map(c, _hex=False):
    """ Create customized color map
    Arge:
        c(dict): A dict includes the all colors
    Returns:
        A plt color map
    """
    __color_code = np.array(list(c.values())) / 255
    if not _hex:
        return __color_code
    else:
        __color_map = list(map(to_hex, __color_code))
        return __color_map


def get_color_palette(c, size, _hex=True):
    """ Compute a color palette

    Args:
        c(dict): A dict includes the all colors
        size (int): Indicates the size of the palette
        _hex (boolean): If True, it returns the hexadecimal value of the colors

    Returns:
        A list of the colores
    """
    if size is None:
        return get_color_map(c, _hex)
    elif size > len(c):
        raise ValueError("Error: There is not enough colors in the list")
    else:
        return get_color_map(c, _hex)[0:size]


def palette_size(cat):
    """ Computes the palett size

    Args:
        cat (pandas series): Indicates a categorical pandas series

    Returns:
        An int indicates the size of unique set of the categorical series
    """
    if isinstance(cat, np.ndarray):
        return len(np.unique(cat))
    elif isinstance(cat, list):
        return len(list(set(cat)))
    else:
        return len(cat.unique())


def density_plot_3d(df, x_lab='x', y_lab='y', _bw=0.08):
    """ Draws a 3D histogram

    Args:
        df (pandas): includes data
        x_lab (str): indicates x features
        y_lab (str): indicates y feature
        _bw (float): indicate the bandwidth

    Returns:
    """
    # Extract x and y
    x = df[x_lab]
    y = df[y_lab]

    # Define the borders
    deltaX = (max(x) - min(x)) / 5
    deltaY = (max(y) - min(y)) / 5
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY

    # Create mesh-grid
    xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]

    # We will fit a gaussian kernel using the scipy’s gaussian_kde method
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    if not _bw:
        kernel = st.gaussian_kde(values)
    else:
        kernel = st.gaussian_kde(values, bw_method=_bw)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(22, 14))
    ax = plt.axes(projection='3d')

    surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap=cm.coolwarm, edgecolor='k')
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('PDF', fontsize=20, rotation=90)
    fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='z', labelsize=16)
    ax.view_init(60, -115)
    plt.show()


def density_plot_2d(df, x_lab='x', y_lab='y'):
    """ Draws an interactive 2D histogram

    Args:
        df (pandas): includes data
        x_lab (str): indicates x features
        y_lab (str): indicates y feature

    Returns:
    """
    # Extract x and y
    x = df[x_lab]
    y = df[y_lab]

    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
        x=x,
        y=y,
        colorscale='Blues',
        reversescale=True,
        xaxis='x',
        yaxis='y'
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        xaxis='x',
        yaxis='y',
        mode='markers',
        marker=dict(
            color='rgba(0,0,0,0.3)',
            size=3
        )
    ))
    fig.add_trace(go.Histogram(
        y=y,
        xaxis='x2',
        marker=dict(
            color='rgba(0,0,0,1)'
        )
    ))
    fig.add_trace(go.Histogram(
        x=x,
        yaxis='y2',
        marker=dict(
            color='rgba(0,0,0,1)'
        )
    ))

    fig.update_layout(
        autosize=False,
        xaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showgrid=False
        ),
        yaxis=dict(
            zeroline=False,
            domain=[0, 0.85],
            showgrid=False
        ),
        xaxis2=dict(
            zeroline=False,
            domain=[0.85, 1],
            showgrid=False
        ),
        yaxis2=dict(
            zeroline=False,
            domain=[0.85, 1],
            showgrid=False
        ),
        height=600,
        width=600,
        bargap=0,
        hovermode='closest',
        showlegend=False
    )

    fig.show()


def density_plotly_3d(df, x_lab='x', y_lab='y'):
    """ Draws an interactive 3D histogram

    Args:
        df (pandas): includes data
        x_lab (str): indicates x features
        y_lab (str): indicates y feature

    Returns:
    """
    # Extract x and y
    x = df[x_lab]
    y = df[y_lab]
    # Define the borders
    deltaX = (max(x) - min(x)) / 10
    deltaY = (max(y) - min(y)) / 10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY

    # Create mesh-grid
    xx, yy = np.mgrid[xmin:xmax:600j, ymin:ymax:600j]

    # We will fit a gaussian kernel using the scipy’s gaussian_kde method
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values, bw_method=0.1)
    f = np.reshape(kernel(positions).T, xx.shape)
    fig = go.Figure(data=[go.Surface(x=xx, y=yy, z=f)])
    fig.show()


def scatter_plot(df, _x='X', _y='Y', _c=None, _s=40, _drop_nan=True, _title='Scatter plot', _show_grid=True, _font=16,
                 _clusteringsuit=True):
    """ Adds a scatter plot as layer to a plot

    Args:
        df (ndarray): The dataset
        _x (str): The x-axis feature name
        _y (str): The y-axis feature name
        _c (str): A list of the labels or a feature name
        _drop_nan (boolean): If True It drops the nans
        _s (int): Indicates the data points size in scatter plot
        _title (str): Indicates the plot title
        _show_grid (boolean): If True It draws grid
        _font (int): Indicates the legend font
        _clusteringsuit (boolean): If True, the data was generated by clusteringsuit (toolsbsc.es)

    Returns:
    """

    if isinstance(df, str) and os.path.exists(df):
        df = pd.read_csv(df)
        print(df.describe())

    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    if isinstance(_c, str):
        df['lab'] = df[_c]
    else:
        df['lab'] = _c
        _c = 'lab'

    if _drop_nan:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

    unique_labels = set(df[_c].to_list())

    reduction_index = 5 if _clusteringsuit else -1

    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    color_list = get_color_palette(PARAVER_STATES_COLOR, None, True)

    for cluster_item in list(unique_labels):
        cluster_item = cluster_item
        cluster_item_df = df[df[_c] == cluster_item]

        if (cluster_item - reduction_index) == 0 or _clusteringsuit:
            _label = 'Noise'
        else:
            _label = 'Cluster_{}'.format(cluster_item - reduction_index)

        ax.scatter(cluster_item_df[[_x]], cluster_item_df[[_y]],
                   c=color_list[cluster_item - reduction_index],
                   s=_s,
                   marker=MARKER[cluster_item - reduction_index],
                   label=_label)

    ax.margins(0.15)
    plt.xticks(rotation=90)
    font_p = FontProperties().set_size('small')
    ax.legend(prop=font_p)
    if _show_grid:
        ax.grid(linestyle='--', which='major')

    plt.xlabel(_x)
    plt.ylabel(_y)
    plt.title(_title)

    ax.tick_params(axis='x', labelsize=_font)
    ax.tick_params(axis='y', labelsize=_font)
    plt.show()
    return fig


def plot3D(df, x, y, z, c):
    """

    Args:
        df:
        x:
        y:
        z:
        c:

    Returns:

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x], df[y], df[z], c=df[c], marker='+')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    plt.show()


def plot3D_interactive(df, x, y, z, c):
    """

    Args:
        df:
        x:
        y:
        z:
        c:

    Returns:

    """
    fig = px.scatter_3d(df, x, y, z, c)
    fig.update_traces(mode='markers', marker_size=1)
    fig.show()


def __test_me():
    __path = '../ECII/data/PhD_thesis/gromacs_64p.chop1.cl.IPCxINS.DATA.ECII.csv'
    scatter_plot(df=__path, _x='L1_Rat', _y='n_PAPI_TOT_INS', _c='ClusterID', _s=30, _show_grid=True, _font=16,
                 _title='Scatter Plot of Gromacs_64p_FSCM')

    __path = '../../data/row2prv_data/4kaveh_jesus/RD_BW_in_long_burst@lulesh_4+4_with_uncores_sockets1+2' \
             '.chop_5it_final.csv'
    scatter_plot(df=__path, _x='RD_BW', _y='WR_BW', _c='ClusterID_T', _s=30, _show_grid=True, _font=16,
                 _title='Scatter Plot of lulesh4x4', _clusteringsuit=False)


if __name__ == '__main__':
    __test_me()
