#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Utility functions

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
First update: 07/10/2022
"""
from kavica.utils import BColors
from sklearn.preprocessing import normalize
import pandas as pd
import copy

__all__ = [
    "cat2int",
    "map_index_level",
    "normalize_pandas",
    "record_log",
    "cat_num_columns",
    "num2cat"
]


def cat2int(_cat):
    """ Map the categorical labels to the int

    Returns:
        A dict of the mapping function
    """
    label_map = dict(enumerate(list(set(_cat))))
    label_map = {value: key for key, value in label_map.items()}
    _cat = _cat.map(label_map)
    print(BColors.WARNING + 'The labels are mapped through: {}'.format(label_map) + BColors.ENDC)
    return _cat


def cat2int_map(_cat, _map=True):
    """ Maps the categorical labels to the int

    Returns:
        A dict of the mapping function
    """
    label_map = dict(enumerate(list(set(_cat))))
    label_map = {value: key for key, value in label_map.items()}
    print(BColors.WARNING + 'The labels are mapped through: {}'.format(label_map) + BColors.ENDC)
    return label_map


def map_index_level(index, mapper, level=0):
    """
    Returns a new Index or MultiIndex, with the level values being mapped.

    Args:
        index (pandas.index): It is a df.index or df.columns
        mapper (dict or function): Mapping correspondence
        level (int, default=0): It is the integer position of the level.
    """
    assert (isinstance(index, pd.Index))
    if isinstance(index, pd.MultiIndex):
        new_level = index.levels[level].map(mapper)
        new_index = index.set_levels(new_level, level=level)
    else:
        # Single level index.
        assert (level == 0)
        new_index = index.map(mapper)
    return new_index


def normalize_pandas(df, _norm='l1', _axis=0):
    """ Compute the normalized pandas data frame

    Args:
        df (pandas): The input data
        _norm (str): Indicates the norm L1 , L2, or Max
        _axis (0 or 1, optional (1 by default)): axis used to normalize the data along. If 1, independently normalize
        each sample, otherwise (if 0) normalize each feature.

    Returns:

    """
    df_normalized = normalize(df, norm=_norm, axis=_axis)
    return pd.DataFrame(df_normalized, columns=df.columns, index=df.index)


def record_log(color=None, tag=None, text=None, *args):
    """ Print/Write a log line.

    Args:
        text (str): The log massage or information.
        tag (str): The module or type of the information.
        color (BColors): Indicates the line color.

    Returns:

    """
    print(color + BColors.BOLD + tag.upper() + " :: " + BColors.ENDC + color + text.format(*args) + BColors.ENDC)


def cat_num_columns(_df):
    """ Reports the list of categorical and numerical colons

    Args:
        _df (pandas): Includes the input data

    Returns:
        A dict {'cat': [] , 'num':[]}
    """
    cat = _df.select_dtypes(include=['object']).columns.tolist()
    num = _df.select_dtypes(exclude=['object']).columns.tolist()
    return {'cat': cat, 'num': num}


def num2cat(df, num_list):
    """ Changes the datatype of the columns

    Args:
        df:
        num_list:

    Returns:

    """
    _df = copy.deepcopy(df)
    for col_item in num_list:
        _df[col_item] = _df[col_item].astype('object')
    return _df
