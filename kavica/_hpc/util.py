#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Utility functions for the traces and HPC data

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
First update: 09/11/2022
"""
import pandas as pd
import numpy as np

__all__ = [
    'clusters_duration_sort'
]


def clusters_duration_sort(df, label, due='Duration', drop_due=True):
    """ Resorts the colores to uniform it with Paraver

    Args:
        df (pandas): Includes the dataset
        label (str): Indicates the class labels
        due (str)

    Returns:
        A pandas with resorted colors

    Note:
        The duration feature has to be in the list
    """
    _agr_duration = df.groupby(label).sum()
    if _agr_duration.index[0] == 5:
        _agr_duration = _agr_duration[1:]
    __sorted_color = np.flip(_agr_duration[due].argsort()).reset_index(drop=True)
    __sorted_color = __sorted_color.to_dict()
    __sorted_color = dict([(value, key) for key, value in __sorted_color.items()])
    df[label].replace(__sorted_color, inplace=True)
    if drop_due:
        df.drop(due, inplace=True, axis=1)
    return df
