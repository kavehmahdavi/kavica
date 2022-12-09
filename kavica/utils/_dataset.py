#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Import and handling the datasets


Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
last update: 15/11/2022
"""
import pandas as pd
import wget
import os

__all__ = [
    'import_dataset'
]


def import_dataset(data='titanic', url=None, sep=','):
    """Import example dataset from github source.

    Import one of the few datasets from github source or specify your own download url link.

    data (str): Name of datasets: 'sprinkler', 'titanic', 'student', 'fifa', 'cancer', 'waterpump', 'retail'
    url (str): url link to dataset.

    Returns
        A pandas containing mixed features.

    """
    if url is None:
        if data == 'sprinkler':
            url = 'https://erdogant.github.io/datasets/sprinkler.zip'
        elif data == 'titanic':
            url = 'https://erdogant.github.io/datasets/titanic_train.zip'
        elif data == 'student':
            url = 'https://erdogant.github.io/datasets/student_train.zip'
        elif data == 'cancer':
            url = 'https://erdogant.github.io/datasets/cancer_dataset.zip'
        elif data == 'fifa':
            url = 'https://erdogant.github.io/datasets/FIFA_2018.zip'
        elif data == 'waterpump':
            url = 'https://erdogant.github.io/datasets/waterpump/waterpump_test.zip'
        elif data == 'retail':
            url = 'https://erdogant.github.io/datasets/marketing_data_online_retail_small.zip'
            sep = ';'
    else:
        data = wget.filename_from_url(url)

    if url is None:
        print('Nothing to download.')
        return None

    curpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    PATH_TO_DATA = os.path.join(curpath, wget.filename_from_url(url))
    if not os.path.isdir(curpath):
        os.makedirs(curpath, exist_ok=True)

    # Check file exists.
    if not os.path.isfile(PATH_TO_DATA):
        print('Downloading [%s] dataset from github source..' % data)
        wget.download(url, curpath)

    # Import local dataset
    print('Import dataset [%s]' % data)
    df = pd.read_csv(PATH_TO_DATA, sep=sep)
    # Return
    return df
