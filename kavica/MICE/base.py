#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
It is a method for missing value imputation in data-set.
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/
"""
# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
#
# License: BSD 3 clause
import numpy as np
import pandas as pd
import warnings
from terminaltables import DoubleTable
from scipy.stats.mstats import gmean, hmean
from time import sleep
import itertools
import matplotlib.pyplot as plt
from sklearn import linear_model, discriminant_analysis
import json
import argparse
import sys
import time

__all__ = ['__configoration',
           'data_structure_Compatibilization',
           'MissingValuePreProcessing',
           'Mice'
           ]


# TODO: add arguments parser to the module

# read the configuration file for preparing the features
def __configoration(config, data):
    # read the configuration file
    with open(config, 'r') as config:
        config_dict = json.load(config)

    # Read the data file
    df = pd.read_csv(data)

    columns_order = list(df.columns.values)
    active_features = list(set(list(config_dict['hardware_counters'].values())
                               + list(config_dict['complimentary'].values())))

    pass_through_features = list(set(list(config_dict['pass_through'].values())
                                     + list(config_dict['complimentary'].values())))

    # config the data set based on configuration information
    df = df[active_features]  # sub set of features
    return df, pass_through_features, columns_order


def arguments_parser():
    # set/receive the arguments
    if len(sys.argv) == 1:
        # It is used for testing and developing time.
        arguments = ['config.json',
                     'source2.csv',
                     '-m',
                     'norm',
                     '-o',
                     'imputed.csv'
                     ]
        sys.argv.extend(arguments)
    else:
        pass

    # parse the arguments
    parser = argparse.ArgumentParser(description='The files that are needed for selecting features most important.')
    parser.add_argument('config', help='A .json configuration file that included the'
                                       'thread numbers,hardware counters and etc.')
    parser.add_argument('csvfile', help='A .csv dataset file')

    # MICE prediction method
    parser.add_argument('-m',
                        dest='m',
                        default='norm',
                        choices=['norm', 'norm.nob', 'lda', 'qda', 'polyreg', 'logreg'],
                        action='store',
                        type=str.lower,
                        help="The imputation method that is either norm, norm.nob, lda, qda, polyreg, logreg.")

    parser.add_argument('-i',
                        dest='i',
                        default=10,
                        action='store',
                        type=int,
                        help="It significances the number of the MICE algorithm iteration.")

    parser.add_argument('-o',
                        dest='o',
                        default='imputed.csv',
                        action='store',
                        type=str,
                        help="path to custom root results directory")

    args = parser.parse_args()

    return ({"configPath": args.config,
             "csvPath": args.csvfile,
             "predict_method": args.m,
             "imputedPath": args.o,
             "iteration": args.i})


def data_structure_Compatibilization(data=None, header=True, index=True):
    if data is None:
        raise ValueError("The data set is empty")

    # Convert to dataframe
    def __numpy2panda(data, header, index):
        # not empty data set
        def __datashape(data):
            if len(data.shape) is not 2:  # Check the shape
                raise ValueError("Expected 2d matrix, got %s array" % (data.shape,))
            elif data.empty:
                raise ValueError("Not expected empty data set.")
            else:
                print("2d matrix is gotten %s array" % (data.shape,))

        if type(data) is not pd.core.frame.DataFrame:
            if header:
                if index:
                    dataFrame = pd.DataFrame(data=data[1:, 1:],  # values
                                             index=data[1:, 0],  # 1st column as index
                                             columns=data[0, 1:])
                else:
                    dataFrame = pd.DataFrame(data=data[1:, 0:],  # values
                                             columns=data[0, 0:])
            elif index:
                dataFrame = pd.DataFrame(data=data[0:, 1:],  # values
                                         index=data[0:, 0])  # 1st column as index)
            else:
                dataFrame = pd.DataFrame(data=data)
        else:
            dataFrame = data
        __datashape(dataFrame)
        return dataFrame.apply(pd.to_numeric)

    return __numpy2panda(data, header, index)
