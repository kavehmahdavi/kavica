#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
It is a method for missing value imputation in data-set.

Note:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/

Author: Kaveh Mahdavi <kavehmahdavi74@yahoo.com>
License: BSD 3 clause
last update: 23/09/2022
"""

import pandas as pd
import json
import argparse
import sys

__all__ = ['__config',
           'compatible_data_structure',
           ]


# TODO: add the utility functions to this module from other modules in this subpackage

def __config(config_path, _data_path):
    """ Read the configuration file (.json)

    In order to impute a file, we need to indicate the main features, pass_through and the complementaries.

    Args:
        config_path (str): indicates the path of the configuration file.
        _data_path (str): indicates the path of the data file (.csv)

    Return:
        df (pandas data frame): Represents the data set where feature subset is {complimentary, hardware conters}
        pass_through_features (list):includes of all feature that are not used in MICE, and they will stake to output.
        columns_order (list): It is a order (original order) list of the feature in data set.
    """
    with open(config_path, 'r') as config_path:
        config_dict = json.load(config_path)

    df = pd.read_csv(_data_path)

    columns_order = list(df.columns.values)
    active_features = list(set(list(config_dict['hardware_counters'].values())
                               + list(config_dict['complimentary'].values())))

    pass_through_features = list(set(list(config_dict['pass_through'].values())
                                     + list(config_dict['complimentary'].values())))

    df = df[active_features]  # sub set of features
    return df, pass_through_features, columns_order


def arguments_parser():
    """ Parse the arguments

    Return:
        A dict includes {"configPath", "csvPath", "predict_method", "imputedPath", "iteration"}
    """
    # set the arguments
    if len(sys.argv) == 1:
        # It is used for testing and developing time.
        arguments = ['config_lulesh_27p.json',
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


def compatible_data_structure(data=None, header=True, index=True):
    """ Reconstruct/ uniformize the input data as pandas data frame

    Args:
        data (a Numpy array or pandas data frame): is the data set
        header (boolean): if True, the first row of the data includes the header.
        index (boolean): if True, the first column of the data includes the index.

    Return:
        A pandas data frame.
    """
    if data is None:
        raise ValueError("The data set is empty")

    # Convert to dataframe
    def __numpy2panda(_data, _header, _index):
        # not empty data set
        def __datashape(__data):
            if len(__data.shape) is not 2:  # Check the shape
                raise ValueError("Expected 2d matrix, got %s array" % (__data.shape,))
            elif __data.empty:
                raise ValueError("Not expected empty data set.")
            else:
                print("2d matrix is gotten %s array" % (__data.shape,))

        if type(_data) is not pd.core.frame.DataFrame:
            if _header:
                if _index:
                    dataFrame = pd.DataFrame(data=_data[1:, 1:],  # values
                                             index=_data[1:, 0],  # 1st column as index
                                             columns=_data[0, 1:])
                else:
                    dataFrame = pd.DataFrame(data=_data[1:, 0:],  # values
                                             columns=_data[0, 0:])
            elif _index:
                dataFrame = pd.DataFrame(data=_data[0:, 1:],  # values
                                         index=_data[0:, 0])  # 1st column as index)
            else:
                dataFrame = pd.DataFrame(data=_data)
        else:
            dataFrame = _data
        __datashape(dataFrame)
        return dataFrame.apply(pd.to_numeric)

    return __numpy2panda(data, header, index)
