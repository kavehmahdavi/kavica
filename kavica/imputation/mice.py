#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Multiple Imputation by Chained Equations (MICE)

These routines execute the Multiple Imputation by Chained Equations, and implement various
equations to interpolate the missing values in order to complete data.

Multivariate imputation by chained equations (MICE) has emerged as a principled method of dealing
with missing data. Despite properties that make MICE particularly useful for large imputation procedures.

We will modify the MICE due to mitigate certain complexities and limitations of the original approach.
The modified MICE will applicable on complex data like:
    - Strategy for imputing longitudinal data
    - Automatically incorporate the MICE procedure and clustering (on-going)
    - Addressing multi-level missing data
    - Multiple Imputation by Chained Multi_liner Equations (MICME)

Example:
    $ python3 mice.py config/config_IM_gromacs_64p.json ../parser/source.csv -i 10

Note:
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/

# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# Last update: 03/10/2022
"""

import numpy as np
import pandas as pd
import warnings
from terminaltables import DoubleTable
from scipy.stats.mstats import gmean, hmean
from time import sleep
import itertools
from sklearn import linear_model, discriminant_analysis
import json
import argparse
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from scipy import stats

# warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 365)
pd.set_option('display.width', 700)

__all__ = ['___config',
           'arguments_parser',
           'compatible_data_structure',
           'missing_pattern_plot',
           'MissingValuePreProcessing',
           'Mice',
           'scale_into_range',
           'dict_inner_joint',
           ]


# Fixme: pandas ix replace with loc/iloc
def scale_into_range(variable, r_min, r_max, t_min, t_max):
    """ Scales variable into a range [t_min,t_max].

    Args:
        variable (float): ∈ [r_min,r_max] denote your measurement to be scaled
        r_min (float): denote the minimum of the range of your measurement
        r_max (float): denote the maximum of the range of your measurement
        t_min (float): denote the minimum of the range of your desired target scaling
        t_max (float): denote the maximum of the range of your desired target scaling

    Returns:
        A float number indicates the scaled value.

    Note:
        See https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range.

    """
    return ((variable - r_min) * (t_max - t_min) / (r_max - r_min)) + t_min


def ___config(config_path, data_path):
    """ Read the configuration file (.json)

    In order to impute a file, we need to indicate the main features, pass_through and the complementaries.

    Args:
        config_path (str): indicates the path of the configuration file.
        data_path (str): indicates the path of the data file (.csv)

    Return:
        df (pandas data frame): Represents the data set where feature subset is {complimentary, hardware conters}
        pass_through_features (list):includes of all feature that are not used in MICE and they will staked to output.
        columns_order (list): It is a order (original order) list of the feature in data set.
        hardware_counters (list): Includes the list of the all feature that are imputed.
    """

    with open(config_path, 'r') as config_path:
        config_dict = json.load(config_path)
    df = pd.read_csv(data_path)  # Read the data file
    columns_order = list(df.columns.values)
    active_features = list(set(list(config_dict['hardware_counters'].values())
                               + list(config_dict['complimentary'].values())))

    pass_through_features = list(set(list(config_dict['pass_through'].values())
                                     + list(config_dict['complimentary'].values())))
    df = df[active_features]  # sub set of features

    return (df,
            pass_through_features,
            columns_order,
            list(config_dict['hardware_counters'].values()),
            config_dict['hardware_counters'])


def arguments_parser():
    """ Parse the arguments

    Return:
        A dict includes {"configPath", "csvPath", "predict_method", "imputedPath", "iteration"}
    """
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
                        help="The imputation method that is either norm, norm.nob, lda, qda, polyreg or logreg.")

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
        def __data_shape(__data):
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
        __data_shape(dataFrame)
        return dataFrame.apply(pd.to_numeric)

    return __numpy2panda(data, header, index)


def missing_pattern_plot(data, method='matrix', plot_name=None):
    """ Visualizing the patterns of missing value occurrence.

    Args:
        data (pandas): Includes the dataset.
        method (str): Indicates the plot format ("heatmap", "matrix", and "mosaic")
        plot_name (str): Identify the plot output file name

    Return:
        A jpeg image of the missing value patterns
    """
    # TODO: visualisation with the other plots such as Strictplot, bwplot, and densityplot
    if method.lower() == 'matrix':
        msno.matrix(data)
    elif method.lower() == 'mosaic':
        sns.heatmap(data.isnull(), cbar=False)
    elif method.lower() == 'bar':
        msno.bar(data)
    elif method.lower() == 'dendrogram':
        msno.dendrogram(data)
    if method.lower() == 'heatmap':
        msno.heatmap(data)
    plt.subplots_adjust(top=0.7)
    plt.savefig('{}.jpg'.format(plot_name))
    plt.show()


def dict_inner_joint(dict_left, dict_right):
    """Update the key of the left dictionary with the value of the right dictionary when left_value=right_key.

    Args:
        dict_left (dict): Includes the left dictionary
        dict_right (dict): Includes the right dictionary

    Returns:
        A dictionary.
    """
    new_dict = {}
    for item_key, item_value in dict_left.items():
        new_dict[dict_right.get(item_key)] = item_value
    return new_dict


class MissingValuePreProcessing(object):
    """ Class to preprocess the missing values

    Attributes:
        original_data (pandas): includes the original data before imputation.
        data (pandas): includes a copy of the original data that is gradually changed during the imputation process.
        missed_values_map (tuple): includes the two array, one for the row index and other for the columns of MVs.
        impute_method (str): indicates the initiative imputation method (default: Mean).
        impute_mask (np.array): indicates the initiate imputing value for a predicting feature in any epoc.
        imputed_data (pandas): includes the final output (imputed data)
        missing_value_number (int): indicates the number of the missing values in a dat set
        drop_column_threshold (float): defines a threshold that raises a warning about the high proportion of the missing
                                     value in a feature.
        drop_column (list): includes the names of some columns that we need to drop from the final output data.
        inplace (boolean): If True, the original csv fill will be replaced by the imputed dta set.
        not_drop_column_map (dict): represents the name and column number of the features
        feature_list (list): includes all feature names in the data frame

    Methods:
        __csv2hdf5: converts a csv file to a hdf5 format.
        __log_transformer: Applies log/ exponential transform on whole data set.
        __zero2nan: Replace the zero in indicated features with NaN
        _extract_missing_pattern: Analyses the missing data patterns
        write_csv: Writes the output (completed data set) into a csv file.
        _compatible_data_structure: Initializes and uniforms the input data.
        _compatible_data_structure: Initializes and uniforms the input data.
        _missing_value_map: Computes the missing value map (pattern)
        drop_null_row: Drops the rows that are fully NaN for all features.
        drop_null_column: Drops the columns that are fully NaN for all features.

    Properties:

    """

    def __init__(self, data=None, missed_values_map=None, impute_method=None, drop_column=False,
                 not_drop_column_map=dict(), drop_column_threshold=0.60, inplace=False, feature_list=None):
        """

        Args:
            data:
            missed_values_map:
            impute_method:
            drop_column:
            not_drop_column_map:
            drop_column_threshold:
            inplace:
            feature_list:
        """
        self.original_data = data
        self.original_data_dtypes = None
        self.data = data
        self.missed_values_map = missed_values_map
        self.impute_method = impute_method
        self.impute_mask = np.array([])
        self.imputed_data = None
        self.missing_value_number = None
        self.drop_column_threshold = drop_column_threshold
        self.drop_column = drop_column
        self.inplace = inplace
        self.not_drop_column_map = not_drop_column_map  # it is a binary array
        self.feature_list = feature_list

    def __call__(self):
        self._compatible_data_structure()
        self.__zero2nan(feature_list=self.feature_list)
        missing_pattern_plot(data=self.data, plot_name='initial_missing_pattern')
        self.__log_transformer()
        self._missing_value_map()
        self.write_csv()

    def __log_transformer(self, inverse=False):
        """Do log/exponential transform.

        We use the log and exponential transform to enforce the predication result to the positive value.

        Args:
            inverse (boolean): IF it is True, Log function is applied on self.data, else Exponential function is
            applied on self.imputedData.
        Return:
            The data frame
        """

        if inverse:
            self.imputed_data = self.imputed_data.apply(np.exp)
        else:
            self.data = self.data.apply(np.log)
        return self.data

    def __zero2nan(self, feature_list=None):
        """Replace the zero in indicated features with NaN

        Args:
            feature_list (list): indicates features that we would like to do the replacement on them.
        Return:
            self in order to apply chain action.
        """
        if not feature_list:
            self.data.replace(0, np.nan, inplace=True)
        else:
            self.data[feature_list] = self.data[feature_list].replace(0, np.nan)
        return self

    def _extract_missing_pattern(self):
        # TODO: (develop mode) do the imputation based on the complete record availability.
        print(self.data.columns)
        missing_value_groups = self.data.isnull().groupby(list(self.data.columns)).groups
        missing_value_patterns = pd.DataFrame(list(missing_value_groups.keys()), columns=self.data.columns)
        print(missing_value_patterns[['PAPI_L2_DCM', 'PAPI_L1_DCM', 'PAPI_BR_INS', 'PAPI_L3_TCM', 'PAPI_BR_MSP']])
        print(missing_value_groups)

    def __reset_dtypes(self):
        """ Reset the data type of the imputed data (output).

        It uses the original data type and reconstruct the output. But, in case of the INT, it rounds the data first
        then sets the type.

        Returns:
            self
        """
        for feature, data_type in self.original_data_dtypes.items():
            if 'int' in str(data_type):
                # Convert the decimal values to the integer
                self.imputed_data[feature] = self.imputed_data[feature].round().astype(str(data_type))
            else:
                self.imputed_data[feature] = self.imputed_data[feature].astype(str(data_type))
        return self

    def write_csv(self, append_to=None, csv_path=None, order=None, output_path='imputed.csv',
                  manipulating_list_path='manipulating_list.csv', manipulating=True, feature_dic=None):
        """ Write the output as CSV dataset

        Args:
             append_to (list) : of the pass_through_features that from the original dataset(data) and
                               append to the final output.
             csv_path (str): Includes the original dataset path.
             order (list): Includes the columns order of the final output.
             output_path (str): Indicates the output path.

        Return:
            A string includes the csv_path
        """

        if isinstance(self.imputed_data, pd.core.frame.DataFrame):
            self.__log_transformer(inverse=True)
            self.__reset_dtypes()
            appending_columns = pd.read_csv(csv_path, usecols=append_to)
            sin_complimentary = list(set(self.imputed_data.columns) - set(appending_columns))

            # Todo: (it needs more test) Compute the manipulating list.
            if manipulating:
                manipulating_list = pd.DataFrame({'row': self.missed_values_map[0],
                                                  'Hardware_Counter': self.missed_values_map[1]})
                manipulating_list['Values'] = manipulating_list.apply(
                    lambda item: self.imputed_data.iloc[item.row, item.Hardware_Counter], axis=1)
                for column_item in ['Object_id', 'Timestamp']:
                    manipulating_list[column_item] = manipulating_list.apply(
                        lambda item: appending_columns.loc[item.row, column_item], axis=1)
                manipulating_list = manipulating_list.drop(['row'], axis=1)
                mapping_list = dict_inner_joint(dict(map(reversed, feature_dic.items())),
                                                dict(map(reversed, enumerate(self.imputed_data.columns))))
                manipulating_list['Hardware_Counter'].replace(mapping_list, inplace=True)
                manipulating_order = ['Object_id', 'Timestamp', 'Hardware_Counter', 'Values']
                manipulating_list = manipulating_list[manipulating_order].sort_values(by=['Timestamp', 'Object_id'])
                manipulating_list.to_csv(manipulating_list_path, index=False)

            # Compute the imputed output csv file.
            self.imputed_data = pd.concat([appending_columns, self.imputed_data[sin_complimentary]], axis=1)
            self.imputed_data = self.imputed_data[order]  # reordering the data before writing csv
            self.imputed_data.to_csv(output_path, index=False)

            del appending_columns  # release the memory
        else:
            warnings.warn('The imputed data has not initiated yet.', UserWarning)
        return csv_path

    def _compatible_data_structure(self, data=None, header=True, index=True):
        """ Initialize and reappear the dataset (Internal)

        Args:
            data (a Numpy array or pandas data frame): is the data set
            header (boolean): if True, the first row of the data includes the header.
            index (boolean): if True, the first column of the data includes the index.

        Return:
            self
        """

        def __init(df):
            """ Test the empty data set

                data (pandas): The input dat set
            Return:
                A pandas data frame
            """
            if df is None:
                if self.data is None:
                    raise ValueError("The data set is empty")
                else:
                    pass
            else:
                self.data = df

        def __numpy2panda(headers, indexes):
            """ Convert 2D numpy array to pandas data frame.

            Args:
                headers (Boolean): If true, the first row includes header
                indexes (Boolean): If true, the first columns includes indexes

            Return:
                A pandas data frame
            """
            if type(self.data) is not pd.core.frame.DataFrame:
                if headers:
                    if indexes:
                        self.data = pd.DataFrame(data=self.data[1:, 1:],  # values
                                                 index=self.data[1:, 0],  # 1st column as index
                                                 columns=self.data[0, 1:])
                    else:
                        self.data = pd.DataFrame(data=self.data[1:, 0:],  # values
                                                 columns=self.data[0, 0:])
                elif indexes:
                    self.data = pd.DataFrame(data=self.data[0:, 1:],  # values
                                             index=self.data[0:, 0])  # 1st column as index)
                else:
                    self.data = pd.DataFrame(data=self.data)
            else:
                pass
            return self.data

        def __data_shape():
            """ Test the shape of the data frame

            The input data has to be 2D (Data Frame)

            Return:

            """
            if len(self.data.shape) is not 2:  # Check the shape
                raise ValueError("Expected 2d matrix, got %s array" % (self.data.shape,))
            elif self.data.empty:
                raise ValueError("Not expected empty data set.")
            else:
                print("The data frame is fitted with shape {}".format(self.data.shape))
            return self

        __init(data)
        __numpy2panda(header, index)
        __data_shape()
        self.original_data_dtypes = self.data.dtypes
        return self

    def _missing_value_map(self):
        """Computes the missing value map (pattern)

        Return:

        """

        def __sort_column_wise(column_wise=True):
            """ Sorts the missed Values Map.

            Args:
                column_wise (Boolean): If True, the missed Values Map is sorted vertically.
            """

            if column_wise is None:
                # TODO: do the row_wise sort
                pass
            else:
                missing_rows_index = np.array(self.missed_values_map[0])
                columns = np.array(self.missed_values_map[1])
                if column_wise:
                    ind = columns.argsort()
                    missing_rows_index = missing_rows_index[ind]
                    columns.sort()
                else:
                    ind = missing_rows_index.argsort()
                    columns = columns[ind]
                    missing_rows_index.sort()
            self.missed_values_map = (missing_rows_index, columns)

        rows = self.data.shape[0]
        _is_nulls = self.data.isnull()

        if not _is_nulls.sum().sum():
            raise ValueError('There is not any missing value in data frame.')
        elif _is_nulls.all().any():
            warnings.warn('All values are missed, therefore imputation is not possible.', UserWarning)
        else:
            tableData = [['', 'Missed\nValues']]
            featureList = self.data.columns.values.tolist()
            missedValueList = _is_nulls.sum().tolist()
            print(featureList)
            for [featureItem, missingValues] in zip(featureList, missedValueList):
                missingValues = missingValues / rows
                if missingValues < self.drop_column_threshold:
                    self.not_drop_column_map.update({featureItem: featureList.index(featureItem)})
                elif self.drop_column:
                    self.data = self.data.drop([featureItem], axis=1)
                    print('\n {} is deleted.'.format(featureItem))
                else:
                    warnings.warn('\n The feature {} has {}% missing value,it should drop or request for new data set.'.
                                  format(featureItem, missingValues * 100))
                    sleep(0.01)
                    decision = input('\n\033[1m\033[95mD\033[0mrop the feature and continue' +
                                     '\n\033[1m\033[95mC\033[0montinue without dropping' +
                                     '\n\033[1m\033[95mE\033[0mxit' +
                                     '\n\033[6mInsert the code(D|C|E):\033[0').upper()
                    while True:
                        if decision == 'D':
                            # fixme: The dropping a column will cusses problem in reordering the columns to write csv
                            print(self.original_data_dtypes)
                            self.data = self.data.drop([featureItem], axis=1)
                            # self.original_data_dtypes = self.original_data_dtypes.drop(featureItem)
                            print('\n {} is deleted.'.format(featureItem))
                            break
                        elif decision == 'C':
                            self.not_drop_column_map.update({featureItem: featureList.index(featureItem)})
                            break
                        elif decision == 'E':
                            raise ValueError('The data set has massive amount of missing values.')
                        else:
                            decision = input('\n\033[6mInsert the code(D|C|E):\033[0')
                tableData.append([featureItem,
                                  '{:3.1f}%'.format(missingValues * 100)])
            table = DoubleTable(tableData)
            table.justify_columns[1] = 'center'
            print(table.table)

            # Reindexing the self.property based on the feature that are dropped
            _is_nulls = self.data.isnull()
            # initiate the imputation mask and missed value map
            self.missed_values_map = np.asarray(_is_nulls).nonzero()
            self.impute_mask = np.zeros(len(self.missed_values_map[0]))
            self.missing_value_number = _is_nulls.sum().sum()
            __sort_column_wise()

    def drop_null_row(self):
        """ Drop the full NaN rows.

        Return:
            An int that indicates the number of the draped rows
        """
        if self.inplace:
            df_length_before = self.data.shape[0]
            self.data = self.data.dropna(how='any', axis=0)
            number_of_dropped_rows = df_length_before - self.data.shape[0]
        else:
            df_length_before = self.imputed_data.shape[0]
            self.imputed_data = self.data.dropna(how='any', axis=0)
            number_of_dropped_rows = df_length_before - self.imputed_data.shape[0]
        print("{} number of rows have been draped as fully NaN rows.".format(number_of_dropped_rows))
        return number_of_dropped_rows

    def drop_column(self):
        """ Drop the full NaN columns.

        Return:
            An int that indicates the number of the draped columns
        """
        if self.inplace:
            df_length_before = self.data.shape[1]
            self.data = self.data.dropna(how='all', axis=1)
            number_of_dropped_columns = df_length_before - self.data.shape[1]
        else:
            df_length_before = self.imputed_data.shape[1]
            self.imputed_data = self.data.dropna(how='all', axis=1)
            number_of_dropped_columns = df_length_before - self.imputed_data.shape[1]
        print("{} number of rows have been draped as fully NaN columns.".format(number_of_dropped_columns))
        return number_of_dropped_columns

    def simple_imputation(self, impute_method='imputeMean', inplace=False):
        """ Initially impute the missing values.

        Args:
            impute_method (str): indicates the initiate imputation method of the NaNs
                                'imputZero',
                                'imputMedian',
                                'imputMax',
                                'imputMin',
                                'imputMean',
                                'imputGeometricMean',
                                'imputHarmonicMean',
            inplace (boolean): if True, the imputation will done on the original dataset.

        Returns:
            self in order to apply the chain of the functions.
        """
        impute_methods = [
            'imputeZero',
            'imputeMedian',
            'imputeMax',
            'imputeMin',
            'imputeMean',
            'imputeGeometricMean',
            'imputeHarmonicMean',
            None
        ]
        assert impute_method in impute_methods

        def __geo_mean(df):
            """Compute the geometric mean of any feature

            Args:
                df (Pandas): Includes the data

            Returns:
                A list of the geometric mean for all feature (column)
            """
            __geo_means = []
            for column_item in df:
                no_zero_nan_column_item = list(df[column_item].replace(0, pd.np.nan).dropna(axis=0, how='any'))
                __geo_means.append(gmean(no_zero_nan_column_item))
            return __geo_means

        def __harmo_mean(df):
            """Compute the harmonic mean of any feature

            Args:
                df (Pandas): Includes the data

            Returns:
                A list of the harmonic mean for all feature (column)
            """
            _harmo_means = []
            for column_item in df:
                no_zero_nan_column_item = list(df[column_item].replace(0, pd.np.nan).dropna(axis=0, how='any'))
                _harmo_means.append(hmean(no_zero_nan_column_item))
            return _harmo_means

        def __generat_missed_values_map():
            """ Generator the missing values map

            Yields:
                A list of a pair [index, column] of a missing value position in the data.
            """
            not_droped_feature_index = self.not_drop_column_map.values()
            for [index_item, header_item] in zip(self.missed_values_map[0], self.missed_values_map[1]):
                if header_item in not_droped_feature_index:
                    real_header_index = list(self.not_drop_column_map.values()).index(header_item)
                    yield [index_item, real_header_index]

        def _impute():
            """Applies the initial imputation

            Returns:
                self
            """
            if inplace:
                for [index_item, header_item] in zip(self.missed_values_map[0], self.missed_values_map[1]):
                    self.data.iat[index_item, header_item] = self.impute_mask[header_item]
            else:
                self.imputed_data = self.data.copy(deep=True)
                for [index_item, header_item] in zip(self.missed_values_map[0], self.missed_values_map[1]):
                    self.imputed_data.iat[index_item, header_item] = self.impute_mask[header_item]
            return self

        if impute_method == 'imputeZero':
            self.impute_mask.fill(0)
        elif impute_method == 'imputeMedian':
            self.impute_mask = np.array(self.data.median(axis=0, skipna=True))
        elif impute_method == 'imputeMax':
            self.impute_mask = np.array(self.data.max(axis=0, skipna=True))
        elif impute_method == 'imputeMin':
            self.impute_mask = np.array(self.data.min(axis=0, skipna=True))
        elif impute_method == 'imputeMean':
            self.impute_mask = np.array(self.data.mean(axis=0, skipna=True))
        elif impute_method == 'imputeGeometricMean':
            self.impute_mask = np.array(__geo_mean(self.data))
        elif impute_method == 'imputeHarmonicMean':
            self.impute_mask = np.array(__harmo_mean(self.data))
        else:
            raise ValueError('\n Initial impute method is selected \n ')

        _impute()
        return self


class Mice(MissingValuePreProcessing):
    """Multiple imputation by chained equation.

    Attributes:
        impute_method (str): Indicate the initiate imputation method.
        train_subset_x (pandas): Includes a slice of independent variables for training the predictive equation.
        test_subset_x (pandas): Includes a slice of independent variables for testing the predictive equation.
        train_subset_y (pandas): Includes a slice of dependent variables for training the predictive equation.
        test_subset_y (pandas): Includes a slice of dependent variables for testing the predictive equation.
        iteration (int): Indicates the number of imputation epocs.
        iteration_log (list): Include the list of list of the imputed values
        predict_Method (str): Indicates a predicative model.

    Methods:
        predictive_model:
        __place_holder:
        __impute:
        imputer:
    Output:
        The standard outputs of the mice are:
            - imputed.csv: includes the completed csv file
            - manipulating_list.csv: includes the list of the changes/imputation which was applied
    """

    def __init__(self, data=None, impute_method=None, predict_method='norm', iteration=10, feature_list=None):
        """Initiate the MICE class object.

        Args:
            data (pandas): Includes the data set
            impute_method (str): indicates the initiate imputation method of the NaNs
            predict_method (str): indicates the mice predictive method.
            iteration (int): Indicates the number of imputation epoch.
            feature_list (list): Includes the feature list

        """
        super(Mice, self).__init__(data=data, feature_list=feature_list)
        self.impute_method = impute_method
        self.train_subset_x = None
        self.test_subset_x = None
        self.train_subset_y = None
        self.test_subset_y = None
        self.iteration = iteration
        self.iteration_log = np.zeros(shape=(0, 0))
        self.predict_Method = predict_method

    def __call__(self):
        super(Mice, self).__call__()
        # After running the supper __call__, we need to reshape the iteration log.
        self.iteration_log = np.zeros(shape=(self.iteration, self.missing_value_number))
        self.imputer()
        missing_pattern_plot(self.imputed_data, method='matrix', plot_name='imputed_missing_pattern')

    def predictive_model(self):
        """Setup and predict the missing values.

        Returns:
            A numpy array includes the predicted value.

        Note:
            - QDA is sensitive about the number of the instances in a class (>1).
        """
        # TODO: complete the function list
        # TODO: Write the customised functions and define the functions (Tensor, Fuzzy, Fourier model, ...)
        # fixme: test the function's quality
        methods = [
            'pmm',  # Predictive mean matching (numeric) fixme
            'norm',  # Bayesian liner regression (numeric)
            'norm.nob',  # Linear regression, non-Bayesian (numeric)
            'mean.boot',  # Linear regression with bootstrap (numeric) fixme
            'mean',  # Unconditional mean imputation (numeric) fixme
            '2l.norm',  # Two-level linear model (numeric) fixme
            'logreg',  # Logistic regression (factor, level2)
            'logreg.bot',  # Logistic regression with bootstrap (factor, level2) fixme
            'polyreg',  # Multinomial logit model (factor > level2)
            'lda',  # Linear discriminant analysis (factor)
            'qda',  # QuadraticDiscriminantAnalysis (factor),
            'SRS',  # Simple random sampling  fixme
            'fuzzy',  # fixme
            'KNN',  # fixme
            None
        ]
        assert self.predict_Method in methods

        def modeler(method_to_run, scale_in_range=True):  # Receives the function as parameter
            """Fit the predictive model

            Receives the function as parameter

            Args:
                method_to_run (str): Indicate the name of the predictive model.
                scale_in_range (boolean): If True, the predicted values are scaled into the observed range.
            Returns:
                A numpy array includes the predicted value.
            """
            # Fitting the training y, it is needed when we are using 'sklearn' package.
            flat_train_y = np.array(self.train_subset_y.iloc[:, 0].values.tolist())

            # Create linear regression object
            predictor = method_to_run

            # Train the model using the training sets
            predictor.fit(self.train_subset_x, flat_train_y)

            # Make predictions using the testing set
            predictedY = predictor.predict(self.test_subset_x)
            # The predicted values  ->    print(predictedY)
            # The coefficients      ->    print('Coefficients: \n', predictor.coef_)

            # scale the predicted values in a range of the real values.
            if scale_in_range:
                v_func = np.vectorize(scale_into_range)
                predictedY = v_func(predictedY, min(predictedY), max(predictedY), min(flat_train_y), max(flat_train_y))

            # standardise the output format 2D np.array
            if not any(isinstance(e, np.ndarray) for e in predictedY):
                predictedY = np.array([np.array([element]) for element in predictedY])

            itemSize = set([element.size for element in predictedY])
            if bool(itemSize.difference({1})):
                raise ValueError(
                    '\n MICE Predication Error: The prediction method {} output is not standardised.'.format(
                        self.predict_Method))

            return predictedY

        # MICE prediction method switch-case
        if self.predict_Method == 'norm.nob':
            method = linear_model.LinearRegression(fit_intercept=False)
        elif self.predict_Method == 'norm':
            method = linear_model.BayesianRidge(compute_score=True)
        elif self.predict_Method == 'lda':
            method = discriminant_analysis.LinearDiscriminantAnalysis()
        elif self.predict_Method == 'qda':
            method = discriminant_analysis.QuadraticDiscriminantAnalysis()
        elif self.predict_Method == 'polyreg':
            method = linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        elif self.predict_Method == 'logreg':
            method = linear_model.LogisticRegression(random_state=0, solver='sag', multi_class='ovr')
        return modeler(method)

    def __place_holder(self, feature_item):
        """ Hold the missing value place.

        In any imputation epoch, we need to create the train and test_cp data.So, after any iteration we need to
        retrieve the original place of the missing data in order to create the test_cp data set.

        Args:
            feature_item (str): Indicate the dependent feature that will be predicted.

        Returns:
            A list of the row index that indicates the missing values places in feature item.

        """
        feature_name = self.data.columns.values.tolist()[feature_item]
        place_holder_column_index = list(map(lambda x: 1 if x == feature_item else 0, self.missed_values_map[1]))
        place_holder_rows = list(itertools.compress(self.missed_values_map[0], place_holder_column_index))
        # Converting the rows coordinate to the data frame Index before imputing the None
        place_holder_row_index = [self.data.index.tolist()[x] for x in place_holder_rows]
        if self.inplace:
            self.data.loc[place_holder_row_index, feature_item] = None
            train_subset = self.data.iloc[self.data[feature_name].notnull()]
            test_subset = self.data[self.data[feature_name].isnull()]
        else:
            self.imputed_data.loc[place_holder_row_index, feature_name] = None
            train_subset = self.imputed_data[self.imputed_data[feature_name].notnull()]
            test_subset = self.imputed_data[self.imputed_data[feature_name].isnull()]
        self.train_subset_x = train_subset.drop(feature_name, axis=1).copy()
        self.train_subset_y = train_subset[[feature_name]].copy()
        self.test_subset_x = test_subset.drop(feature_name, axis=1).copy()
        self.test_subset_y = test_subset[[feature_name]].copy()
        return place_holder_rows

    def __impute(self, row_indexes=None, predicted_values=None, column_index=None):
        """Update the data set with the imputed values.

        In any epoc, the original/imputed data set is updated by the predicted values.

        Args:
            row_indexes (list): Includes the missing value row index.
            predicted_values (list): Includes the predicted values that is ordered by the index.
            column_index (str): Indicates the predicted (dependent) feature name.

        Returns:
            self
        """
        if self.inplace:
            for [rowIndex, predicted_value] in zip(row_indexes, predicted_values):
                self.data.iat[rowIndex, column_index] = predicted_value[0]
        else:
            for [rowIndex, predicted_value] in zip(row_indexes, predicted_values):
                self.imputed_data.iat[rowIndex, column_index] = predicted_value[0]
        return self

    def imputer(self):
        """Imputation engine

        It does the imputation feature by feature, and it repeats the imputation by the number of iteration.

        Returns:
            self
        """

        def __plot_conversion(missing_value_index=0):
            """Plot the conversion curve

            We use this function in order to test_cp or visualising the progressive conversion of a missing value.

            Args:
                missing_value_index (int): Indicates the value that we would like to see it's conversion plot.
            """
            plt.plot(list(range(0, self.iteration)),
                     self.iteration_log[:, missing_value_index],
                     'bo',
                     list(range(0, self.iteration)),
                     self.iteration_log[:, missing_value_index],
                     'k')
            plt.axis([0, self.iteration,
                      np.min(self.iteration_log[:, missing_value_index]) - 1,
                      np.max(self.iteration_log[:, missing_value_index]) + 1])

            plt.ylabel('Iteration')
            plt.show()

        def __mean_squared_displacement_plot(root=True):
            """Plot Mean Squared Displacement.

            We use this function in order to test_cp or visualising the progressive Mean Squared Displacement.

            Args:
                root (boolean): If True, the root mean squared displacement will be computed
            """

            iteration_log_df = pd.concat([pd.DataFrame(self.impute_mask).T,
                                          pd.DataFrame(self.iteration_log)])
            msd = iteration_log_df.diff().apply(np.square).mean(axis=1)[1:]
            msd.index = msd.index + 1
            y_lab = "Mean Squared Displacement (MSD)"
            if root:
                msd = iteration_log_df.diff().apply(np.square).mean(axis=1).apply(np.sqrt)
                y_lab = "Root Mean Squared Displacement (RMSD)"

            plt.plot(msd.values, marker='o')
            plt.xlabel('Iteration')
            plt.ylabel(y_lab)
            plt.title('Stabilization Curve of Multiple Imputation'.format(y_lab))
            plt.legend([y_lab])
            plt.grid()
            plt.xticks(range(1, self.iteration + 1))
            plt.savefig('Root_Mean_Squared_Displacement.jpg')
            plt.show()

        __feature_with_none = set(self.missed_values_map[1])
        self.simple_imputation(impute_method='imputeMean')  # Step1: Mice
        iterations = iter(range(0, self.iteration))

        __done_loop = False
        while not __done_loop:
            try:
                iteration = next(iterations)
                print('-' * 100, '\n', 'The iteration {} is started:'.format(iteration + 1), '\n', '-' * 100)
                impute_values_ordered_by_col = []
                for feature_item in __feature_with_none:
                    row_indexes = self.__place_holder(feature_item=feature_item)  # Step2: Mice
                    predicted_values = self.predictive_model()  # methodName='norm'
                    self.__impute(row_indexes, predicted_values, feature_item)
                    print(predicted_values.ravel().tolist())
                    impute_values_ordered_by_col.append(list(predicted_values.flatten()))
            except StopIteration:
                __done_loop = True
            else:
                # Flatten the list of list ^ add to the iteration log
                self.iteration_log[iteration] = np.exp(list(itertools.chain(*impute_values_ordered_by_col)))
        table = DoubleTable(self.iteration_log.tolist())
        table.inner_heading_row_border = False
        table.justify_columns[1] = 'center'
        __mean_squared_displacement_plot()
        __plot_conversion()
        return self

    def test_likelihood(self, _feature=None, _plot=True, _log=True):
        """ Measure Probability Distribution Similarity

        Performs the one-sample Kolmogorov-Smirnov test for goodness of fit.

        The one-sample test compares the underlying distribution F(x) of a sample against a given distribution G(x).
        The two-sample test compares the underlying distributions of two independent samples. Both tests are valid only
        for continuous distributions.

        Args:
            x (pandas): indicate the original dats
            y (pandas): indicate the sample dats
            _feature (str): indicates the feature name to test. If None, ic computes the average test value of features.
            _plot (boolean): plot the histogram. Note that is just applicable for the one feature.
            _log (boolean): compute the logarithmic value of the features

        Returns:
            statistic (float):  KS test statistic, either D, D+ or D-.
            pvalue (float): One-tailed p-value.

        Note:
            See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
        """
        x = self.original_data.copy(deep=True)
        y = self.imputed_data.copy(deep=True)
        if _feature is None:
            _kstest_dict = pd.DataFrame(columns=['Feature', 'Statistic', 'Pvalue'])
            for _feature_item in self.feature_list:
                _kstest_temp = stats.kstest(x[_feature_item], y[_feature_item])
                _kstest_temp = stats.kstest(x[_feature_item], y[_feature_item])
                _kstest_dict = _kstest_dict.append({'Feature': _feature_item,
                                                    'Statistic': _kstest_temp[0],
                                                    'Pvalue': _kstest_temp[1]}, ignore_index=True)
            _kstest_dict = _kstest_dict.append(_kstest_dict.agg('mean', numeric_only=True), ignore_index=True)
            _kstest_dict.at[_kstest_dict.index[-1], 'Feature'] = 'mean'
        else:
            _kstest_dict = {_feature: stats.kstest(x[_feature], y[_feature])}

        if _plot and _feature:
            if _log:
                x[_feature], y[_feature] = np.log(x[_feature]), np.log(y[_feature])
                x, y = x.replace([np.inf, -np.inf], np.nan), y.replace([np.inf, -np.inf], np.nan)
                x, y = x.dropna(), y.dropna()
            sns.distplot(y[_feature], label='Imputed')
            sns.distplot(x[_feature], label='Original')
            plt.title('Distribution Likelihood od {}'.format(_feature))
            plt.legend()
            plt.show()
        print(_kstest_dict)
        return _kstest_dict

    # TODO: Post-possessing
    """
        - Post-possessing  ( Non-negative, 
                                    Integer , 
                                    In the boundary)
    """
    # TODO: Define the constraints
    """
        - Define the constraints (Fully conditional specification-FCS, 
                                  Monotone data imputation, 
                                  Joint modeling)
    """


def __mice():
    """Handling the command line

    Example:
        $ python3 mice.py config/config_IM_gromacs_64p.json ../parser/source.csv -i 10

    Returns:
        An object of class MICE.
    """
    start = time.time()
    try:
        args = arguments_parser()
        df, features_appending_list, columns_order, feature_list, feature_dic = ___config(args['configPath'],
                                                                                          args['csvPath'])

        _mice = Mice(df, predict_method=args['predict_method'], iteration=args['iteration'], feature_list=feature_list)
        _mice()
        _mice.write_csv(output_path=args['imputedPath'],
                        append_to=features_appending_list,
                        csv_path=args['csvPath'],
                        order=columns_order,
                        manipulating=True,
                        feature_dic=feature_dic)
        print(_mice.test_likelihood())
        print("\033[32mThe missing value imputation process is successfully completed by MICE method.")
        return _mice
    except AssertionError as error:
        print(error)
        print("\033[31mThe missing value imputation proses is failed.")
    finally:
        duration = time.time() - start
        print('\033[0mTotal duration is: %.3f' % duration)


# ---------------------------------------------------------------------------
def __test_me():
    data = np.array([("ind", "F1", "F2", "F3", "F4", "F5", "F6"),
                     (1, 2, 0, 13, None, 12, None),
                     (2, 2, 45, 23, 24, 13, 16),
                     (3, 4, 45, 23, 24, 19, 16),
                     (4, 2, 44, 23, 22, 13, 11),
                     (5, 4, 7, 50, 5, 20, 89),
                     (6, None, None, 34, 7, None, 67)])
    obj = Mice(data)
    print(obj.original_data)
    obj()
    print(obj.imputed_data)
    print(obj.missed_values_map)


def __test_me_iris():
    from sklearn import datasets
    from sklearn.metrics import r2_score
    import random

    data = datasets.load_iris().data[:, :4]
    data = pd.DataFrame(data, columns=['F1', 'F2', 'F3', 'F4'])
    data1 = data.copy()
    x = []
    y = []
    old_value = []
    mean_ind = data1.mean(axis=0).values
    mean_list = []

    for i in range(16):
        xv = random.randint(0, 145)
        yv = random.randint(0, 3)
        old_value.append(data1.iloc[xv, yv])
        mean_list.append(mean_ind[yv])
        data.iloc[xv, yv] = np.NaN
        x.append(xv)
        y.append(yv)

    obj = Mice(data, iteration=100)
    obj()

    pred = []
    for i, j, v in zip(x, y, old_value):
        print(i, j, '--', v, '-', obj.imputed_data.iloc[i, j])
        pred.append(obj.imputed_data.iloc[i, j])

    print(r2_score(old_value, pred, multioutput='variance_weighted'))
    print(1 - (1 - r2_score(old_value, pred, multioutput='variance_weighted')) * (data1.shape[0] - 1) / (
            data1.shape[0] - data1.shape[1] - 1))
    print('-+' * 30)
    print(r2_score(old_value, mean_list, multioutput='variance_weighted'))
    print(1 - (1 - r2_score(old_value, mean_list, multioutput='variance_weighted')) * (data1.shape[0] - 1) / (
            data1.shape[0] - data1.shape[1] - 1))


if __name__ == '__main__':
    object1 = __mice()
