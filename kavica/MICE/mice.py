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
from sklearn import linear_model, discriminant_analysis
import json
import argparse
import sys
import time
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

__all__ = ['__configoration',
           'data_structure_compatibilization',
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
    return df, pass_through_features, columns_order, list(config_dict['hardware_counters'].values())


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


def data_structure_compatibilization(data=None, header=True, index=True):
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


class MissingValuePreProcessing(object):

    def __init__(self, data=None,
                 missedValuesMap=None,
                 imputMethod=None,
                 dropColumn=False,
                 notDropColumnMap=dict(),
                 dropColumnThreshold=0.3,
                 inplace=False,
                 feature_list=None):
        self.orginalData = data
        self.data = data
        self.missedValuesMap = missedValuesMap
        self.imputMethod = imputMethod
        self.imputMask = np.array([])
        self.imputedData = None
        self.missingValueNumber = None
        self.dropColumnThreshold = dropColumnThreshold
        self.dropColumn = dropColumn
        self.inplace = inplace
        self.notDropColumnMap = notDropColumnMap  # it is a binary array
        self.feature_list = feature_list

    def __call__(self):
        self._data_structure_compatibilization()
        self.__zero2nan(feature_list=self.feature_list)
        self._extract_missing_pattern()
        self.missing_pattern_plot()
        self._missing_value_map()
        self._write_csv()

    # TODO
    def __csv2hdf5(self):
        pass

    def __zero2nan(self, feature_list=None):
        """Replace the zero in indicated features with NaN

        Return:

        """
        if not feature_list:
            self.data.replace(0, np.nan, inplace=True)  # for test package
        else:
            self.data[feature_list] = self.data[feature_list].replace(0, np.nan)

    def _extract_missing_pattern(self):

        print(self.data.columns)
        missing_value_groups=self.data.isnull().groupby(list(self.data.columns)).groups

        missing_value_patterns=pd.DataFrame(list(missing_value_groups.keys()),columns=self.data.columns)
        print(missing_value_patterns[['PAPI_L2_DCM', 'PAPI_L1_DCM', 'PAPI_BR_INS','PAPI_L3_TCM', 'PAPI_BR_MSP']])
        print(missing_value_groups)
        exit()
        pass


    def _write_csv(self, appendTo=None, csvPath=None, order=None, output_path='imputed.csv'):
        """ Write the output as CSV dataset
        :param appendTo:
        :param csvPath:
        :param order:
        :param output_path:
        :return:
        """
        if isinstance(self.imputedData, pd.core.frame.DataFrame):
            # read the pass_through_features from the original dataset(data) and append to the final output.
            appending_columns = pd.read_csv(csvPath, usecols=appendTo)
            sin_complimentary = list(set(self.imputedData.columns) - set(appending_columns))

            self.imputedData = pd.concat([appending_columns, self.imputedData[sin_complimentary]], axis=1)

            # release the memory
            del appending_columns

            # reordering the data before writing csv
            self.imputedData = self.imputedData[order]
            self.imputedData.to_csv(output_path, index=False)

        else:
            warnings.warn('The imputed data has not initiated yet.', UserWarning)

    def _data_structure_compatibilization(self, data=None, header=True, index=True):
        """ Initialization the data set
        :param data:
        :param header:
        :param index:
        :return:
        """

        def __init(data):
            if data is None:
                if self.data is None:
                    raise ValueError("The data set is empty")
                else:
                    pass
            else:
                self.data = data

        # Convert to data frame
        def __numpy2panda(header, index):
            if type(self.data) is not pd.core.frame.DataFrame:
                if header:
                    if index:
                        self.data = pd.DataFrame(data=self.data[1:, 1:],  # values
                                                 index=self.data[1:, 0],  # 1st column as index
                                                 columns=self.data[0, 1:])
                    else:
                        self.data = pd.DataFrame(data=self.data[1:, 0:],  # values
                                                 columns=self.data[0, 0:])
                elif index:
                    self.data = pd.DataFrame(data=self.data[0:, 1:],  # values
                                             index=self.data[0:, 0])  # 1st column as index)
                else:
                    self.data = pd.DataFrame(data=self.data)
            else:
                pass

        def __datashape():
            if len(self.data.shape) is not 2:  # Check the shape
                raise ValueError("Expected 2d matrix, got %s array" % (self.data.shape,))
            elif self.data.empty:
                raise ValueError("Not expected empty data set.")
            else:
                pass

        __init(data)
        __numpy2panda(header, index)
        __datashape()

    def _missing_value_map(self):
        def __sortColumnwise(columnwise=True):

            if columnwise is None:
                pass
            else:
                rows = np.array(self.missedValuesMap[0])
                columns = np.array(self.missedValuesMap[1])
                if columnwise:
                    ind = columns.argsort()
                    rows = rows[ind]
                    columns.sort()
                else:
                    ind = rows.argsort()
                    columns = columns[ind]
                    rows.sort()
            self.missedValuesMap = (rows, columns)

        rows = self.data.shape[0]
        isnulls = self.data.isnull()

        if not isnulls.sum().sum():
            raise ValueError('There is not any missing value in data frame.')
        elif isnulls.all().any():
            warnings.warn('All values are missed, therefore imputation is not possible.',
                          UserWarning)
        else:
            tableData = [['', 'Missed\nValues']]
            featureList = self.data.columns.values.tolist()
            missedValueList = isnulls.sum().tolist()
            print(featureList)
            for [featureItem, missingValues] in zip(featureList,
                                                    missedValueList):
                missingValues = missingValues / rows
                if missingValues < self.dropColumnThreshold:
                    self.notDropColumnMap.update({featureItem: featureList.index(featureItem)})
                elif self.dropColumn:
                    self.data = self.data.drop([featureItem], axis=1)
                    print('\n {} is deleted.'.format(featureItem))
                else:
                    warnings.warn('\n The feature {} has {}% missing value,'
                                  ' it should drop or request for new data set.'.
                                  format(featureItem,
                                         missingValues * 100))
                    sleep(0.01)
                    decision = input('\n\033[1m\033[95mD\033[0mrop the feature and continue' +
                                     '\n\033[1m\033[95mC\033[0montinue without dropping' +
                                     '\n\033[1m\033[95mE\033[0mxit' +
                                     '\n\033[6mInsert the code(D|C|E):\033[0').upper()

                    while (True):
                        if decision == 'D':
                            self.data = self.data.drop([featureItem], axis=1)
                            print('\n {} is deleted.'.format(featureItem))
                            break
                        elif decision == 'C':
                            self.notDropColumnMap.update({featureItem: featureList.index(featureItem)})
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

            # Reindexing the self.property based on teh feature that are dropped
            isnulls = self.data.isnull()
            # initiate the impute mask and missed value map
            self.missedValuesMap = np.asarray(isnulls).nonzero()
            self.imputMask = np.zeros(len(self.missedValuesMap[0]))
            self.missingValueNumber = isnulls.sum().sum()
            __sortColumnwise()

    def drop_null_row(self):
        if self.inplace:
            self.data = self.data.dropna(how='any',
                                         axis=0)
        else:
            self.imputedData = self.data.dropna(how='any',
                                                axis=0)

    def drop_column(self, inplace=False):
        if self.inplace:
            self.data = self.data.dropna(how='all',
                                         axis=1)
        else:
            self.imputedData = self.data.dropna(how='all',
                                                axis=1)

    def simple_imputation(self, imputMethod='imputMean', inplace=False):
        imputMethods = [
            'imputZero',
            'imputMedian',
            'imputMax',
            'imputMin',
            'imputMean',
            'imputGeometricMean',
            'imputHarmonicMean',
            None
        ]
        assert imputMethod in (imputMethods)

        def __gMean(df):
            gmeans = []
            for columnItem in df:
                noZeroNanColumnItem = list(df[columnItem].replace(0, pd.np.nan).
                                           dropna(axis=0, how='any'))
                gmeans.append(gmean(noZeroNanColumnItem))
            return gmeans

        def __hMean(df):
            hmeans = []
            for columnItem in df:
                noZeroNanColumnItem = list(df[columnItem].replace(0, pd.np.nan).
                                           dropna(axis=0, how='any'))
                hmeans.append(hmean(noZeroNanColumnItem))
            return hmeans

        def generatorMissiedValuesMap():
            notDropedFeatureIndex = self.notDropColumnMap.values()
            for [indexItem, headerItem] in zip(self.missedValuesMap[0],
                                               self.missedValuesMap[1]):
                if headerItem in notDropedFeatureIndex:
                    realHeaderIndex = list(self.notDropColumnMap.values()).index(headerItem)
                    yield [indexItem, realHeaderIndex]

        def _imput():
            if inplace:
                for [indexItem, headerItem] in zip(self.missedValuesMap[0],
                                                   self.missedValuesMap[1]):
                    self.data.iat[indexItem, headerItem] = self.imputMask[headerItem]
            else:
                self.imputedData = self.data.copy(deep=True)
                for [indexItem, headerItem] in zip(self.missedValuesMap[0],
                                                   self.missedValuesMap[1]):
                    self.imputedData.iat[indexItem, headerItem] = self.imputMask[headerItem]

        if imputMethod == 'imputZero':
            self.imputMask.fill(0)
        elif imputMethod == 'imputMedian':
            self.imputMask = np.array(self.data.median(axis=0,
                                                       skipna=True))
        elif imputMethod == 'imputMax':
            self.imputMask = np.array(self.data.max(axis=0,
                                                    skipna=True))
        elif imputMethod == 'imputMin':
            self.imputMask = np.array(self.data.min(axis=0,
                                                    skipna=True))
        elif imputMethod == 'imputMean':
            self.imputMask = np.array(self.data.mean(axis=0,
                                                     skipna=True))
        elif imputMethod == 'imputGeometricMean':
            self.imputMask = np.array(__gMean(self.data))
        elif imputMethod == 'imputHarmonicMean':
            self.imputMask = np.array(__hMean(self.data))
        else:
            warnings.warn('\n Nan impute method is selected \n ', UserWarning)
            return
        _imput()

    # TODO: visualisation with: Stricplot, bwplot, densityplot
    def missing_pattern_plot(self, method='matrix'):
        """Visualizing the patterns of missing value occurrence

        Args:
            method (str): Indicates the plot format ("heatmap", "matrix", and "mosaic")
        Return:
            A jpeg image of the missing value patterns
        """
        if method.lower() == 'matrix':
            msno.matrix(self.data)
        elif method.lower() == 'mosaic':
            sns.heatmap(self.data.isnull(), cbar=False)
        elif method.lower() == 'bar':
            msno.bar(self.data)
        elif method.lower() == 'dendrogram':
            msno.dendrogram(self.data)
        if method.lower() == 'heatmap':
            msno.heatmap(self.data)
        plt.show()


class Mice(MissingValuePreProcessing):

    def __init__(self, data=None, imputMethod=None, predictMethod='norm', iteration=10, feature_list=None):
        super(Mice, self).__init__(data=data, feature_list=feature_list)
        self.imputMethod = imputMethod
        self.trainSubsetX = None
        self.testSubsetX = None
        self.trainSubsetY = None
        self.testSubsetY = None
        self.model = None
        self.iteration = iteration
        self.iterationLog = np.zeros(shape=(0, 0))
        self.predictMethod = predictMethod

    def __call__(self):
        super(Mice, self).__call__()
        # After running teh supper __call__, i need to reshape teh iteration log.
        self.iterationLog = np.zeros(shape=(self.iteration,
                                            self.missingValueNumber))
        self.imputer()

    def predictive_model(self):
        """
        Note:
            - QDA is sensitive about the number of the instances in a class (>1).
        """
        methods = [
            # TODO: complete the function list
            # TODO: Write the customised functions and define the functions
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
        assert self.predictMethod in (methods)

        # IsDone: send the function as parameter
        def modeler(methodToRun):
            # Fitting the training y, it is needed when we are using 'Sklearn' package.
            flatedTrainY = np.array(self.trainSubsetY.iloc[:, 0].values.tolist())

            # Create linear regression object
            regr = methodToRun

            # Train the model using the training sets
            regr.fit(self.trainSubsetX, flatedTrainY)

            # Make predictions using the testing set
            predictedY = regr.predict(self.testSubsetX)
            # The predicted values  ->    print(predictedY)
            # The coefficients      ->    print('Coefficients: \n', regr.coef_)

            # standardise the output format 2D np.array
            if not any(isinstance(e, np.ndarray) for e in predictedY):
                predictedY = np.array([np.array([element]) for element in predictedY])

            itemSize = set([element.size for element in predictedY])
            if bool(itemSize.difference({1})):
                raise ValueError(
                    '\n MICE Predication Error: The prediction method {} output is not standardised.'.format(
                        self.predictMethod))
            return predictedY

        # MICE prediction method switch-case
        if self.predictMethod == 'norm.nob':
            method = linear_model.LinearRegression(fit_intercept=False)
        elif self.predictMethod == 'norm':
            method = linear_model.BayesianRidge(compute_score=True)
        elif self.predictMethod == 'lda':
            method = discriminant_analysis.LinearDiscriminantAnalysis()
        elif self.predictMethod == 'qda':
            method = discriminant_analysis.QuadraticDiscriminantAnalysis()
        elif self.predictMethod == 'polyreg':
            method = linear_model.LogisticRegression(random_state=0, solver='lbfgs',
                                                     multi_class='multinomial')
        elif self.predictMethod == 'logreg':
            method = linear_model.LogisticRegression(random_state=0, solver='sag',
                                                     multi_class='ovr')

        return modeler(method)

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

    def __place_holder(self, featureItem):
        featureName = self.data.columns.values.tolist()[featureItem]
        placeHolderColumnIndex = list(map(lambda x: 1 if x == featureItem else 0,
                                          self.missedValuesMap[1]))
        placeHolderRows = list(itertools.compress(self.missedValuesMap[0],
                                                  placeHolderColumnIndex))
        # Converting the rows coordinate to the data-frame Index before imputing the None
        placeHolderRowIndex = [self.data.index.tolist()[x] for x in placeHolderRows]
        if self.inplace:
            self.data.loc[placeHolderRowIndex,
                          featureItem] = None
            trainSubset = self.data.iloc[self.data[featureName].notnull()]
            testSubset = self.data[self.data[featureName].isnull()]

        else:
            self.imputedData.loc[placeHolderRowIndex,
                                 featureName] = None
            trainSubset = self.imputedData[self.imputedData[featureName].notnull()]
            testSubset = self.imputedData[self.imputedData[featureName].isnull()]
        self.trainSubsetX = trainSubset.drop(featureName, axis=1).copy()
        self.trainSubsetY = trainSubset[[featureName]].copy()
        self.testSubsetX = testSubset.drop(featureName, axis=1).copy()
        self.testSubsetY = testSubset[[featureName]].copy()

        return placeHolderRows

    def __imput(self, rowIndexes=None, predictedValues=None, columnIndex=None):
        if self.inplace:
            for [rowIndex, predictedVlaue] in zip(rowIndexes,
                                                  predictedValues):
                self.data.iat[rowIndex, columnIndex] = predictedVlaue[0]
        else:
            for [rowIndex, predictedVlaue] in zip(rowIndexes,
                                                  predictedValues):
                self.imputedData.iat[rowIndex, columnIndex] = predictedVlaue[0]

    def imputer(self, imputMethod='norm.nob'):

        def __plot_conversion(missingValueIndex=0):
            plt.plot(list(range(0, self.iteration)),
                     self.iterationLog[:, missingValueIndex],
                     'bo',
                     list(range(0, self.iteration)),
                     self.iterationLog[:, missingValueIndex],
                     'k')
            plt.axis([0,
                      self.iteration,
                      np.min(self.iterationLog[:, missingValueIndex]) - 1,
                      np.max(self.iterationLog[:, missingValueIndex]) + 1])
            plt.ylabel('Iteration')
            plt.show()

        featureWithNone = set(self.missedValuesMap[1])
        self.simple_imputation(imputMethod='imputMean')  # Step1: Mice
        iterations = iter(range(0, self.iteration))
        doneLooping = False
        while not doneLooping:
            try:
                iteration = next(iterations)
                print('The iteration {} is started:'.format(iteration + 1))
                imputeValuesOrdByCol = []
                for featureItem in featureWithNone:
                    rowIndexes = self.__place_holder(featureItem=featureItem)  # Step2: Mice
                    predictedValues = self.predictive_model()  # methodName='norm'
                    self.__imput(rowIndexes, predictedValues, featureItem)
                    print(predictedValues.ravel().tolist())
                    imputeValuesOrdByCol.append(list(predictedValues.flatten()))
            except StopIteration:
                doneLooping = True
            else:
                # Flatten the list of list ^ add to the iteration log
                self.iterationLog[iteration] = list(itertools.chain(*imputeValuesOrdByCol))
                print('-' * 100)
        table = DoubleTable(self.iterationLog.tolist())
        table.inner_heading_row_border = False
        table.justify_columns[1] = 'center'
        # print(table.table)
        __plot_conversion()


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
    print(obj.orginalData)
    obj()
    print(obj.imputedData)
    obj.missing_pattern_plot('heatmap')


def __test_me_iris():
    from sklearn import datasets
    from sklearn.metrics import r2_score
    import random

    data = datasets.load_iris().data[:, :4]
    data = pd.DataFrame(data, columns=['F1', 'F2', 'F3', 'F4'])
    data1=data.copy()
    x=[]
    y=[]
    old_value=[]
    mean_ind=data1.mean(axis = 0).values
    mean_list=[]

    for i in range(16):
        xv=random.randint(0,145)
        yv=random.randint(0,3)
        old_value.append(data1.iloc[xv, yv])
        mean_list.append(mean_ind[yv])
        data.iloc[xv,yv]=np.NaN
        x.append(xv)
        y.append(yv)

    obj = Mice(data,iteration=100)
    obj()

    pred=[]
    for i,j,v in zip(x,y,old_value):
        print(i,j,'--',v,'-',obj.imputedData.iloc[i,j])
        pred.append(obj.imputedData.iloc[i,j])

    obj.missing_pattern_plot('heatmap')
    print(r2_score(old_value, pred, multioutput = 'variance_weighted'))
    print(1-(1-r2_score(old_value, pred, multioutput = 'variance_weighted'))*(data1.shape[0]-1)/(data1.shape[0]-data1.shape[1]-1))
    print('-+'*30)
    print(r2_score(old_value, mean_list, multioutput = 'variance_weighted'))
    print(1-(1-r2_score(old_value, mean_list, multioutput = 'variance_weighted'))*(data1.shape[0]-1)/(data1.shape[0]-data1.shape[1]-1))

def __mice():
    start = time.time()
    try:
        args = arguments_parser()
        df, features_appending_list, columns_order, feature_list = __configoration(args['configPath'], args['csvPath'])
        obj = Mice(df, predictMethod=args['predict_method'], iteration=args['iteration'], feature_list=feature_list)
        obj()
        obj._write_csv(output_path=args['imputedPath'],
                       appendTo=features_appending_list,
                       csvPath=args['csvPath'],
                       order=columns_order)
        print("\033[32mThe missing value imputation process is successfully completed by MICE method.")
        return obj
    except AssertionError as error:
        print(error)
        print("\033[31mThe feature selection proses is failed.")
    finally:
        duration = time.time() - start
        print('\033[0mTotal duration is: %.3f' % duration)


if __name__ == '__main__':
    #__test_me_iris()
    __mice()


