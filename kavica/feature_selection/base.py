#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Generic feature selection

# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# Last update: 03/10/2022
"""

import abc
import warnings

__all__ = ['FeatureSelection']


class FeatureSelection(object):
    """ The base class of the feature selection

    """

    @abc.abstractmethod
    def output_mask(self, binary_mask=False, feature_vector=None, selected_features=None):
        """ It returns a selected feature index list/ a binary list of all features.

        Args:
            binary_mask(boolean= False)
                If True, the return will be a binary mask list,
                rather than a list of selected features.
            feature_vector(array=None):
                It is a list of the feature vector.
            selected_features(array=None):
                the list of selected features indexes.
        Returns:
            - array
                It will be an index list of selected features or
                a binary mask in size of the feature vector, where
                the value is 1 for selected feature rather than 0
                for omitted features.
        """

        @staticmethod
        def __binary_mask(originList, maskIndex):
            binaryOutputList = [0] * len(originList)
            for featurItem in maskIndex:
                binaryOutputList[featurItem] = 1
            return binaryOutputList

        if feature_vector is None:
            raise ValueError("The feature vector is empty.")
        elif selected_features is None:
            raise ValueError("No Features Selected.")
        elif binary_mask:
            return __binary_mask(feature_vector, selected_features)
        else:
            return selected_features

    @staticmethod
    def output_ranked_list(feature_vector=None, feature_ranks=None, sorted_list=True, descending=True):
        """ It returns a feature ranked list.(It's applicable only for feature ranking methods)

        args:
            feature_vector(array=None): It is a list of the feature vector.
            feature_ranks(array=None):the list of the feature ranks.
            sorted_list (boolean=True): if true, the list will be sorted by the rank
            descending (boolean=True): if true, the sorting will be descending.
        Return:
            - dictionary <feature,rank>
                a tuple of all feature vector items and their ranks.
        """
        if feature_vector is None:
            raise ValueError("The feature vector is empty.")
        elif feature_ranks is None:
            raise ValueError("No Features Selected.")
        elif len(feature_vector) != len(feature_ranks):
            raise ValueError("The feature vector({}) and feature rank({})  have not had the same size.".
                             fomat(len(feature_vector), len(feature_ranks)))
        else:
            rankedOutputList = dict(zip(feature_vector, feature_ranks))
            if not sorted_list:
                return rankedOutputList
            elif descending:
                return dict(sorted(rankedOutputList.items(), key=lambda value: value[1], reverse=True))
            else:
                return dict(sorted(rankedOutputList.items(), key=lambda value: value[1], reverse=False))

    @staticmethod
    def reconstruction(origin_dataset, selected_features=None):
        """ Drop the unselected features from originDataset.

        Args:
            origin_dataset (panda.dataframe[n_samples, n_features]): The original data_set.
            selected_features(array=None): the list of selected features indexes.

        Returns
            - RedactedDataset: numpy.array [n_samples, n_selected_features]
                The origin data with only the selected features.
        """
        if len(selected_features) == 0:
            selected_features = None
        if selected_features is None:
            warnings.warn("No features were selected", UserWarning)
            return origin_dataset[origin_dataset.columns[[]]]
        elif len(selected_features) > origin_dataset.shape[1]:
            raise ValueError("The selected feature are more than the original features")
        elif all(isinstance(selectedItem, str) for selectedItem in selected_features):
            return origin_dataset[selected_features]
        elif all(isinstance(selectedItem, int) for selectedItem in selected_features):
            return origin_dataset[origin_dataset.columns[[selected_features]]]
        else:
            raise ValueError("Reconstruction is only based on either column name or index.")

    def preprocessor(self):
        # TODO: write the pre processing
        pass
