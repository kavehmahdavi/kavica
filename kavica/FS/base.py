# -*- coding: utf-8 -*-
"""Universal feature selection mold"""

# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause

import abc
import warnings

__all__ = ['FeaturSelectionMold']


class FeaturSelectionMold(object):

    @abc.abstractmethod
    def output_mask(self, binaryMask=False, featureVector=None, selectedFeatures=None):
        """ It return a selected feature index list/ a binary list of all features.

        Parameters:
            - binaryMask(boolean= False)
                If True, the return will be a binary mask list,
                rather than a list of selected features.
            - featureVector(array=None):
                It is a list of the feature vector.
            - selectedFeatures(array=None):
                the list of selected features indexes.
        Return:
            - array
                It will be an index list of selected features or
                a binary mask in size of the feature vector, where
                the value is 1 for selected feature rather than 0
                fro omitted features.
        """

        @staticmethod
        def __binary_mask(originList, maskIndex):
            binaryOutputList = [0] * len(originList)
            for featurItem in maskIndex:
                binaryOutputList[featurItem] = 1
            return binaryOutputList

        if featureVector is None:
            raise ValueError("The feature vector is empty.")
        elif selectedFeatures is None:
            raise ValueError("No Features Selected.")
        elif binaryMask:
            return __binary_mask(featureVector, selectedFeatures)
        else:
            return selectedFeatures

    def output_ranked_list(self, featureVector=None, featureRanks=None,
                           sortedList=True, descending=True):
        """ It return a feature ranked list.(It's applicable only for feature ranking methods)

        Parameters:
            - featureVector(array=None):
                It is a list of the feature vector.
            - featureRanks(array=None):
                the list of the feature ranks.
            - sorted (boolean=True): if true, the list will be sorted by the rank
            - descending (boolean=True): if true, the sorting will be descending.
        Return:
            - dictionary <feature,rank>
                a tuple of all feature vector items and their ranks.
        """
        if featureVector is None:
            raise ValueError("The feature vector is empty.")
        elif featureRanks is None:
            raise ValueError("No Features Selected.")
        elif len(featureVector) != len(featureRanks):
            raise ValueError("The feature vector({}) and feature rank({})  have not had the same size.".
                             fomat(len(featureVector), len(featureRanks)))
        else:
            rankedOutputList = dict(zip(featureVector, featureRanks))
            if not sortedList:
                return rankedOutputList
            elif descending:
                return dict(sorted(rankedOutputList.items(), key=lambda value: value[1], reverse=True))
            else:
                return dict(sorted(rankedOutputList.items(), key=lambda value: value[1], reverse=False))

    def reconstruction(self, originDataset, selectedFeatures=None):
        """ Drop the unselected features from originDataset.

        Parameters
        ----------
            - originDataset (panda.dataframe[n_samples, n_features])
                The original data_set.
            - selectedFeatures(array=None):
                the list of selected features indexes.

        Returns
        -------
            - RedactedDataset: numpy.array [n_samples, n_selected_features]
                The origin data with only the selected features.
        """
        if len(selectedFeatures) == 0:
            selectedFeatures = None
        if selectedFeatures is None:
            warnings.warn("No features were selected", UserWarning)
            return originDataset[originDataset.columns[[]]]
        elif len(selectedFeatures) > originDataset.shape[1]:
            raise ValueError("The selected feature are more than the original features")
        elif all(isinstance(selectedItem, str) for selectedItem in selectedFeatures):
            return originDataset[selectedFeatures]
        elif all(isinstance(selectedItem, int) for selectedItem in selectedFeatures):
            return originDataset[originDataset.columns[[selectedFeatures]]]
        else:
            raise ValueError("Reconstruction is only based on either column name or index.")

    def preprocessor(self):
        # TODO: write the pre processing
        pass
