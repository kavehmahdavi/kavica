"""
The benchmark is :
https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/feature_selection
"""
# TODO: Add more method (optional)

from .base import FeaturSelectionMold
from .spectral_methods import _BaseSpectralSelector, SPEC, MultiClusterScore, LaplacianScore
from .feature_analysis import _BaseFeatureAnalysis, IndependentFeatureAnalysis, PrincipalFeatureAnalysis

__all__ = ['FeaturSelectionMold',
           '_BaseSpectralSelector',
           'SPEC',
           'MultiClusterScore',
           'LaplacianScore',
           '_BaseFeatureAnalysis',
           'IndependentFeatureAnalysis',
           'PrincipalFeatureAnalysis'
           ]
