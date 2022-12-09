# TODO: Add more method (optional)


from .base import FeatureSelection
from .spectral_methods import _BaseSpectralSelector, SPEC, MultiClusterScore, LaplacianScore
from .feature_analysis import _BaseFeatureAnalysis, IndependentFeatureAnalysis, PrincipalFeatureAnalysis

__all__ = ['FeatureSelection',
           '_BaseSpectralSelector',
           'SPEC',
           'MultiClusterScore',
           'LaplacianScore',
           '_BaseFeatureAnalysis',
           'IndependentFeatureAnalysis',
           'PrincipalFeatureAnalysis',
           '_BaseFeatureAnalysis',
           ]