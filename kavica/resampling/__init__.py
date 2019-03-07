"""
The benchmark is :
https://github.com/scikit-learn/scikit-learn/tree/master/sklearn/feature_selection
"""
# TODO: Add more method (optional)

from .bootstrapper import _BaseBootstrapping, WightedBootstrapping

__all__ = ['WightedBootstrapping',
           '_BaseBootstrapping'
           ]
