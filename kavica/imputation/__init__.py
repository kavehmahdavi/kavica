"""
The :mod:`kavica.imputation` module gathers popular imputation algorithms.

"""
from .mice import (MissingValuePreProcessing, Mice, scale_into_range, dict_inner_joint)
from .base import (compatible_data_structure)

__all__ = ['Mice',
           'MissingValuePreProcessing',
           'scale_into_range',
           'dict_inner_joint',
           'compatible_data_structure',]
