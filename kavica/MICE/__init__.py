"""
The :mod:`kavica.imputation` module gathers popular imputation algorithms.
"""
from .mice import (MissingValuePreProcessing,
                   Mice)

from .base import (data_structure_Compatibilization)

__all__ = ['Mice',
           'MissingValuePreProcessing',
           'data_structure_Compatibilization']
