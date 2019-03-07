"""
The :mod:`kavica.factor_analysis` module gathers popular factor analysis
algorithms.
"""

# TODO: add more factor analysis the
from .factor_rotation import (Rotatin, ObliqueRotation,
                              OrthogonalRotation, _normalize_numpy)

__all__ = ['Rotatin',
           'ObliqueRotation',
           'OrthogonalRotation',
           '_normalize_numpy']
