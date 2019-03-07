"""
The :mod:`kavica.transformer` module transformer is for building transformed
data_set /column with pre-defined function.
"""

from .BatchTransformer import (assert_allclose_dense_sparse,
                               TransformerFunction)
from .VerticalTransformer import VerticalTransformer
from .TransformingFunctions import (_identity,
                                    testfunction3Vec,
                                    testfunction2Vec,
                                    testfunction,
                                    myfunc)

__all__ = [
    'assert_allclose_dense_sparse',
    'TransformerFunction',
    'VerticalTransformer',
    '_identity',
    'testfunction',
    'myfunc',
    'testfunction2Vec',
    'testfunction3Vec'
]
