#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform all data-set based on unique function.
"""
# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# last update: 14/12/2018

import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.externals.six import string_types
import scipy as sp
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose as assert_allclose
import numpy as np
from kavica.transformer.TransformingFunctions import (testfunction,
                                                      _identity,
                                                      testfunction2Vec,
                                                      testfunction3Vec,
                                                      myfunc)

__all__ = [
    'assert_allclose_dense_sparse',
    'TransformerFunction'
]


# TODO: revised the methods and classes names'
# TODO: write the complete description of the methods
# TODO: Write the sub package description.
# --------------------------------------------------------------------------------------
def assert_allclose_dense_sparse(x, y, rtol=1e-5, atol=0, err_msg=''):
    """Assert allclose for sparse and dense data.
    Both x and y need to be either sparse or dense, they
    can't be mixed.
    Parameters
    ----------
    x : array-like or sparse matrix
        First array to compare.
    y : array-like or sparse matrix
        Second array to compare.
    rtol : float, optional
        relative tolerance; see numpy.allclose
    atol : float, optional
        absolute tolerance; see numpy.allclose. Note that the default here is
        more tolerant than the default for numpy.testing.assert_allclose, where
        atol=0.
    err_msg : string, default=''
        Error message to raise.
    """
    if sp.sparse.issparse(x) and sp.sparse.issparse(y):
        x = x.tocsr()
        y = y.tocsr()
        x.sum_duplicates()
        y.sum_duplicates()
        assert_array_equal(x.indices, y.indices, err_msg=err_msg)
        assert_array_equal(x.indptr, y.indptr, err_msg=err_msg)
        assert_allclose(x.data, y.data, rtol=rtol, atol=atol, err_msg=err_msg)
    elif not sp.sparse.issparse(x) and not sp.sparse.issparse(y):
        # both dense

        assert_allclose(x, y, rtol=rtol, atol=atol, err_msg=err_msg)
    else:
        raise ValueError("Can only compare two sparse matrices,"
                         " not a sparse matrix and an array.")


# TODO: It is ok, i just need to optimise it and kavicaizaid.
# TODO: it have to be independent module. (As .py file)
class TransformerFunction(BaseEstimator, TransformerMixin):
    """Constructs a transformer from an arbitrary callable.
    A FunctionTransformer forwards its X (and optionally y) arguments to a
    user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.
    Note: If a lambda is used as the function, then the resulting
    transformer will not be pickleable.
    .. versionadded:: 0.17
    Read more in the :ref:`User Guide <function_transformer>`.
    Parameters
    ----------
    func : callable, optional default=None
        The callable to use for the transformation. This wil be passed
        the same arguments as transform, with args and kwargs forwarded.
        If func is None, then func will be the identity function.
    inverse_func : callable, optional default=None
        The callable to use for the inverse transformation. This will be
        passed the same arguments as inverse transform, with args and
        kwargs forwarded. If inverse_func is None, then inverse_func
        will be the identity function.
    validate : bool, optional default=True
        Indicate that the input X array should be checked before calling
        ``func``. The possibilities are:
        - If False, there is no input validation.
        - If True, then X will be converted to a 2-dimensional NumPy array or
          sparse matrix. If the conversion is not possible an exception is
          raised.
        .. deprecated:: 0.20
           ``validate=True`` as default will be replaced by
           ``validate=False`` in 0.22.
    accept_sparse : boolean, optional
        Indicate that func accepts a sparse matrix as input. If validate is
        False, this has no effect. Otherwise, if accept_sparse is false,
        sparse matrix inputs will cause an exception to be raised.
    pass_y : bool, optional default=False
        Indicate that transform should forward the y argument to the
        inner callable.
        .. deprecated::0.19
    check_inverse : bool, default=True
       Whether to check that or ``func`` followed by ``inverse_func`` leads to
       the original inputs. It can be used for a sanity check, raising a
       warning when the condition is not fulfilled.
       .. versionadded:: 0.20
    kw_args : dict, optional
        Dictionary of additional keyword arguments to pass to func.
    inv_kw_args : dict, optional
        Dictionary of additional keyword arguments to pass to inverse_func.
    """

    def __init__(self, func=None, inverse_func=None, validate=None,
                 accept_sparse=False, pass_y='deprecated', check_inverse=True,
                 kw_args=None, inv_kw_args=None):
        self.func = func
        self.inverse_func = inverse_func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.pass_y = pass_y
        self.check_inverse = check_inverse
        self.kw_args = kw_args
        self.inv_kw_args = inv_kw_args

    def _check_input(self, X):
        # FIXME: Future warning
        if self.validate is None:
            self._validate = True
            warnings.warn("The default validate=True will be replaced by "
                          "validate=False in 0.22.", FutureWarning)
        else:
            self._validate = self.validate
        if self._validate:
            return check_array(X, accept_sparse=self.accept_sparse)
        return X

    def _check_inverse_transform(self, X):
        """Check that func and inverse_func are the inverse."""

        idx_selected = slice(None, None, max(1, X.shape[0] // 100))
        print(idx_selected)
        try:
            assert_allclose_dense_sparse(
                X[idx_selected],
                self.inverse_transform(self.transform(X[idx_selected])))
        except AssertionError:
            warnings.warn("The provided functions are not strictly"
                          " inverse of each other. If you are sure you"
                          " want to proceed regardless, set"
                          " 'check_inverse=False'.", UserWarning)

    def fit(self, X, y=None):
        """Fit transformer by checking X.
        If ``validate`` is ``True``, ``X`` will be checked.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        Returns
        -------
        self
        """
        X = self._check_input(X)
        if (self.check_inverse and not (self.func is None or
                                        self.inverse_func is None)):
            self._check_inverse_transform(X)
        return self

    def transform(self, X, y='deprecated'):
        """Transform X using the forward function.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        y : (ignored)
            .. deprecated::0.19
        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn("The parameter y on transform() is "
                          "deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)

        return self._transform(X, y=y, func=self.func, kw_args=self.kw_args)

    def inverse_transform(self, X, y='deprecated'):
        """Transform X using the inverse function.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input array.
        y : (ignored)
            .. deprecated::0.19
        Returns
        -------
        X_out : array-like, shape (n_samples, n_features)
            Transformed input.
        """
        if not isinstance(y, string_types) or y != 'deprecated':
            warnings.warn("The parameter y on inverse_transform() is "
                          "deprecated since 0.19 and will be removed in 0.21",
                          DeprecationWarning)
        return self._transform(X, y=y, func=self.inverse_func,
                               kw_args=self.inv_kw_args)

    def _transform(self, X, y=None, func=None, kw_args=None):
        X = self._check_input(X)

        if func is None:
            func = testfunction3Vec

        if (not isinstance(self.pass_y, string_types) or
                self.pass_y != 'deprecated'):
            # We do this to know if pass_y was set to False / True
            pass_y = self.pass_y
            warnings.warn("The parameter pass_y is deprecated since 0.19 and "
                          "will be removed in 0.21", DeprecationWarning)
        else:
            pass_y = False

        return func(X, *((y,) if pass_y else ()),
                    **(kw_args if kw_args else {}))


def main():
    data = np.array([(1, 9, 6, 13, 1, 72, 4),
                     (1, 9, 0, 13, 1, 12, 4),
                     (2, 2, 45, 23, 24, 13, 16),
                     (3, 12, 0, 9, 5, 20, 89)])
    data1 = [1, 9, 6, 13, 1, 72, 4]
    data2 = np.array([1, 9, 6, 13, 1, 72, 4])

    tr = [("norm1", 'f1', [0, 1]), ("norm2", 'f2', slice(2, 4))]
    b = TransformerFunction()
    b.fit(data)
    print(b.transform(data))


if __name__ == '__main__':
    main()
