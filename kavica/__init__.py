"""
HPC data science module for Python
==================================
kavica is a Python module HPC data mining algorithms in the tightly-knit world of
scientific & HPC Python packages (numpy, scipy, pandas, Multiprocess, Multithread, mpi4py).
It aims to provide High performance solutions to data mining problems
that are accessible to everybody and reusable in various contexts:
The especial empirical application of this package is HPC application analysis.
"""

from .distance_measure import check_input_compatible, rbf_kernel, euclidean_distance, mkl_blas_euclidean_matrix
from .mutual_knn import KNN, KnnMatrix
from .utils._bcolors import BColors
from .utils._progress_bar import progressbar
from .factor_analysis import factor_rotation

__all__ = ['BColors',
           'progressbar',
           'KNN',
           'KnnMatrix',
           'check_input_compatible',
           'rbf_kernel',
           'euclidean_distance',
           'mkl_blas_euclidean_matrix',
           'factor_rotation'
           ]
