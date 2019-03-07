# ---------------------------------------------------------------------------------------
# the basic unary
#
# transformation function
# ---------------------------------------------------------------------------------------
import numpy as np
import math


def _identity(X):
    """The identity function.
    """
    return X


def testfunction(X):
    return X * 2


def myfunc(z):
    return 1 / (1 + math.exp(-z))


def testfunction2Vec(X):
    def testfunction2(X):
        return math.log2(X + 4)

    return np.vectorize(testfunction2)(X)


def testfunction3Vec(X):
    def testfunction3(X):
        return np.log10(X + 4)
    return np.vectorize(testfunction3)(X)
