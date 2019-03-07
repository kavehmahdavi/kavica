#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap and recompiling methods.

--------------------------------------------------------------------------------------------------------------------
References:
    -
--------------------------------------------------------------------------------------------------------------------
"""

# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause

import pandas as pd
import numpy as np

from multiprocessing import Process
from multiprocessing.managers import BaseManager


######################################################################
# Base class
######################################################################
class _BaseBootstrapping(object):
    pass


######################################################################
# Specific methods
######################################################################
class WightedBootstrapping(_BaseBootstrapping):

    @classmethod
    def wighted_resampeling(self, X, bag_size, weight=None, replace=True, bags=1):
        """ Re_sample from original data set/
        Parameters
        ----------
        X: pandas data frame, , shape = [n_samples, n_features].
        weight:.
        bag_size: the sample size.
        replace=True:.
        bags=1:.

        Returns
        -------
        dict: {bag_number(str): sampled_df(pandas dataframe)}
        """
        resampeld_bags = {}
        for bag in range(bags):
            resampeld_bags.update({str(bag): X.sample(n=bag_size,
                                                      weights=weight,
                                                      replace=replace)})
        return resampeld_bags


'''
def __test_me():
    data1 = np.array([(1, 1, 1, 1, 1, 1, 1),
                      (2, 2, 2, 2, 2, 2, 2),
                      (3, 4, 45, 23, 24, 19, 16),
                      (4, 2, 44, 23, 22, 13, 11),
                      (5, 2, 4, 3, 2, 1, 1),
                      (6, 1, 1, 1, 1, 1, 1),
                      (7, 2, 2, 2, 2, 2, 2),
                      (8, 2, 45, 23, 24, 13, 16),
                      (9, 12, 0, 9, 5, 20, 89),
                      (10, 6, 7, 8, 3, 8, 2)])

    headers = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    df = pd.DataFrame(data1, columns=headers, dtype=np.int)

    #rs = WightedBootstrapping()
    #q = rs.wighted_resampeling(X=df,bag_size=3, replace=False,bags=6)
    #print(q)

    s=WightedBootstrapping.wighted_resampeling(X=df,bag_size=4, replace=False,bags=1,weight='C')
    print(s)

if __name__ == '__main__':
    __test_me()
'''
