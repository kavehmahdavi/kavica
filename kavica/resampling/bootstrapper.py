#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Bootstrap and recompiling methods.

Author: Kaveh Mahdavi <kavehmahdavi74@yahoo.com>
License: BSD 3 clause
last update: 03/10/2022
"""


class WightedBootstrapping(object):
    """ Return a random sample of items from an axis of object

    """
    @classmethod
    def weighted_resampling(cls, x, bag_size, weight=None, replace=True, bags=1):
        """  Resample from original dataset

        Args:
            x (pandas): indicates the original dataset shape = [n_samples, n_features].
            weight (str or ndarray-like): Default ‘None’ results in equal probability weighting
            bag_size (int): the sample size
            replace (boolean): It allows or disallow sampling of the same row more than once.
            bags (int): The number of the pages

        Returns:
            A dict {bag_number(str): sampled_df(pandas dataframe)}
        """
        _bags = {}
        for bag in range(bags):
            _bags.update({str(bag): x.sample(n=bag_size, weights=weight, replace=replace)})
        return _bags


"""
def __test_me():
    import numpy as np
    import pandas as pd
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

    rs = WightedBootstrapping()
    q = rs.weighted_resampling(x=df,bag_size=3, replace=False,bags=6)
    print(q)

if __name__ == '__main__':
    __test_me()
"""
