#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Engineering features for HPC performance analysis

This module will be included a set of functionsto engineer new feature from the original HWCs.
The feature engineering process is:
    1. Brainstorming or testing features.
    2. Deciding what features to create.
    3. Creating features.
    4. Checking how the features work with your model.
    5. Improving your features if needed.
    6. Go back to brainstorming/creating more features until the work is done.

The Functions:
    - burst_overlie= The amount of overlie between a CPU burst and other CPU bursts

See:
    - Mahdavi, K., Mancho, J. L., & Lucas, J. G. (2021, July). Organization Component Analysis: The method for
      extracting insights from the shape of cluster. In 2021 International Joint Conference on Neural Networks
      (IJCNN) (pp. 1-10). IEEE.

Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
License: BSD 3 clause
Last update: 04/10/2022
"""
import pandas as pd

__all__ = ['burst_overlying_factor']


def burst_overlying_factor(data):
    """ Compute CPU burst overlying

    CPU burst overlying factor is proportion of time that a CPU burst works with the overlapping.
    Where:
        T = ∆t_1 + ∆t_2 + ... + ∆t_n
        (Overlying Factor) O_i = The umber of threads that work in interval ∆_i
        n= Number of threads
    Linear Formula:
        OI(burst_j)=Σ(∆t_i * O_i)/T, i ∈ [1,n]
        1 ≤ OI(burst_j) ≤ n
    Args:
        data (pandas): Includes the input data set that is a Multiple time series.

    Returns:
        A pandas.core.series.Series includes all the Cpu bursts overlying factor.
    """

    # Fixme: The computation is slow. It needs to  be improved
    # Fixme: There is some budges in the computation
    def __overlie(reference_burst):
        """ Compute the cumulative overlying index for a single burst.

        Args:
            reference_burst (pandas.core.series.Series): includes a single Cpu burst data.

        Returns:
            A float indicates the overlying index for the Cpu burst.
        """

        matches = data.apply(lambda row: max(0, min(row['End_Time'], reference_burst['End_Time']) -
                                             max(row['Begin_Time'], reference_burst['Begin_Time'])), axis=1)

        return sum(matches) / (reference_burst['End_Time'] - reference_burst['Begin_Time'])

    return data.apply(lambda row: __overlie(row), axis=1)


def __test_me(x):
    print(burst_overlying_factor(x))


if __name__ == '__main__':
    df = pd.DataFrame({'Begin_Time': [1, 4, 2, 6, 4, 3], 'End_Time': [2, 6, 4, 8, 6, 8]})
    __test_me(df)
