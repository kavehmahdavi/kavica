#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" A text progressbar function


# Author: Kaveh Mahdavi <kavehmahdavi74@gmail.com>
# License: BSD 3 clause
# Last update: 04/10/2022
"""
import sys

__all__ = ['progressbar']


def progressbar(count, total, status='', bar_len=100, functionality=None):
    """ Shows and updates progressbar bar.

    Args:
        bar_len (int): Indicates the progressbar bar length.
        count (int): Indicates the actual value
        total (int): Indicates the overall value
        status (str): Shows the more related information about the progressbar bar
        functionality (str): Indicates the functionality of the bar
    Returns:

    """
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('\r\033[1;36;m[%s] %s%s ...%s - %s' % (bar, percents, '%', status, functionality))
    sys.stdout.flush()
